"""
Post-hoc CFY adaptors for benchmark prediction outputs.

This module applies the model-agnostic CFY plugin to existing benchmark
predictions written as:

    all_predictions.json
    gene_names.json

The adaptor is intentionally non-invasive: base model scripts remain unchanged,
and CFY is trained as a residual enhancer on top of dual perturbation outputs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import CFYPluginMixin

logger = logging.getLogger(__name__)
PLUGIN_DIR = Path(__file__).resolve().parent
REPO_DIR = PLUGIN_DIR.parent
BENCHMARK_DIR = REPO_DIR / "benchmark"

IGNORE_INDEX = -100
LABEL_MAP = {
    "Additive": 0,
    "Synergy": 1,
    "Buffering": 2,
    "Opposite": 3,
    "Other": 4,
    # Robustness for alternative capitalization
    "additive": 0,
    "synergy": 1,
    "buffering": 2,
    "opposite": 3,
    "other": 4,
}


def _benchmark_path(*parts: str) -> Path:
    return BENCHMARK_DIR.joinpath(*parts)


def _split_condition_tokens(condition: str) -> List[str]:
    """Split benchmark condition names that may use '+' or '_' between genes."""
    if "+" in condition:
        return [token for token in condition.split("+") if token]
    if "_" in condition:
        return [token for token in condition.split("_") if token]
    return [condition] if condition else []


def _condition_separators(condition: str) -> List[str]:
    separators: List[str] = []
    if "+" in condition:
        separators.append("+")
    if "_" in condition:
        separators.append("_")
    if not separators:
        separators.append("+")
    return separators


def _condition_variants(condition: str) -> List[str]:
    """
    Enumerate plausible condition spellings across benchmark outputs.

    GEARS commonly writes dual perturbations as ``A_B`` while datasets/splits
    store them as ``A+B``. We also tolerate swapped dual order (``B+A`` /
    ``B_A``).
    """
    tokens = _split_condition_tokens(condition)
    if not tokens:
        return [condition]

    variants: List[str] = []
    non_ctrl_tokens = [token for token in tokens if token != "ctrl"]

    orders = [tokens]
    if len(tokens) == 2:
        orders.append([tokens[1], tokens[0]])

    for separator in _condition_separators(condition):
        for order in orders:
            candidate = separator.join(order)
            if candidate not in variants:
                variants.append(candidate)

    # Also check the alternative separator if the input only used one.
    if len(tokens) == 2:
        for separator in ["+", "_"]:
            for order in orders:
                candidate = separator.join(order)
                if candidate not in variants:
                    variants.append(candidate)

    # GEARS single-perturbation outputs often drop '+ctrl' and use only the gene name.
    if len(non_ctrl_tokens) == 1:
        single_gene = non_ctrl_tokens[0]
        if single_gene not in variants:
            variants.append(single_gene)
        for separator in ["+", "_"]:
            for order in ([single_gene, "ctrl"], ["ctrl", single_gene]):
                candidate = separator.join(order)
                if candidate not in variants:
                    variants.append(candidate)

    return variants


def parse_condition_genes(condition: str) -> List[str]:
    """Return non-control genes from a perturbation condition string."""
    return [gene for gene in _split_condition_tokens(condition) if gene and gene != "ctrl"]


def swap_condition_order(condition: str) -> str:
    """Swap A+B to B+A. Single perturbations are returned unchanged."""
    genes = _split_condition_tokens(condition)
    if len(genes) != 2:
        return condition
    separator = _condition_separators(condition)[0]
    return f"{genes[1]}{separator}{genes[0]}"


def resolve_condition_name(condition: str, available: Iterable[str]) -> Optional[str]:
    """Resolve a condition against available names across '+'/'_' spellings."""
    available_set = set(available)
    for candidate in _condition_variants(condition):
        if candidate in available_set:
            return candidate
    return None


def _build_aligned_embedding_matrix(
    output_gene_names: Sequence[str],
    embedding_table: Dict[str, np.ndarray],
    source_name: str,
) -> np.ndarray:
    if not embedding_table:
        raise ValueError(f"{source_name} embedding table is empty.")

    first_vector = next(iter(embedding_table.values()))
    embedding_dim = int(first_vector.shape[0])
    matrix = np.zeros((len(output_gene_names), embedding_dim), dtype=np.float32)
    found = 0
    for idx, gene_name in enumerate(output_gene_names):
        vector = embedding_table.get(str(gene_name))
        if vector is not None:
            matrix[idx] = vector.astype(np.float32, copy=False)
            found += 1

    if found == 0:
        raise ValueError(
            f"Could not align any output genes to the {source_name} embedding table."
        )

    logger.info(
        "Aligned %s/%s output genes to %s embeddings (dim=%s)",
        found,
        len(output_gene_names),
        source_name,
        embedding_dim,
    )
    return matrix


def _load_scfoundation_embedding_matrix(output_gene_names: Sequence[str]) -> np.ndarray:
    import anndata as ad

    singlecell_model_path = Path(
        os.getenv(
            "PPB_SCF_CKPT",
            str(REPO_DIR / "scFoundation" / "model" / "models" / "models.ckpt"),
        )
    )
    if not singlecell_model_path.exists():
        raise FileNotFoundError(
            f"scFoundation checkpoint not found: {singlecell_model_path}"
        )

    checkpoint = torch.load(singlecell_model_path, map_location="cpu")
    embedding_matrix = checkpoint["gene"]["state_dict"]["model.pos_emb.weight"].detach().cpu().numpy().astype(np.float32)

    gene_names: Optional[List[str]] = None
    # 1) Best-effort from checkpoint payload (if provided by ckpt producer).
    for key in ("gene_names", "genes", "vocab", "gene_vocab"):
        maybe = checkpoint.get(key)
        if isinstance(maybe, (list, tuple)) and len(maybe) > 0:
            gene_names = [str(x) for x in maybe]
            break

    # 2) Prefer official 19264 index table shipped with scFoundation.
    if gene_names is None:
        index_candidates = [
            Path(os.getenv("PPB_SCF_GENE_INDEX", "")).expanduser() if os.getenv("PPB_SCF_GENE_INDEX") else None,
            REPO_DIR / "scFoundation" / "OS_scRNA_gene_index.19264.tsv",
        ]
        for idx_path in index_candidates:
            if idx_path is None:
                continue
            if idx_path.exists():
                lines = [ln.strip() for ln in idx_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                # first column stores the symbol used by the model vocabulary
                parsed = [ln.split("\t")[0] for ln in lines]
                if len(parsed) > 0:
                    gene_names = parsed
                    break

    # 3) Fallback to demo.h5ad if index is unavailable.
    if gene_names is None:
        demo_adata_env = os.getenv("PPB_SCF_DEMO_H5AD", "").strip()
        if demo_adata_env:
            demo_adata_path = Path(demo_adata_env)
        else:
            demo_candidates = [
                REPO_DIR / "scFoundation" / "GEARS" / "data" / "demo" / "demo.h5ad",
                REPO_DIR / "scFoundation" / "model" / "data" / "demo.h5ad",
                _benchmark_path("data", "models", "scfoundation", "demo.h5ad"),
            ]
            demo_adata_path = next((p for p in demo_candidates if p.exists()), demo_candidates[0])

        if not demo_adata_path.exists():
            raise FileNotFoundError(
                "Could not resolve scFoundation gene vocabulary. Tried checkpoint keys, "
                "OS_scRNA_gene_index.19264.tsv, and demo.h5ad fallback."
            )
        demo_adata = ad.read_h5ad(demo_adata_path)
        if "gene_name" in demo_adata.var.columns:
            gene_names = demo_adata.var["gene_name"].astype(str).tolist()
        else:
            gene_names = demo_adata.var_names.astype(str).tolist()

    gene_names = gene_names + ["log10TotalCount1", "log10TotalCount2", "<pad>"]
    embedding_table = {
        gene_name: embedding_matrix[idx]
        for idx, gene_name in enumerate(gene_names)
        if idx < embedding_matrix.shape[0]
    }
    return _build_aligned_embedding_matrix(output_gene_names, embedding_table, "scFoundation")


def _load_scgpt_embedding_matrix(output_gene_names: Sequence[str]) -> np.ndarray:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    load_model = Path(
        os.getenv(
            "PPB_SCGPT_MODEL_DIR",
            _benchmark_path("data", "models", "scgpt", "scGPT_human"),
        )
    )
    model_config_file = load_model / "args.json"
    model_file = load_model / "best_model.pt"
    vocab_file = load_model / "vocab.json"
    if not model_config_file.exists() or not model_file.exists() or not vocab_file.exists():
        raise FileNotFoundError(
            f"scGPT model files not found under {load_model}. "
            "Expected args.json, best_model.pt, vocab.json."
        )

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)

    with model_config_file.open("r", encoding="utf8") as handle:
        model_configs = json.load(handle)

    model = TransformerModel(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        vocab=vocab,
        pad_value=-2,
        n_input_bins=51,
    )

    try:
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
    except Exception:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location="cpu")
        pretrained_dict = {
            key: value
            for key, value in pretrained_dict.items()
            if key in model_dict and value.shape == model_dict[key].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()
    gene_to_idx = vocab.get_stoi()
    gene_ids = np.asarray(list(gene_to_idx.values()))
    gene_names = list(gene_to_idx.keys())
    with torch.no_grad():
        gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long))
    embedding_table = {
        gene_name: gene_embeddings[idx].detach().cpu().numpy().astype(np.float32)
        for idx, gene_name in enumerate(gene_names)
    }
    return _build_aligned_embedding_matrix(output_gene_names, embedding_table, "scGPT")


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _geneformer_resource_roots() -> list[Path]:
    roots = []
    env_dir = os.getenv("PPB_GENEFORMER_SOURCE_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.extend(
        [
            _benchmark_path("tmp", "geneformer_src_hf"),
            _benchmark_path("tmp", "geneformer_src_hf", "geneformer"),
        ]
    )
    return roots


def _resolve_geneformer_resource(filename: str) -> Path:
    for root in _geneformer_resource_roots():
        candidate = root / filename
        if candidate.exists() and not _looks_like_git_lfs_pointer(candidate):
            return candidate
    raise FileNotFoundError(
        f"Geneformer resource not found: {filename}. "
        "Set PPB_GENEFORMER_SOURCE_DIR or run git lfs pull in benchmark/tmp/geneformer_src_hf."
    )


def _resolve_geneformer_mapping_file() -> Path:
    env_path = os.getenv("PPB_GENEFORMER_MAPPING_PKL")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists() and not _looks_like_git_lfs_pointer(candidate):
            return candidate
        raise FileNotFoundError(f"Invalid Geneformer mapping file: {candidate}")

    local_candidates = [
        _benchmark_path("data", "models", "geneformer", "ensembl_mapping_dict_gc104M.pkl"),
        _benchmark_path("data", "models", "geneformer", "ensembl_mapping_dict_gc95M.pkl"),
    ]
    for candidate in local_candidates:
        if candidate.exists() and not _looks_like_git_lfs_pointer(candidate):
            return candidate

    for filename in ["ensembl_mapping_dict_gc104M.pkl", "ensembl_mapping_dict_gc95M.pkl"]:
        try:
            return _resolve_geneformer_resource(filename)
        except FileNotFoundError:
            continue

    raise FileNotFoundError("Geneformer mapping file was not found.")


def _resolve_geneformer_model_dir() -> Path:
    env_dir = os.getenv("PPB_GENEFORMER_MODEL_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            _benchmark_path("data", "models", "geneformer"),
            _benchmark_path("data", "models", "geneformer", "gf-12L-95M-i4096"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Geneformer model directory was not found.")


def _load_geneformer_embedding_matrix(output_gene_names: Sequence[str]) -> np.ndarray:
    from transformers import AutoModel

    model_dir = _resolve_geneformer_model_dir()
    token_dict_path = _resolve_geneformer_resource("token_dictionary_gc104M.pkl")
    mapping_path = _resolve_geneformer_mapping_file()

    with token_dict_path.open("rb") as handle:
        token_dict = pickle.load(handle)
    with mapping_path.open("rb") as handle:
        gene_to_ensembl = pickle.load(handle)

    model = AutoModel.from_pretrained(model_dir)
    word_embeddings = (
        model.embeddings.word_embeddings.weight.detach().cpu().numpy().astype(np.float32)
    )

    embedding_matrix = np.zeros((len(output_gene_names), word_embeddings.shape[1]), dtype=np.float32)
    found = 0
    for idx, gene_name in enumerate(output_gene_names):
        ensembl_id = str(gene_name)
        if ensembl_id not in token_dict:
            ensembl_id = gene_to_ensembl.get(str(gene_name))
        if ensembl_id is None:
            continue
        token_id = token_dict.get(ensembl_id)
        if token_id is None or token_id >= word_embeddings.shape[0]:
            continue
        embedding_matrix[idx] = word_embeddings[token_id]
        found += 1

    if found == 0:
        raise ValueError("Could not align any output genes to Geneformer token embeddings.")

    logger.info(
        "Aligned %s/%s output genes to Geneformer embeddings (dim=%s)",
        found,
        len(output_gene_names),
        word_embeddings.shape[1],
    )
    return embedding_matrix


def load_pretrained_gene_embedding_matrix(
    output_gene_names: Sequence[str],
    embedding_source: str,
) -> Optional[np.ndarray]:
    source = embedding_source.lower()
    if source in {"learned", "gene_index", "none"}:
        return None
    if source == "scfoundation":
        return _load_scfoundation_embedding_matrix(output_gene_names)
    if source == "scgpt":
        return _load_scgpt_embedding_matrix(output_gene_names)
    if source == "geneformer":
        return _load_geneformer_embedding_matrix(output_gene_names)
    raise ValueError(
        f"Unsupported embedding_source={embedding_source}. "
        "Use learned, scfoundation, scgpt, or geneformer."
    )


def load_condition_label_arrays(csv_path: str, num_genes: int) -> Dict[str, np.ndarray]:
    """
    Load gene-level coeffect labels into per-condition dense arrays.

    Arrays use IGNORE_INDEX for genes without an annotation.
    """
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        if "CoEffect_Type" not in df.columns:
            raise ValueError(
                f"Expected either 'label' or 'CoEffect_Type' in {csv_path}."
            )
        df["label"] = df["CoEffect_Type"].map(LABEL_MAP)

    if "gene_idx" not in df.columns:
        raise ValueError(f"Expected 'gene_idx' column in {csv_path}.")

    condition_to_labels: Dict[str, np.ndarray] = {}
    for condition, group in df.groupby("condition"):
        labels = np.full(num_genes, IGNORE_INDEX, dtype=np.int64)
        valid = group["gene_idx"].between(0, num_genes - 1) & group["label"].notna()
        for row in group.loc[valid, ["gene_idx", "label"]].itertuples(index=False):
            labels[int(row.gene_idx)] = int(row.label)
        condition_to_labels[condition] = labels

    logger.info(
        "Loaded gene-level CFY labels for %s conditions from %s",
        len(condition_to_labels),
        csv_path,
    )
    return condition_to_labels


class PredictionCFYAdaptor(CFYPluginMixin):
    """
    Residual CFY adaptor over per-condition gene-expression predictions.

    The adaptor treats each (condition, target_gene) pair as one sample.
    The target-gene feature is a learnable embedding indexed by the prediction
    output order, while the two perturbation genes share the same embedding
    table for semantic routing.
    """

    def __init__(
        self,
        gene_names: Sequence[str],
        gene_embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
        device: str = "cpu",
        pretrained_gene_embeddings: Optional[np.ndarray] = None,
        trainable_gene_embeddings: bool = True,
        cfy_config: Optional[Dict[str, object]] = None,
    ):
        self.device = torch.device(device)
        self.gene_names = [str(gene) for gene in gene_names]
        self.num_genes = len(self.gene_names)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        if pretrained_gene_embeddings is not None:
            pretrained_gene_embeddings = np.asarray(pretrained_gene_embeddings, dtype=np.float32)
            if pretrained_gene_embeddings.shape[0] != self.num_genes:
                raise ValueError(
                    "pretrained_gene_embeddings row count must match output gene count: "
                    f"{pretrained_gene_embeddings.shape[0]} vs {self.num_genes}"
                )
            self.embedding_dim = int(pretrained_gene_embeddings.shape[1])
        else:
            self.embedding_dim = int(gene_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        config = dict(cfy_config or {})
        config.setdefault("hidden_dim", self.hidden_dim)
        config.setdefault("num_classes", self.num_classes)
        config.setdefault("dropout", dropout)

        CFYPluginMixin.__init__(self, cfy_config=config)

        if pretrained_gene_embeddings is not None:
            self.gene_embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_gene_embeddings, dtype=torch.float32),
                freeze=not trainable_gene_embeddings,
            )
            logger.info(
                "Initialized CFY gene embeddings from backbone features: dim=%s trainable=%s",
                self.embedding_dim,
                trainable_gene_embeddings,
            )
        else:
            self.gene_embedding = nn.Embedding(self.num_genes, self.embedding_dim)
            nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
            logger.info(
                "Initialized CFY gene embeddings from scratch: dim=%s",
                self.embedding_dim,
            )

        self.setup_cfy(
            input_dim=1 + (3 * self.embedding_dim),
            hidden_dim=self.hidden_dim,
        )

        default_class_weights = [1.0, 2.5, 3.0, 2.0, 1.5][: self.num_classes]
        class_weights_list = config.get("class_weights", default_class_weights)
        if len(class_weights_list) != self.num_classes:
            raise ValueError(
                "class_weights length must match num_classes: "
                f"{len(class_weights_list)} vs {self.num_classes}"
            )
        class_weights = torch.tensor(
            class_weights_list,
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        reg_class_weights_list = config.get("regression_class_weights")
        if reg_class_weights_list is None:
            reg_class_weights_list = [1.0] * self.num_classes
        if len(reg_class_weights_list) != self.num_classes:
            raise ValueError(
                "regression_class_weights length must match num_classes: "
                f"{len(reg_class_weights_list)} vs {self.num_classes}"
            )
        reg_class_weights = torch.tensor(
            reg_class_weights_list,
            dtype=torch.float32,
        )
        self.register_buffer("regression_class_weights", reg_class_weights)

        self.cfy_cls_loss_weight = float(config.get("cls_loss_weight", 0.35))
        self.cfy_additive_anchor_weight = float(config.get("additive_anchor_weight", 0.40))
        self.cfy_cls_focal_gamma = float(config.get("cls_focal_gamma", 2.0))
        self.cfy_unlabeled_reg_weight = float(config.get("unlabeled_reg_weight", 1.0))

        self.to(self.device)

    def extract_interaction_features(self, batch_data: Dict[str, Tensor]) -> Tensor:
        target_gene_idx = batch_data["target_gene_idx"]
        target_gene_emb = self.gene_embedding(target_gene_idx)
        base_predictions = batch_data["base_predictions"].unsqueeze(1)
        return torch.cat([base_predictions, target_gene_emb], dim=1)

    def identify_dual_perturbations(
        self,
        batch_data: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        is_dual = batch_data["is_dual"]
        return is_dual, ~is_dual

    def get_perturbation_pair_embeddings(
        self,
        batch_data: Dict[str, Tensor],
    ) -> Optional[Tuple[Tensor, Tensor]]:
        p1_emb = self.gene_embedding(batch_data["p1_idx"])
        p2_emb = self.gene_embedding(batch_data["p2_idx"])
        return p1_emb, p2_emb

    def _get_batch_size(self, batch_data: Dict[str, Tensor]) -> int:
        return int(batch_data["target_gene_idx"].shape[0])

    def _get_device(self, batch_data: Dict[str, Tensor]) -> torch.device:
        return self.device

    def _condition_label_array(
        self,
        condition: str,
        label_arrays: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        labels = np.full(self.num_genes, IGNORE_INDEX, dtype=np.int64)
        if not label_arrays:
            return labels

        resolved = resolve_condition_name(condition, label_arrays.keys())
        if resolved is not None:
            return label_arrays[resolved]
        return labels

    def _build_dual_batch(
        self,
        conditions: Sequence[str],
        base_predictions: Dict[str, Sequence[float]],
        truth_by_condition: Optional[Dict[str, np.ndarray]] = None,
        label_arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Tensor]:
        target_chunks: List[Tensor] = []
        p1_chunks: List[Tensor] = []
        p2_chunks: List[Tensor] = []
        base_chunks: List[Tensor] = []
        truth_chunks: List[Tensor] = []
        label_chunks: List[Tensor] = []
        valid_conditions: List[str] = []

        target_gene_idx = torch.arange(self.num_genes, device=self.device, dtype=torch.long)

        for condition in conditions:
            resolved = resolve_condition_name(condition, base_predictions.keys())
            if resolved is None:
                continue

            genes = parse_condition_genes(resolved)
            if len(genes) != 2:
                continue
            if genes[0] not in self.gene_to_idx or genes[1] not in self.gene_to_idx:
                continue

            base_vec = np.asarray(base_predictions[resolved], dtype=np.float32)
            if base_vec.shape[0] != self.num_genes:
                raise ValueError(
                    f"Prediction length mismatch for {resolved}: expected "
                    f"{self.num_genes}, got {base_vec.shape[0]}"
                )

            base_chunks.append(torch.from_numpy(base_vec).to(self.device))
            target_chunks.append(target_gene_idx)
            p1_chunks.append(
                torch.full(
                    (self.num_genes,),
                    self.gene_to_idx[genes[0]],
                    dtype=torch.long,
                    device=self.device,
                )
            )
            p2_chunks.append(
                torch.full(
                    (self.num_genes,),
                    self.gene_to_idx[genes[1]],
                    dtype=torch.long,
                    device=self.device,
                )
            )

            if truth_by_condition is not None:
                truth_resolved = resolve_condition_name(resolved, truth_by_condition.keys())
                truth_vec = None if truth_resolved is None else truth_by_condition.get(truth_resolved)
                if truth_vec is None:
                    raise ValueError(f"Truth vector missing for condition {resolved}")
                truth_chunks.append(
                    torch.from_numpy(np.asarray(truth_vec, dtype=np.float32)).to(self.device)
                )

            labels = self._condition_label_array(resolved, label_arrays)
            label_chunks.append(torch.from_numpy(labels).to(self.device))
            valid_conditions.append(resolved)

        if not valid_conditions:
            raise ValueError("No valid dual conditions were found for CFY processing.")

        batch = {
            "conditions": valid_conditions,
            "target_gene_idx": torch.cat(target_chunks, dim=0),
            "p1_idx": torch.cat(p1_chunks, dim=0),
            "p2_idx": torch.cat(p2_chunks, dim=0),
            "base_predictions": torch.cat(base_chunks, dim=0),
            "labels": torch.cat(label_chunks, dim=0),
            "is_dual": torch.ones(
                len(valid_conditions) * self.num_genes,
                dtype=torch.bool,
                device=self.device,
            ),
        }

        if truth_chunks:
            batch["truth"] = torch.cat(truth_chunks, dim=0)

        return batch

    def _focal_cross_entropy(self, logits: Tensor, labels: Tensor) -> Tensor:
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.cfy_cls_focal_gamma) * ce).mean()

    def _compute_training_loss(
        self,
        batch: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        predictions = outputs["predictions"].view(-1)
        truth = batch["truth"].view(-1)
        base_predictions = batch["base_predictions"].view(-1)

        squared_error = torch.square(predictions - truth)
        reg_weights = torch.full_like(squared_error, self.cfy_unlabeled_reg_weight)
        cls_loss = predictions.new_tensor(0.0)
        anchor_loss = predictions.new_tensor(0.0)

        labels = batch["labels"]
        labeled_mask = labels != IGNORE_INDEX
        if labeled_mask.any():
            labeled_logits = outputs["class_logits"][labeled_mask]
            labeled_targets = labels[labeled_mask]
            cls_loss = self._focal_cross_entropy(labeled_logits, labeled_targets)
            reg_weights[labeled_mask] = self.regression_class_weights[labeled_targets]

            additive_mask = labeled_targets == 0
            if additive_mask.any():
                labeled_predictions = predictions[labeled_mask]
                labeled_base_predictions = base_predictions[labeled_mask]
                anchor_loss = F.mse_loss(
                    labeled_predictions[additive_mask],
                    labeled_base_predictions[additive_mask],
                )

        reg_loss = (squared_error * reg_weights).sum() / reg_weights.sum().clamp_min(1e-8)

        total_loss = (
            reg_loss
            + (self.cfy_cls_loss_weight * cls_loss)
            + (self.cfy_additive_anchor_weight * anchor_loss)
        )
        return {
            "total": total_loss,
            "regression": reg_loss,
            "classification": cls_loss,
            "additive_anchor": anchor_loss,
        }

    def fit(
        self,
        base_predictions: Dict[str, Sequence[float]],
        truth_by_condition: Dict[str, np.ndarray],
        train_conditions: Sequence[str],
        val_conditions: Optional[Sequence[str]] = None,
        label_arrays: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        conditions_per_batch: int = 4,
        seed: int = 1,
    ) -> List[Dict[str, float]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

        dual_conditions = [
            condition
            for condition in train_conditions
            if resolve_condition_name(condition, base_predictions.keys()) is not None
            and len(parse_condition_genes(condition)) == 2
        ]
        dual_val_conditions = [
            condition
            for condition in (val_conditions or [])
            if resolve_condition_name(condition, base_predictions.keys()) is not None
            and len(parse_condition_genes(condition)) == 2
            and resolve_condition_name(condition, truth_by_condition.keys()) is not None
        ]

        if not dual_conditions:
            logger.warning("No trainable dual conditions were found. Skipping CFY fitting.")
            return []

        optimizer = torch.optim.Adam(
            list(self.get_cfy_parameters()) + [
                parameter
                for parameter in self.gene_embedding.parameters()
                if parameter.requires_grad
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

        history: List[Dict[str, float]] = []
        best_metric = float("inf")
        best_state_dict = deepcopy(self.state_dict())
        best_epoch = 0
        self.train()
        for epoch in range(1, epochs + 1):
            shuffled_conditions = dual_conditions[:]
            random.shuffle(shuffled_conditions)
            epoch_total = 0.0
            epoch_reg = 0.0
            epoch_cls = 0.0
            epoch_anchor = 0.0
            num_batches = 0

            for start in range(0, len(shuffled_conditions), conditions_per_batch):
                batch_conditions = shuffled_conditions[start : start + conditions_per_batch]
                batch = self._build_dual_batch(
                    batch_conditions,
                    base_predictions=base_predictions,
                    truth_by_condition=truth_by_condition,
                    label_arrays=label_arrays,
                )

                features = self.extract_interaction_features(batch)
                perturb_pair = self.get_perturbation_pair_embeddings(batch)
                outputs = self.cfy_forward(
                    features,
                    perturb_pair=perturb_pair,
                    base_predictions=batch["base_predictions"],
                    return_intermediate=True,
                )
                losses = self._compute_training_loss(batch, outputs)

                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

                epoch_total += float(losses["total"].detach().cpu())
                epoch_reg += float(losses["regression"].detach().cpu())
                epoch_cls += float(losses["classification"].detach().cpu())
                epoch_anchor += float(losses["additive_anchor"].detach().cpu())
                num_batches += 1

            summary = {
                "epoch": float(epoch),
                "loss_total": epoch_total / max(num_batches, 1),
                "loss_regression": epoch_reg / max(num_batches, 1),
                "loss_classification": epoch_cls / max(num_batches, 1),
                "loss_additive_anchor": epoch_anchor / max(num_batches, 1),
            }

            if dual_val_conditions:
                val_dual_mse = self._evaluate_dual_condition_mse(
                    dual_val_conditions,
                    base_predictions=base_predictions,
                    truth_by_condition=truth_by_condition,
                    conditions_per_batch=conditions_per_batch,
                )
                summary["val_dual_mse"] = val_dual_mse
                metric = val_dual_mse
            else:
                metric = summary["loss_total"]

            is_best = metric < best_metric
            summary["is_best"] = float(is_best)
            if is_best:
                best_metric = metric
                best_epoch = epoch
                best_state_dict = deepcopy(self.state_dict())

            history.append(summary)
            logger.info(
                "CFY epoch %s | total=%.6f reg=%.6f cls=%.6f anchor=%.6f%s%s",
                epoch,
                summary["loss_total"],
                summary["loss_regression"],
                summary["loss_classification"],
                summary["loss_additive_anchor"],
                f" val_dual_mse={summary['val_dual_mse']:.6f}" if "val_dual_mse" in summary else "",
                " [best]" if is_best else "",
            )

        self.load_state_dict(best_state_dict)
        logger.info(
            "Restored best CFY epoch %s with selection metric %.6f",
            best_epoch,
            best_metric,
        )
        return history

    def _evaluate_dual_condition_mse(
        self,
        conditions: Sequence[str],
        base_predictions: Dict[str, Sequence[float]],
        truth_by_condition: Dict[str, np.ndarray],
        conditions_per_batch: int = 4,
    ) -> float:
        self.eval()
        total_se = 0.0
        total_n = 0
        with torch.no_grad():
            for start in range(0, len(conditions), conditions_per_batch):
                batch_conditions = conditions[start : start + conditions_per_batch]
                batch = self._build_dual_batch(
                    batch_conditions,
                    base_predictions=base_predictions,
                    truth_by_condition=truth_by_condition,
                    label_arrays=None,
                )
                features = self.extract_interaction_features(batch)
                perturb_pair = self.get_perturbation_pair_embeddings(batch)
                outputs = self.cfy_forward(
                    features,
                    perturb_pair=perturb_pair,
                    base_predictions=batch["base_predictions"],
                    return_intermediate=True,
                )
                pred = outputs["predictions"].view(-1)
                truth = batch["truth"].view(-1)
                total_se += float(torch.square(pred - truth).sum().detach().cpu())
                total_n += int(pred.numel())
        self.train()
        if total_n == 0:
            return float("inf")
        return total_se / total_n

    def predict(
        self,
        base_predictions: Dict[str, Sequence[float]],
        conditions: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[float]]:
        target_conditions = list(conditions) if conditions is not None else list(base_predictions.keys())
        predictions: Dict[str, List[float]] = {}
        dual_conditions: List[str] = []

        for condition in target_conditions:
            resolved = resolve_condition_name(condition, base_predictions.keys())
            if resolved is None:
                continue
            genes = parse_condition_genes(resolved)
            if len(genes) == 2 and all(gene in self.gene_to_idx for gene in genes):
                dual_conditions.append(resolved)
            else:
                predictions[resolved] = list(np.asarray(base_predictions[resolved], dtype=np.float32))

        if dual_conditions:
            self.eval()
            with torch.no_grad():
                batch = self._build_dual_batch(
                    dual_conditions,
                    base_predictions=base_predictions,
                )
                features = self.extract_interaction_features(batch)
                perturb_pair = self.get_perturbation_pair_embeddings(batch)
                outputs = self.cfy_forward(
                    features,
                    perturb_pair=perturb_pair,
                    base_predictions=batch["base_predictions"],
                    return_intermediate=False,
                )
                flat_predictions = outputs.view(-1).detach().cpu().numpy()

            for idx, condition in enumerate(batch["conditions"]):
                start = idx * self.num_genes
                end = start + self.num_genes
                predictions[condition] = flat_predictions[start:end].tolist()

        return predictions
