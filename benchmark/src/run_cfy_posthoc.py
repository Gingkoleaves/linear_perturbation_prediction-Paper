from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Dict, Sequence
import uuid

import torch
import anndata as ad
import numpy as np
import session_info

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_DIR = BENCHMARK_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from cfy_plugin.posthoc import (
    PredictionCFYAdaptor,
    load_condition_label_arrays,
    load_pretrained_gene_embedding_matrix,
)

logger = logging.getLogger(__name__)


def _benchmark_path(*parts: str) -> Path:
    return BENCHMARK_DIR.joinpath(*parts)


def _build_arg_parser(default_model_name: str = "unknown") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply CFY post-hoc to benchmark predictions")
    parser.add_argument("--dataset_name", required=True, help="Benchmark dataset name")
    parser.add_argument("--test_train_config_id", required=True, help="Split JSON file under working_dir/results")
    parser.add_argument("--working_dir", required=True, help="Directory containing results/")
    parser.add_argument("--result_id", required=True, help="Output result directory id")
    parser.add_argument("--base_result_id", help="Base model result directory id under working_dir/results")
    parser.add_argument("--base_result_dir", help="Base model result directory path")
    parser.add_argument("--model_name", default=default_model_name, help="Metadata only")
    parser.add_argument("--label_csv", help="Optional gene-level coeffect CSV")
    parser.add_argument(
        "--disable_label_loss",
        action="store_true",
        help="Ignore gene-level coeffect labels even if a label CSV exists.",
    )
    parser.add_argument(
        "--preset",
        default="none",
        choices=[
            "none",
            "gears_gse220974_overall",
            "gears_gse220974_nonadd",
        ],
        help="Apply a validated hyperparameter preset before training.",
    )
    parser.add_argument(
        "--embedding_source",
        default="auto",
        help="learned, scfoundation, scgpt, geneformer, or auto",
    )
    parser.add_argument(
        "--freeze_gene_embeddings",
        action="store_true",
        help="Freeze loaded backbone gene embeddings instead of fine-tuning them",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--conditions_per_batch", type=int, default=4)
    parser.add_argument("--gene_embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cls_loss_weight", type=float, default=0.35)
    parser.add_argument("--additive_anchor_weight", type=float, default=0.40)
    parser.add_argument("--cls_focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--class_weights",
        default="1.0,2.5,3.0,2.0,1.5",
        help="Comma-separated class weights for Additive,Synergy,Buffering,Opposite,Other.",
    )
    parser.add_argument(
        "--regression_class_weights",
        default="1.0,1.0,1.0,1.0,1.0",
        help="Comma-separated regression weights for Additive,Synergy,Buffering,Opposite,Other.",
    )
    parser.add_argument(
        "--unlabeled_reg_weight",
        type=float,
        default=1.0,
        help="Regression weight for unlabeled targets.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _parse_args(default_model_name: str = "unknown") -> argparse.Namespace:
    return _build_arg_parser(default_model_name=default_model_name).parse_args()


def _resolve_base_result_dir(args: argparse.Namespace) -> Path:
    if args.base_result_dir:
        return Path(args.base_result_dir).resolve()
    if args.base_result_id:
        return (Path(args.working_dir) / "results" / args.base_result_id).resolve()
    raise ValueError("One of --base_result_dir or --base_result_id is required.")


def _load_prediction_result(result_dir: Path) -> tuple[Dict[str, Sequence[float]], list[str]]:
    prediction_path = result_dir / "all_predictions.json"
    gene_names_path = result_dir / "gene_names.json"
    if not prediction_path.exists() or not gene_names_path.exists():
        raise FileNotFoundError(
            f"Expected {prediction_path} and {gene_names_path} in base result directory."
        )

    with prediction_path.open("r", encoding="utf8") as handle:
        predictions = json.load(handle)
    with gene_names_path.open("r", encoding="utf8") as handle:
        gene_names = json.load(handle)

    return predictions, [str(gene) for gene in gene_names]


def _load_truth_adata(dataset_name: str) -> ad.AnnData:
    adata_path = _benchmark_path("data", "gears_pert_data", dataset_name, "perturb_processed.h5ad")
    if not adata_path.exists():
        raise FileNotFoundError(f"Dataset h5ad not found: {adata_path}")
    return ad.read_h5ad(adata_path)


def _adata_gene_names(adata: ad.AnnData) -> list[str]:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _mean_expression_by_condition(
    adata: ad.AnnData,
    output_gene_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    adata_gene_names = _adata_gene_names(adata)
    name_to_idx = {gene: idx for idx, gene in enumerate(adata_gene_names)}

    mapped_idx = np.asarray([name_to_idx.get(str(gene), -1) for gene in output_gene_names], dtype=np.int64)
    matched_mask = mapped_idx >= 0
    matched = int(matched_mask.sum())

    if matched == 0:
        if len(output_gene_names) == adata.shape[1]:
            logger.warning(
                "No output genes matched by name; falling back to positional alignment (length=%s).",
                len(output_gene_names),
            )
            mapped_idx = np.arange(adata.shape[1], dtype=np.int64)
            matched_mask = np.ones_like(mapped_idx, dtype=bool)
        else:
            raise ValueError(
                "Could not align output genes to dataset genes by name, and positional fallback "
                f"is impossible (output={len(output_gene_names)}, dataset={adata.shape[1]})."
            )
    elif matched < len(output_gene_names):
        logger.warning(
            "Partial gene-name alignment: matched %s/%s output genes; filling missing targets with 0.",
            matched,
            len(output_gene_names),
        )

    conditions = adata.obs["condition"].astype(str).tolist()
    unique_conditions = sorted(set(conditions))

    truth_by_condition: Dict[str, np.ndarray] = {}
    for condition in unique_conditions:
        mask = adata.obs["condition"].astype(str).values == condition
        mean_expr = np.asarray(adata[mask, :].X.mean(axis=0)).ravel().astype(np.float32)
        aligned = np.zeros((len(output_gene_names),), dtype=np.float32)
        aligned[matched_mask] = mean_expr[mapped_idx[matched_mask]]
        truth_by_condition[condition] = aligned

    return truth_by_condition


def _default_label_csv(dataset_name: str) -> Path:
    return _benchmark_path(
        "data",
        "gears_pert_data",
        dataset_name,
        "perturb_processed_with_coeffect_gene_level.csv",
    )


def _infer_embedding_source(model_name: str) -> str:
    name = model_name.lower()
    if "scgpt" in name:
        return "scgpt"
    if "scfoundation" in name:
        return "scfoundation"
    if "geneformer" in name:
        return "geneformer"
    return "learned"


def _make_local_tmp_output_dir(output_parent: Path, result_id: str) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_parent / f".tmp-{result_id}-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_class_weights(raw: str) -> list[float]:
    weights = [float(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not weights:
        raise ValueError("--class_weights must contain at least one numeric value.")
    return weights


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "none":
        return args

    if args.preset == "gears_gse220974_overall":
        args.embedding_source = "learned"
        args.freeze_gene_embeddings = False
        args.disable_label_loss = True
        args.epochs = 10
        args.lr = 1e-3
        args.weight_decay = 1e-4
        args.conditions_per_batch = 4
        args.gene_embedding_dim = 128
        args.hidden_dim = 128
        args.dropout = 0.1
        args.cls_loss_weight = 0.0
        args.additive_anchor_weight = 0.0
        args.cls_focal_gamma = 2.0
        args.class_weights = "1.0,2.5,3.0,2.0,1.5"
        args.regression_class_weights = "1.0,1.0,1.0,1.0,1.0"
        args.unlabeled_reg_weight = 1.0
        return args

    if args.preset == "gears_gse220974_nonadd":
        args.embedding_source = "learned"
        args.freeze_gene_embeddings = False
        args.disable_label_loss = False
        args.epochs = 20
        args.lr = 1e-3
        args.weight_decay = 1e-4
        args.conditions_per_batch = 4
        args.gene_embedding_dim = 128
        args.hidden_dim = 128
        args.dropout = 0.1
        args.cls_loss_weight = 0.1
        args.additive_anchor_weight = 0.05
        args.cls_focal_gamma = 1.0
        args.class_weights = "1.0,1.5,1.5,1.2,1.0"
        args.regression_class_weights = "1.0,1.0,1.0,1.0,1.0"
        args.unlabeled_reg_weight = 1.0
        return args

    raise ValueError(f"Unknown preset: {args.preset}")


def main(default_model_name: str = "unknown") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args(default_model_name=default_model_name)
    args = _apply_preset(args)

    base_result_dir = _resolve_base_result_dir(args)
    output_dir = Path(args.working_dir) / "results" / args.result_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.working_dir) / "results" / args.test_train_config_id

    logger.info("Loading base predictions from %s", base_result_dir)
    base_predictions, gene_names = _load_prediction_result(base_result_dir)

    logger.info("Loading truth data for dataset %s", args.dataset_name)
    adata = _load_truth_adata(args.dataset_name)
    truth_by_condition = _mean_expression_by_condition(adata, gene_names)

    logger.info("Loading split file %s", split_path)
    with split_path.open("r", encoding="utf8") as handle:
        set2conditions = json.load(handle)

    label_csv = Path(args.label_csv).resolve() if args.label_csv else _default_label_csv(args.dataset_name)
    if args.disable_label_loss:
        logger.info("disable_label_loss=True, skipping gene-level coeffect supervision.")
        label_arrays = None
    elif label_csv.exists():
        label_arrays = load_condition_label_arrays(str(label_csv), num_genes=len(gene_names))
    else:
        logger.warning("Gene-level coeffect CSV not found at %s. Training without classification loss.", label_csv)
        label_arrays = None

    embedding_source = args.embedding_source
    if embedding_source == "auto":
        embedding_source = _infer_embedding_source(args.model_name)

    pretrained_gene_embeddings = load_pretrained_gene_embedding_matrix(
        gene_names,
        embedding_source=embedding_source,
    )

    adaptor = PredictionCFYAdaptor(
        gene_names=gene_names,
        gene_embedding_dim=args.gene_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=args.device,
        pretrained_gene_embeddings=pretrained_gene_embeddings,
        trainable_gene_embeddings=not args.freeze_gene_embeddings,
        cfy_config={
            "cls_loss_weight": args.cls_loss_weight,
            "additive_anchor_weight": args.additive_anchor_weight,
            "cls_focal_gamma": args.cls_focal_gamma,
            "class_weights": _parse_class_weights(args.class_weights),
            "regression_class_weights": _parse_class_weights(args.regression_class_weights),
            "unlabeled_reg_weight": args.unlabeled_reg_weight,
        },
    )

    history = adaptor.fit(
        base_predictions=base_predictions,
        truth_by_condition=truth_by_condition,
        train_conditions=set2conditions.get("train", []),
        val_conditions=set2conditions.get("val", []),
        label_arrays=label_arrays,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        conditions_per_batch=args.conditions_per_batch,
        seed=args.seed,
    )

    enhanced_predictions = adaptor.predict(base_predictions)

    tmp_out_dir = _make_local_tmp_output_dir(output_dir.parent, args.result_id)
    with (tmp_out_dir / "all_predictions.json").open("w", encoding="utf8") as handle:
        json.dump(_to_jsonable(enhanced_predictions), handle, indent=4)
    with (tmp_out_dir / "baseline_all_predictions.json").open("w", encoding="utf8") as handle:
        json.dump(_to_jsonable(base_predictions), handle, indent=4)
    with (tmp_out_dir / "gene_names.json").open("w", encoding="utf8") as handle:
        json.dump(_to_jsonable(list(gene_names)), handle, indent=4)
    with (tmp_out_dir / "cfy_history.json").open("w", encoding="utf8") as handle:
        json.dump(_to_jsonable(history), handle, indent=4)
    with (tmp_out_dir / "cfy_metadata.json").open("w", encoding="utf8") as handle:
        json.dump(
            _to_jsonable(
                {
                "base_result_dir": str(base_result_dir),
                "model_name": args.model_name,
                "embedding_source": embedding_source,
                "freeze_gene_embeddings": args.freeze_gene_embeddings,
                "preset": args.preset,
                "disable_label_loss": args.disable_label_loss,
                "dataset_name": args.dataset_name,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "conditions_per_batch": args.conditions_per_batch,
                "gene_embedding_dim": args.gene_embedding_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "cls_loss_weight": args.cls_loss_weight,
                "additive_anchor_weight": args.additive_anchor_weight,
                "cls_focal_gamma": args.cls_focal_gamma,
                "class_weights": _parse_class_weights(args.class_weights),
                "regression_class_weights": _parse_class_weights(args.regression_class_weights),
                "unlabeled_reg_weight": args.unlabeled_reg_weight,
                "seed": args.seed,
                "device": args.device,
                "label_csv": None if args.disable_label_loss else (str(label_csv) if label_csv.exists() else None),
                }
            ),
            handle,
            indent=4,
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(tmp_out_dir), str(output_dir))

    session_info.show()
    print("Python done")


if __name__ == "__main__":
    main()
