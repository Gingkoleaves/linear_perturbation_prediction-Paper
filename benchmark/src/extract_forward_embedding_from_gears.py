import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def _get_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_gears_model(gears_obj):
    # GEARS wrapper typically stores the torch model under `.model`
    if hasattr(gears_obj, "model"):
        return gears_obj.model
    raise AttributeError("GEARS object has no attribute 'model'; cannot attach hook")


def _read_conditions_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _conditions_from_set2conditions(set2conditions: Dict[str, List[str]]) -> List[str]:
    conds = []
    for split in ("train", "val", "test"):
        conds.extend(set2conditions.get(split, []))
    # preserve order but unique
    seen = set()
    uniq = []
    for c in conds:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _parse_condition(cond: str) -> Tuple[str, str]:
    # Expected formats:
    # - 'ctrl'
    # - 'GENE+ctrl' or 'ctrl+GENE'
    # - 'GENE1+GENE2'
    if cond == "ctrl":
        return ("ctrl", "ctrl")
    parts = cond.split("+")
    if len(parts) == 1:
        return (parts[0], "ctrl")
    if len(parts) >= 2:
        p1, p2 = parts[0], parts[1]
        return (p1, p2)
    return ("ctrl", "ctrl")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-(pert1, pert2, gene) forward embeddings from GEARS by hooking the latent right before the final gene-specific regression."
        )
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--test_train_config_id", required=True)
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--result_id", required=True)
    parser.add_argument("--epochs", type=int, default=0, help="0 = do not train, just load/init and extract")
    parser.add_argument(
        "--device",
        default="",
        help="e.g. 'cuda' or 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--hook",
        default="transform",
        choices=["transform", "recovery_w"],
        help="Which module to hook. 'transform' captures latent after shared MLP input ReLU; 'recovery_w' captures output of shared decoder MLP.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output npz path. Default: results/<result_id>/forward_embeddings.npz",
    )
    parser.add_argument(
        "--max_conds",
        type=int,
        default=0,
        help="If >0, limit number of conditions for quick debugging.",
    )
    args = parser.parse_args()

    out_dir = Path(args.working_dir) / "results" / args.result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)

    # Import GEARS
    from gears import PertData, GEARS

    pert_data_folder = Path("data/gears_pert_data/")
    pert_data = PertData(pert_data_folder)
    if args.dataset_name in ["norman", "adamson", "dixit"]:
        pert_data.load(args.dataset_name)
    else:
        pert_data.load(data_path=f"data/gears_pert_data/{args.dataset_name}")

    set2conditions = _read_conditions_json(Path(args.working_dir) / "results" / args.test_train_config_id)
    pert_data.set2conditions = set2conditions
    pert_data.split = "custom"
    pert_data.subgroup = None
    pert_data.seed = 1
    pert_data.train_gene_set_size = 0.75
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    if args.epochs and args.epochs > 0:
        gears_model.train(epochs=args.epochs)

    # GEARS.predict runs self.best_model(batch), not self.model.
    # The torch model that actually executes forward is stored under `.best_model`.
    model = getattr(gears_model, "best_model", None)
    if model is None:
        model = _resolve_gears_model(gears_model)

    # choose module to hook
    if not hasattr(model, args.hook):
        raise AttributeError(f"Underlying GEARS model has no module '{args.hook}'. Available: {dir(model)}")
    module = getattr(model, args.hook)

    captured: List[torch.Tensor] = []

    def hook_fn(mod, inp, out):
        # inp is a tuple; we want the first tensor
        if args.hook == "transform":
            # transform takes base_emb and outputs base_emb (ReLU)
            tensor = out
        else:
            # recovery_w outputs per-gene hidden features
            tensor = out
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        captured.append(tensor.detach().cpu())

    handle = module.register_forward_hook(hook_fn)

    # Run predictions condition-by-condition, capturing forward latent.
    # We'll use GEARS.predict because it does batching and calls forward internally.
    # But to align captured tensors with conditions, we call one condition at a time.

    gene_names = pert_data.adata.var["gene_name"].values.tolist()
    num_genes = len(gene_names)

    conds = _conditions_from_set2conditions(set2conditions)
    if args.max_conds and args.max_conds > 0:
        conds = conds[: args.max_conds]

    rows_p1 = []
    rows_p2 = []
    rows_cond = []
    rows_emb = []

    model.eval()
    with torch.no_grad():
        for cond in conds:
            p1, p2 = _parse_condition(cond)
            # gears.predict expects list of list-of-perts without 'ctrl'
            perts = [x for x in [p1, p2] if x != "ctrl"]

            captured.clear()
            pred = gears_model.predict([perts])
            # ensure forward happened (some GEARS versions may bypass the hooked module for ctrl-like cases)
            if len(captured) == 0:
                raise RuntimeError(
                    f"No tensors captured for condition {cond}. Try --hook recovery_w (or check GEARS internals/version)."
                )
            if len(captured) > 1:
                # keep last call if GEARS triggers multiple forwards internally
                # (e.g. internal batching/ensembling). This keeps extraction robust.
                pass

            emb = captured[-1]
            # emb shape could be [B*num_genes, hidden] where B=1, or [num_genes, hidden]
            if emb.ndim != 2:
                raise RuntimeError(f"Unexpected embedding shape {tuple(emb.shape)} for condition {cond}")

            if emb.shape[0] == num_genes:
                per_gene = emb
            elif emb.shape[0] == num_genes * 1:
                per_gene = emb.reshape(1, num_genes, -1)[0]
            else:
                # If GEARS internally expands batch, we only support batch=1 here
                if emb.shape[0] % num_genes != 0:
                    raise RuntimeError(f"Embedding rows {emb.shape[0]} not divisible by num_genes {num_genes}")
                bsz = emb.shape[0] // num_genes
                # GEARS.predict uses a DataLoader(batch_size=300) and then averages predictions across cells.
                # So base embeddings can arrive as [bsz*num_genes, hidden] with bsz=300.
                per_gene = emb.reshape(bsz, num_genes, -1).mean(axis=0)

            rows_p1.append(p1)
            rows_p2.append(p2)
            rows_cond.append(cond)
            rows_emb.append(per_gene.numpy().astype(np.float16))

    handle.remove()

    emb_arr = np.stack(rows_emb, axis=0)  # [num_conds, num_genes, hidden]

    out_path = Path(args.output) if args.output else (out_dir / f"forward_embeddings_{args.hook}.npz")
    np.savez_compressed(
        out_path,
        embeddings=emb_arr,
        conditions=np.array(rows_cond, dtype=object),
        pert1=np.array(rows_p1, dtype=object),
        pert2=np.array(rows_p2, dtype=object),
        gene_names=np.array(gene_names, dtype=object),
        hook=np.array([args.hook], dtype=object),
    )

    print(f"Saved: {out_path}")
    print(f"embeddings shape: {emb_arr.shape} (conds, genes, hidden)")


if __name__ == "__main__":
    main()
