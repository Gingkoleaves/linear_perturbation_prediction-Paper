"""Extract single-perturbation (gene) embeddings from GEARS GO graph.

This is a Python replacement for benchmark/src/extract_pert_embedding_from_gears.R
and is intended to generate a TSV compatible with CFY runners.

Input:
- data/gears_pert_data/go_essential_all/go_essential_all.csv

Output TSV formats supported downstream:
- Row-wise: gene\tval1\t...\tvalD

We set embedding dim via --dim. For your request, use --dim 64.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _build_adjacency(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    genes = sorted(set(df["source"].astype(str)).union(set(df["target"].astype(str))))
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    a = np.zeros((n, n), dtype=np.float32)
    src = df["source"].astype(str).map(index).to_numpy()
    tgt = df["target"].astype(str).map(index).to_numpy()
    w = df["importance"].to_numpy(dtype=np.float32)

    a[src, tgt] = w
    a[tgt, src] = w
    return a, genes


def _spectral_embed(a: np.ndarray, dim: int, seed: int) -> np.ndarray:
    # symmetric adjacency -> eigenvectors of A (or normalized Laplacian).
    # Use A directly to mimic igraph embed_adjacency_matrix(..., which='lm', scaled=TRUE)
    rng = np.random.default_rng(seed)
    # add tiny noise to break ties / ensure determinism with repeated eigenvalues
    noise = rng.standard_normal(a.shape, dtype=np.float32) * np.float32(1e-6)
    a2 = a + (noise + noise.T) * 0.5

    # eigh for symmetric matrices
    vals, vecs = np.linalg.eigh(a2)
    # take largest magnitude eigenvalues (largest values)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order[:dim]]
    vals = vals[order[:dim]]

    # scale similar to igraph scaled=TRUE (roughly normalize by sqrt(eigenvalue))
    # guard against negative/zero eigenvalues
    scale = np.sqrt(np.maximum(vals, 1e-8)).reshape(1, -1)
    emb = vecs * scale
    return emb.astype(np.float32)


def write_tsv(path: Path, gene_to_vec: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    genes = sorted(gene_to_vec.keys())
    dim = int(next(iter(gene_to_vec.values())).shape[0])
    with open(path, "w") as f:
        for g in genes:
            v = gene_to_vec[g]
            if v.shape != (dim,):
                v = v.reshape(-1)
            f.write(g)
            for x in v.tolist():
                f.write("\t" + (f"{float(x):.8g}"))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--working_dir", required=True)
    ap.add_argument("--result_id", required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    workdir = Path(args.working_dir)
    csv_path = workdir / "data" / "gears_pert_data" / "go_essential_all" / "go_essential_all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)
    required = {"source", "target", "importance"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"missing columns in {csv_path}: {sorted(missing)}")

    a, genes = _build_adjacency(df)
    emb = _spectral_embed(a, dim=args.dim, seed=args.seed)  # [n_genes, dim]

    gene_to_vec = {g: emb[i].copy() for i, g in enumerate(genes)}
    # include ctrl as zeros
    gene_to_vec.setdefault("ctrl", np.zeros((args.dim,), dtype=np.float32))

    out_dir = workdir / "results" / args.result_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pert_emb_go_dim{args.dim}.tsv"
    write_tsv(out_path, gene_to_vec)

    print(f"WROTE {out_path}")
    print(f"genes {len(gene_to_vec)} dim {args.dim}")


if __name__ == "__main__":
    main()
