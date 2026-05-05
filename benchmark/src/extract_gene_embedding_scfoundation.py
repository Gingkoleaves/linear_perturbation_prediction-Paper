from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import argparse
import session_info

import torch
import anndata as ad

parser = argparse.ArgumentParser(description='Extract gene embedding from scFoundation')

parser.add_argument("--working_dir", dest = "working_dir", action='store', required = True, help = "The directory that contains the params, results, scripts etc.")
parser.add_argument("--result_id", dest = "result_id", action='store', required = True, help = "The result_id")
args = parser.parse_args()
# args = parser.parse_args(["--working_dir", "/scratch/ahlmanne/perturbation_prediction_benchmark", 
#                           "--result_id", "0"])
print(args)

# Extract a fixed embedding per perturbation by mapping perturbation gene -> scFoundation gene embedding.
# This uses the pretrained scFoundation checkpoint (vendored under this repo).
#
# Output format: TSV with columns: pert, dim0..dimN

scfoundation_root = Path("/home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/scFoundation")

singlecell_model_path = scfoundation_root / "model" / "models" / "models.ckpt"
if not singlecell_model_path.exists():
  raise FileNotFoundError(f"scFoundation checkpoint not found: {singlecell_model_path}")

ckp = torch.load(str(singlecell_model_path), map_location="cpu")
gene_pos_emb = ckp['gene']['state_dict']['model.pos_emb.weight'].cpu().numpy()  # shape [n_tokens, emb_dim]

# Prefer deriving gene order from the checkpoint itself.
# If `gene_names` are present in the checkpoint, use them; otherwise fall back to scFoundation demo.h5ad.
gene_names = None
for k in ["gene_names", "genes", "vocab", "gene_vocab"]:
  if isinstance(ckp.get(k, None), (list, tuple)):
    gene_names = list(ckp[k])
    break

if gene_names is None:
  demo_h5ad = scfoundation_root / "model" / "data" / "demo.h5ad"
  if not demo_h5ad.exists():
    raise FileNotFoundError(
      "Could not find gene_names in checkpoint, and demo.h5ad is missing at: "
      f"{demo_h5ad}. Provide a checkpoint that includes gene names, or place demo.h5ad there."
    )
  demo_adata = ad.read_h5ad(str(demo_h5ad))
  gene_names = demo_adata.var.gene_name.tolist()

# Append special tokens that exist in the checkpoint.
# Keep the number of names aligned with the number of token embeddings.
# If lengths mismatch, we truncate/pad with synthetic names.
base_gene_names = list(gene_names)
extras = ["log10TotalCount1", "log10TotalCount2", "<pad>"]
all_names = base_gene_names + extras

n_tokens, emb_dim = gene_pos_emb.shape
if len(all_names) < n_tokens:
  all_names = all_names + [f"<extra_{i}>" for i in range(n_tokens - len(all_names))]
elif len(all_names) > n_tokens:
  all_names = all_names[:n_tokens]

name2emb = {name: gene_pos_emb[i] for i, name in enumerate(all_names)}

# GEARS perturbations include: 'GENE+ctrl' (single) and 'GENE1+GENE2' (double).
# We define perturbation embedding as:
# - ctrl: zeros
# - single: embedding(gene)
# - double: mean(embedding(gene1), embedding(gene2))

# Load perturbation names from GEARS gene2go.pkl used in this benchmark.
pert_data_folder = Path("data/gears_pert_data")
gene2go_path = pert_data_folder / "gene2go.pkl"
if not gene2go_path.exists():
  raise FileNotFoundError(f"Missing {gene2go_path}. This file is required to know perturbation genes.")

import pickle
with open(gene2go_path, "rb") as f:
  gene2go = pickle.load(f)

pert_genes = sorted(set(gene2go.keys()))

rows = []
zero = np.zeros((emb_dim,), dtype=gene_pos_emb.dtype)

for pg in pert_genes:
  emb = name2emb.get(pg, None)
  if emb is None:
    continue
  rows.append((pg, emb))

# Also include a 'ctrl' row for convenience.
rows.append(("ctrl", zero))

out = pd.DataFrame({"pert": [r[0] for r in rows]})
emb_mat = np.vstack([r[1] for r in rows])
for j in range(emb_dim):
  out[f"dim{j}"] = emb_mat[:, j]

outfile = Path(args.working_dir) / "results" / args.result_id
outfile.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(str(outfile), sep="\t", index=False)
print(f"Wrote perturbation embeddings for {len(out)} perts to: {outfile}")



session_info.show()
print("Python done")
