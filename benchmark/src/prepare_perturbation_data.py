import argparse
from pathlib import Path
import json 
import os

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
import scanpy as sc
import pickle
import session_info
import urllib.request
import zipfile
import shutil
import pandas as pd
from scipy import io as spio





parser = argparse.ArgumentParser(description='Prepare data for combinatorial perturbation prediction')
parser.add_argument('--dataset_name', dest='dataset_name', action='store', required = True, help='The id of a file in output/results')
parser.add_argument('--seed', dest = 'seed', action = 'store', help = "The seed of the run", default = 1, type = int)

parser.add_argument("--working_dir", dest = "working_dir", action='store', required = True, help = "The directory that contains the params, results, scripts etc.")
parser.add_argument("--result_id", dest = "result_id", action='store', required = True, help = "The result_id")
args = parser.parse_args()
# args = parser.parse_args(["--dataset_name", "norman",
#                           "--working_dir", "/scratch/ahlmanne/perturbation_prediction_benchmark", 
#                           "--result_id", "0"])
print(args)

outfile = args.working_dir + "/results/" + args.result_id
np.random.seed(args.seed)


def normalize_condition_names(obs):
  import pandas as pd
  ser = pd.Series(["+".join(sorted(x.split("+"))) for x in obs['condition']], dtype = "category", index = obs.index)
  obs['condition'] = ser
  return obs

def canon_gse220974_condition(label):
  label = str(label)
  if label == "Non-Targeting":
    return "ctrl"

  genes = []
  for token in label.split("|"):
    token = token.strip()
    if (not token) or token == "Non-Targeting":
      continue
    genes.append(token.split("-")[0])

  genes = sorted(set(genes))
  if len(genes) == 0:
    return "ctrl"
  if len(genes) == 1:
    return genes[0] + "+ctrl"
  return "+".join(genes)

def read_nonempty_lines(path):
  return [line.strip() for line in path.read_text(encoding = "utf8").splitlines() if line.strip()]

def build_gse220974_adata():
  input_dir = Path(os.getenv("GSE220974_DIR", "/home/gingkoleaves/Documents/GSE datats/GSE220974"))
  matrix = spio.mmread(input_dir / "GSE220974_RNA_matrix.mtx").tocsr().transpose().tocsr()
  barcodes = read_nonempty_lines(input_dir / "GSE220974_RNA_barcodes.tsv")
  features = pd.read_csv(
    input_dir / "GSE220974_RNA_features.tsv",
    sep = "\t",
    header = None,
    names = ["feature_id", "gene_name", "feature_type"]
  )
  metadata = pd.read_csv(input_dir / "GSE220974_K562_cell_metadata.csv").set_index("cell").reindex(barcodes)

  if metadata.isna().any().any():
    missing = int(metadata.isna().any(axis = 1).sum())
    raise ValueError(f"Metadata missing for {missing} barcodes after reindex.")

  if matrix.shape != (len(barcodes), len(features)):
    raise ValueError(
      f"RNA matrix shape {matrix.shape} does not match "
      f"{len(barcodes)} barcodes x {len(features)} features."
    )

  obs = metadata.copy()
  obs.index.name = "cell"
  obs["cell_type"] = "K562"
  obs["condition"] = obs["guide_group2"].astype(str).map(canon_gse220974_condition)
  obs["total_count"] = np.asarray(matrix.sum(axis = 1)).ravel().astype(np.float32)

  var = features.copy()
  var.index = pd.Index(var["feature_id"].astype(str), name = "gene_id")
  var["gene_name"] = var["gene_name"].astype(str)

  adata = sc.AnnData(X = matrix, obs = obs, var = var)
  adata.obs_names_make_unique()
  adata.var_names_make_unique()
  return adata
# -------------------------

pert_data_folder = Path("data/gears_pert_data")

if args.dataset_name == "norman_from_scfoundation":
  import sys
   # scfoundation uses a forked version of GEARS which
  sys.path.insert(0, "/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/scfoundation_gears/")
  sys.path.append("/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/model/")
  import gears.version
  assert gears.version.__version__ == '0.0.2'
  from gears import PertData, GEARS

  if not Path("data/gears_pert_data/" + args.dataset_name + "/perturb_processed.h5ad").exists():
    # download_link = "https://figshare.com/ndownloader/files/44477939"
    # urllib.request.urlretrieve(download_link, "data/norman_from_scfoundation_data.zip")
    # with zipfile.ZipFile("data/norman_from_scfoundation_data.zip","r") as zip_ref:
      # zip_ref.extractall("data/norman_from_scfoundation_data")
    # if not (pert_data_folder / "gene2go.pkl"):
      # shutil.copyfile("/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/scfoundation_gears/data/gene2go.pkl", pert_data_folder / "gene2go.pkl")
    pert_data = PertData(pert_data_folder)
    adata = sc.read_h5ad("data/norman_from_scfoundation_data/gse133344_k562gi_oe_pert227_84986_19264_withtotalcount.h5ad")
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    pert_data.new_data_process(dataset_name=args.dataset_name, adata=adata)
    if not (pert_data_folder / "gene2go.pkl"):
      shutil.copyfile("/home/ahlmanne/huber/data/scfoundation_model/data/demo/go.csv", pert_data_folder / args.dataset_name / "go.csv")  
else:
  from gears import PertData, GEARS
  import gears.version
  assert gears.version.__version__ == '0.1.2'
  if args.dataset_name == "gse220974" and not Path("data/gears_pert_data/" + args.dataset_name + "/perturb_processed.h5ad").exists():
    pert_data = PertData(pert_data_folder)
    adata = build_gse220974_adata()
    pert_data.new_data_process(dataset_name=args.dataset_name, adata=adata)


if args.dataset_name == "norman":
  pert_data = PertData(pert_data_folder)
  pert_data.load(args.dataset_name)
  norman_adata = pert_data.adata
  new_obs = normalize_condition_names(norman_adata.obs.copy())
  if not norman_adata.obs.equals(new_obs):
    norman_adata.obs = new_obs
    # Override the perturb_processed.h5ad
    norman_adata.write_h5ad(pert_data_folder / "norman/perturb_processed.h5ad")
    # Delete the data_pyg folder because it has the problematic references to the 
    data_pyg_folder = (pert_data_folder / "norman" / "data_pyg")
    if data_pyg_folder.exists():
      (data_pyg_folder / "cell_graphs.pkl").unlink(missing_ok = True)
      data_pyg_folder.rmdir()
    # Redo
    pert_data = PertData(pert_data_folder)
    pert_data.load(args.dataset_name)
  conds = norman_adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
  single_pert = [x for x in conds if 'ctrl' in x]
  double_pert = np.setdiff1d(conds, single_pert).tolist()
  double_training = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
  double_test = np.setdiff1d(double_pert, double_training).tolist()
  double_test = double_test[0:(len(double_test)//2)]
  double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
  set2conditions = {
      "train": single_pert + double_training,
      "test": double_test,
      "val": double_holdout
  }
elif args.dataset_name == "norman_from_scfoundation":
  pert_data = PertData(pert_data_folder)
  pert_data.load(data_path = "data/gears_pert_data/" + args.dataset_name)
  norman_adata = pert_data.adata
  conds = norman_adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
  single_pert = [x for x in conds if 'ctrl' in x]
  double_pert = np.setdiff1d(conds, single_pert).tolist()
  double_training = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
  double_test = np.setdiff1d(double_pert, double_training).tolist()
  double_test =  np.random.choice(double_test, size = len(double_test)//2, replace = False).tolist()
  double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
  set2conditions = {
      "train": single_pert + double_training,
      "test": double_test,
      "val": double_holdout
  }
elif args.dataset_name == "gse220974":
  pert_data = PertData(pert_data_folder)
  pert_data.load(data_path = "data/gears_pert_data/" + args.dataset_name)
  gse220974_adata = pert_data.adata
  conds = gse220974_adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
  single_pert = [x for x in conds if 'ctrl' in x]
  double_pert = np.setdiff1d(conds, single_pert).tolist()
  double_training = np.random.choice(double_pert, size = len(double_pert) // 2, replace = False).tolist()
  double_test = np.setdiff1d(double_pert, double_training).tolist()
  double_test = np.random.choice(double_test, size = len(double_test) // 2, replace = False).tolist()
  double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
  set2conditions = {
      "train": single_pert + double_training,
      "test": double_test,
      "val": double_holdout
  }
else:
  pert_data = PertData(pert_data_folder)
  pert_data.load(args.dataset_name)
  pert_data.prepare_split(split = 'simulation', seed = args.seed)
  set2conditions = pert_data.set2conditions


with open(outfile, "w") as outfile: 
    json.dump(set2conditions, outfile)
session_info.show()
print("Python done")
