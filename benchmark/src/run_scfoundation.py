from pathlib import Path
import shutil
import numpy as np
import json
import tempfile
import argparse
import faulthandler
import signal
import threading
import time
from pathlib import Path

import session_info


import sys
# scFoundation uses a forked version of GEARS; point PYTHONPATH to local repo copy.
# Expect scFoundation checked out under this repo at ./scFoundation
sys.path.insert(0, "/home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/scFoundation/GEARS")
sys.path.append("/home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/scFoundation/model")
import gears.version
assert gears.version.__version__ == '0.0.2'
from gears import PertData, GEARS
from gears.utils import filter_pert_in_go


def _redirect_stdio_to_log(log_file: str) -> None:
  if not log_file:
    return
  log_path = Path(log_file)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  f = open(log_path, 'a', buffering=1)
  sys.stdout = f
  sys.stderr = f
  print(f"[run_scfoundation] Redirecting stdout/stderr to: {log_path}")


def _tensor_stats(name, t):
  import torch
  if t is None:
    return {"name": name, "present": False}
  tt = t.detach()
  return {
    "name": name,
    "present": True,
    "shape": list(tt.shape),
    "dtype": str(tt.dtype),
    "device": str(tt.device),
    "min": float(tt[~torch.isnan(tt)].min().item()) if tt.numel() and (~torch.isnan(tt)).any() else None,
    "max": float(tt[~torch.isnan(tt)].max().item()) if tt.numel() and (~torch.isnan(tt)).any() else None,
    "has_nan": bool(torch.isnan(tt).any().item()) if tt.numel() else False,
    "has_inf": bool(torch.isinf(tt).any().item()) if tt.numel() else False,
  }


def _dump_nan_debug(args, batch=None, pred=None, loss=None) -> None:
  out_dir = Path(args.working_dir) / args.nan_debug_dir / args.result_id
  out_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    "args": vars(args),
    "loss": _tensor_stats("loss", loss),
    "pred": _tensor_stats("pred", pred),
  }
  if batch is not None:
    payload["batch_x"] = _tensor_stats("batch.x", getattr(batch, 'x', None))
    payload["batch_y"] = _tensor_stats("batch.y", getattr(batch, 'y', None))
  with open(out_dir / "nan_debug.json", "w") as f:
    json.dump(payload, f, indent=2)
  print(f"[run_scfoundation] Wrote NaN debug dump to: {out_dir / 'nan_debug.json'}")

parser = argparse.ArgumentParser(description='Run scfoundation with gears')
parser.add_argument('--dataset_name', dest='dataset_name', action='store', required = True, help='The id of a file in output/results')
parser.add_argument('--test_train_config_id', dest = 'test_train_config_id', action = 'store', required = True, help = "The ID of the test/train/holdout run")
parser.add_argument('--epochs', dest = 'epochs', action = 'store', help = "How many epochs are run", default = 15, type = int)
parser.add_argument('--seed', dest = 'seed', action = 'store', help = "The seed of the run", default = 1, type = int)

parser.add_argument("--working_dir", dest = "working_dir", action='store', required = True, help = "The directory that contains the params, results, scripts etc.")
parser.add_argument("--result_id", dest = "result_id", action='store', required = True, help = "The result_id")
parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='training device')
parser.add_argument('--cpu_threads', type=int, default=0, help='set torch.set_num_threads when >0')
parser.add_argument('--debug_dump_batch', action='store_true', help='Dump one training batch stats and exit early.')
parser.add_argument('--debug_token_checks', action='store_true', help='Enable extra token/shape sanity checks for debugging.')
parser.add_argument('--log_file', type=str, default='', help='If set, redirect stdout/stderr to this file (append).')
parser.add_argument('--stop_on_nan', action='store_true', help='Abort training at first NaN/Inf loss; dump debug stats before exiting.')
parser.add_argument('--nan_debug_dir', type=str, default='results/nan_debug', help='Directory to write NaN debug dumps when --stop_on_nan is set.')
parser.add_argument('--max_train_steps', type=int, default=0, help='If >0, stop after this many training steps (for quick debugging).')
parser.add_argument(
    '--predict_max_conds',
    type=int,
    default=0,
    help='If >0 and --max_train_steps > 0, only predict this many non-ctrl conditions (plus ctrl if present) to quickly produce JSON during debug.'
)
parser.add_argument(
    '--predict_progress_every',
    type=int,
    default=0,
    help='If >0, print prediction progress every N conditions (useful when predict looks stuck).'
)
parser.add_argument('--predict_after_train', action='store_true', help='If set, run prediction/eval right after training.')
parser.add_argument('--debug_hang_seconds', type=int, default=0, help='If >0, dump stack traces when no step log is printed for this many seconds.')
parser.add_argument('--debug_hang_dump_path', type=str, default='', help='Optional file path for hang dumps. Defaults to results/<result_id>_hang_traces.txt')

args = parser.parse_args()
# args = parser.parse_args(["--dataset_name", "norman",
#     "--test_train_config_id", "8443ed21d2ac4-f8716281f960b", "--working_dir",
#     "/scratch/ahlmanne/perturbation_prediction_benchmark", "--result_id", "0"])
_redirect_stdio_to_log(args.log_file)
print(args)

out_dir = args.working_dir + "/results/" + args.result_id


def _install_hang_debugger():
  faulthandler.enable(all_threads=True)
  dump_path = args.debug_hang_dump_path
  if not dump_path:
    dump_path = str(Path(out_dir) / f"{args.result_id}_hang_traces.txt")
  dump_path = str(Path(dump_path))
  Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
  last_log = {'t': time.time()}

  def mark_log():
    last_log['t'] = time.time()

  def dump_trace(reason: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(dump_path, 'a', encoding='utf-8') as f:
      f.write(f"\n[{ts}] HANG DEBUG: {reason}\n")
      faulthandler.dump_traceback(file=f, all_threads=True)

  def sigusr2_handler(_signum, _frame):
    dump_trace('SIGUSR2 received')

  try:
    signal.signal(signal.SIGUSR2, sigusr2_handler)
  except Exception:
    pass

  def watchdog():
    if args.debug_hang_seconds <= 0:
      return
    while True:
      time.sleep(max(5, min(30, args.debug_hang_seconds // 3)))
      idle = time.time() - last_log['t']
      if idle >= args.debug_hang_seconds:
        dump_trace(f"no log for {idle:.1f}s")
        last_log['t'] = time.time()

  threading.Thread(target=watchdog, daemon=True).start()
  print(f"[run_scfoundation] Hang debugger enabled. SIGUSR2 dumps to: {dump_path}")
  return mark_log


mark_log = None
if args.debug_hang_seconds and args.debug_hang_seconds > 0:
  mark_log = _install_hang_debugger()

np.random.seed(args.seed)
# --------------------------------------------------------


pert_data_folder = Path("data/gears_pert_data/")
pert_data = PertData(pert_data_folder)
if args.dataset_name in ['norman', 'adamson', 'dixit']:
  pert_data.load(args.dataset_name)
else:
  pert_data.load(data_path = "data/gears_pert_data/" + args.dataset_name)

with open(args.working_dir + "/results/" + args.test_train_config_id) as json_file:
  set2conditions = json.load(json_file)

print(set2conditions)
set2conditions["train"] = list(set(set2conditions["train"]).difference(set(["IER5L+ctrl", "LYL1+IER5L"])))
set2conditions["test"] = list(set(set2conditions["test"]).difference(set(["IER5L+ctrl", "LYL1+IER5L"])))
set2conditions["val"] = list(set(set2conditions["val"]).difference(set(["IER5L+ctrl", "LYL1+IER5L"])))
print(set2conditions)

pert_data.set2conditions = set2conditions
pert_data.split = "custom"
pert_data.subgroup = None
pert_data.seed = 1
pert_data.train_gene_set_size = 0.75

# These are based on https://github.com/biomap-research/scFoundation/blob/69b0710660aded7e071d4f67e9f3ac03b096e587/GEARS/run_sh/run_singlecell_maeautobin-0.1B-res0-norman.sh
batch_size=6
accumulation_steps=5
test_batch_size=6
hidden_size=512
model_type="maeautobin"
bin_set="autobin_resolution_append"
# This file was downloaded separately from https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA
singlecell_model_path="/home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/scFoundation/model/models/models.ckpt"
finetune_method="frozen"
train_gene_set_size=0.75
mode="v1"
highres=0 # 0
lr=0.01 #1e-3

if args.cpu_threads and args.cpu_threads > 0:
  import torch
  torch.set_num_threads(args.cpu_threads)

pert_data.get_dataloader(batch_size = batch_size, test_batch_size = test_batch_size)
gears_model = GEARS(pert_data, device = args.device)
gears_model.model_initialize(hidden_size = hidden_size, 
                             model_type = model_type,
                             bin_set=bin_set,
                             load_path=singlecell_model_path,
                             finetune_method=finetune_method,
                             accumulation_steps=accumulation_steps,
                             mode=mode,
                             highres=highres)

if args.stop_on_nan or args.max_train_steps:
  import torch
  import types

  orig_forward = gears_model.model.forward
  state = {"step": 0}

  def wrapped_forward(self, batch):
    pred = orig_forward(batch)
    loss = None
    if hasattr(batch, 'y') and batch.y is not None:
      try:
        loss = torch.nn.functional.mse_loss(pred, batch.y)
      except Exception:
        loss = None

    if args.stop_on_nan and loss is not None:
      if bool((torch.isnan(loss) | torch.isinf(loss)).item()):
        _dump_nan_debug(args, batch=batch, pred=pred, loss=loss)
        raise RuntimeError('NaN/Inf detected in proxy MSE loss')

    state["step"] += 1
    if mark_log is not None:
      mark_log()

    if args.max_train_steps and state["step"] >= args.max_train_steps:
      raise RuntimeError(f"Reached max_train_steps={args.max_train_steps}")
    return pred

  gears_model.model.forward = types.MethodType(wrapped_forward, gears_model.model)

if args.debug_dump_batch or args.debug_token_checks:
  import torch
  loader = pert_data.dataloader['train_loader']
  batch = next(iter(loader))
  try:
    # keep a copy on CPU for inspection
    x = batch.x.detach().cpu() if hasattr(batch, 'x') else None
    y = batch.y.detach().cpu() if hasattr(batch, 'y') else None
    pert = getattr(batch, 'pert', None)
    print('DEBUG batch types:', type(batch))
    print('DEBUG has x:', x is not None, 'has y:', y is not None)
    if x is not None:
      print('DEBUG x shape:', tuple(x.shape), 'dtype:', x.dtype, 'min/max:', float(x.min()), float(x.max()))
    if y is not None:
      print('DEBUG y shape:', tuple(y.shape), 'dtype:', y.dtype, 'min/max:', float(y.min()), float(y.max()))
    if pert is not None:
      # pert can be list[str]
      try:
        print('DEBUG pert sample:', pert[:5])
      except Exception:
        print('DEBUG pert:', pert)

    if args.debug_token_checks:
      from gears import GEARS as _GEARS
      # Access underlying model/encoder settings if available
      model = gears_model.model
      # Try to locate singlecell encoder module
      sc_encoder = None
      for name in ['singlecell_encoder', 'sc_model', 'encoder', 'model']:
        if hasattr(model, name):
          sc_encoder = getattr(model, name)
          break
      print('DEBUG gears_model.device:', args.device)
      print('DEBUG model class:', model.__class__.__name__)
      if sc_encoder is not None:
        print('DEBUG sc_encoder class:', sc_encoder.__class__.__name__)
      else:
        print('DEBUG sc_encoder: not found')

  except Exception as e:
    print('DEBUG dump failed:', repr(e))

  if args.debug_dump_batch:
    print('DEBUG exiting early due to --debug_dump_batch')
    raise SystemExit(0)

temp_dir = tempfile.TemporaryDirectory()
try:
  gears_model.train(epochs = args.epochs, result_dir=temp_dir.name,lr=lr)
except RuntimeError as e:
  if args.max_train_steps and "Reached max_train_steps" in str(e):
    print(f"[run_scfoundation] Early stop: {e}")
  else:
    raise
finally:
  temp_dir.cleanup()

# If we early-stopped, shrink prediction set to avoid long predict.
# This is only for debug runs where the user wants to quickly see JSON output.
early_stopped = bool(args.max_train_steps and args.max_train_steps > 0)


tmp_out_dir = tempfile.mkdtemp()
gears_model.save_model(f'{tmp_out_dir}/gears_model')


conds = pert_data.adata.obs["condition"].cat.remove_unused_categories().cat.categories.tolist()
if early_stopped and args.predict_max_conds > 0:
  # Keep ctrl first if present, then take the first N others deterministically.
  non_ctrl = [c for c in conds if c != 'ctrl']
  keep = ['ctrl'] if 'ctrl' in conds else []
  keep += non_ctrl[: args.predict_max_conds]
  conds = keep
split_conds = [x.split("+") for x in conds]
split_conds = [list(filter(lambda y: y != "ctrl", x)) for x in split_conds]

# Predict with optional progress logging
if args.predict_progress_every > 0:
  print(f"[run_scfoundation] Predicting {len(split_conds)} conditions...")

all_pred_vals = {}
if args.predict_progress_every > 0:
  import time as _time
  _t0 = _time.time()

for i, cond in enumerate(split_conds):
  # cond is list of pert genes (ctrl removed)
  if args.predict_progress_every > 0 and (i % args.predict_progress_every == 0):
    dt = _time.time() - _t0
    label = '+'.join(cond) if len(cond) else 'ctrl'
    print(f"[run_scfoundation] predict {i}/{len(split_conds)} elapsed={dt:.1f}s last={label}")
  label = '+'.join(cond) if len(cond) else 'ctrl'
  try:
    _t1 = _time.time() if args.predict_progress_every > 0 else None
    pred_dict = gears_model.predict([cond])
    if args.predict_progress_every > 0:
      dt1 = _time.time() - _t1
      print(f"[run_scfoundation] predict done {i}/{len(split_conds)} dt={dt1:.2f}s cond={label}")
  except Exception as e:
    print(f"[run_scfoundation] predict failed for {label}: {repr(e)}")
    raise
  # pred_dict keys are condition strings
  for k, v in pred_dict.items():
    all_pred_vals[k] = v

all_pred_vals = {k: v.tolist() for k, v in all_pred_vals.items()}
with open(f"{tmp_out_dir}/all_predictions.json", 'w', encoding="utf8") as handle:
  json.dump(all_pred_vals, handle, indent = 4)
with open(f"{tmp_out_dir}/gene_names.json", 'w', encoding="utf8") as handle:
    json.dump(pert_data.adata.var["gene_name"].values.tolist(), handle, indent = 4)

# Move results to out_dir
shutil.move(tmp_out_dir, out_dir)



session_info.show()
print("Python done")
