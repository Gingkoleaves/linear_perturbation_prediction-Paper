from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs CFY predictions by MSE.")
    p.add_argument("--base_result_dir", required=True, help="Directory containing baseline predictions.")
    p.add_argument("--cfy_result_dir", required=True, help="Directory containing CFY predictions.")
    p.add_argument("--ground_truth_dir", required=True, help="Directory containing ground truth predictions.")
    p.add_argument("--base_pred_file", default="all_predictions.json")
    p.add_argument("--cfy_pred_file", default="all_predictions.json")
    p.add_argument("--gt_pred_file", default="all_predictions.json")
    p.add_argument(
        "--no_normalize_keys",
        action="store_true",
        help="Disable condition-key normalization.",
    )
    return p.parse_args()


def _canon_key(key: str) -> str:
    s = key.replace("_", "+")
    tokens = [t.strip() for t in s.split("+") if t.strip()]
    nonctrl = [t for t in tokens if t.lower() != "ctrl"]
    if len(nonctrl) == 0:
        return "ctrl"
    if len(nonctrl) == 1:
        return nonctrl[0]
    return "+".join(sorted(nonctrl))


def _load_predictions(path: Path, normalize_keys: bool) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf8") as f:
        raw = json.load(f)
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        key = _canon_key(str(k)) if normalize_keys else str(k)
        out[key] = np.asarray(v, dtype=np.float64)
    return out


def _eval_mse(
    pred: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
) -> tuple[int, int, float, float]:
    common = sorted(set(pred.keys()) & set(truth.keys()))
    if not common:
        raise ValueError("No common conditions between prediction and ground truth.")
    se_all = [np.square(pred[k] - truth[k]) for k in common]
    mse_all = float(np.mean(np.concatenate(se_all)))
    dual = [k for k in common if "+" in k]
    if not dual:
        return len(common), 0, mse_all, float("nan")
    se_dual = [np.square(pred[k] - truth[k]) for k in dual]
    mse_dual = float(np.mean(np.concatenate(se_dual)))
    return len(common), len(dual), mse_all, mse_dual


def main() -> None:
    args = _parse_args()
    normalize = not args.no_normalize_keys

    base_path = Path(args.base_result_dir) / args.base_pred_file
    cfy_path = Path(args.cfy_result_dir) / args.cfy_pred_file
    gt_path = Path(args.ground_truth_dir) / args.gt_pred_file

    base = _load_predictions(base_path, normalize_keys=normalize)
    cfy = _load_predictions(cfy_path, normalize_keys=normalize)
    gt = _load_predictions(gt_path, normalize_keys=normalize)

    common_all = sorted(set(base.keys()) & set(cfy.keys()) & set(gt.keys()))
    if not common_all:
        raise ValueError("No common conditions across baseline, CFY, and ground truth.")

    base_sub = {k: base[k] for k in common_all}
    cfy_sub = {k: cfy[k] for k in common_all}
    gt_sub = {k: gt[k] for k in common_all}

    n_all, n_dual, mse_base_all, mse_base_dual = _eval_mse(base_sub, gt_sub)
    _, _, mse_cfy_all, mse_cfy_dual = _eval_mse(cfy_sub, gt_sub)

    out = {
        "conditions_all": n_all,
        "conditions_dual": n_dual,
        "gene_dim": int(len(next(iter(gt_sub.values())))),
        "mse_base_all": mse_base_all,
        "mse_cfy_all": mse_cfy_all,
        "delta_all": mse_cfy_all - mse_base_all,
        "rel_change_all_pct": ((mse_cfy_all - mse_base_all) / mse_base_all) * 100.0,
        "mse_base_dual": mse_base_dual,
        "mse_cfy_dual": mse_cfy_dual,
        "delta_dual": mse_cfy_dual - mse_base_dual,
        "rel_change_dual_pct": ((mse_cfy_dual - mse_base_dual) / mse_base_dual) * 100.0,
        "normalized_keys": normalize,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
