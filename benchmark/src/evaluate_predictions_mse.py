from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

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
        "--label_csv",
        default="",
        help=(
            "Optional gene-level label CSV (for example "
            "perturb_processed_with_coeffect_gene_level.csv). "
            "When set, also reports per-class MSE on labeled (condition, gene) pairs."
        ),
    )
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


def _safe_rel_change(new_value: float, old_value: float) -> float:
    if old_value == 0.0:
        return float("nan")
    return ((new_value - old_value) / old_value) * 100.0


def _resolve_label_name(row: dict[str, str]) -> str | None:
    if row.get("CoEffect_Type"):
        return str(row["CoEffect_Type"]).strip()
    if row.get("label"):
        raw = str(row["label"]).strip()
        label_map = {
            "0": "Additive",
            "1": "Synergy",
            "2": "Buffering",
            "3": "Opposite",
            "4": "Other",
        }
        return label_map.get(raw, raw)
    return None


def _eval_labeled_class_mse(
    base: dict[str, np.ndarray],
    cfy: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    label_csv: Path,
    normalize_keys: bool,
) -> dict[str, object]:
    common = set(base.keys()) & set(cfy.keys()) & set(truth.keys())
    if not common:
        raise ValueError("No common conditions across baseline, CFY, and ground truth.")

    sum_base: dict[str, float] = defaultdict(float)
    sum_cfy: dict[str, float] = defaultdict(float)
    count_pairs: dict[str, int] = defaultdict(int)
    class_conditions: dict[str, set[str]] = defaultdict(set)

    with label_csv.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition_raw = str(row.get("condition", "")).strip()
            if not condition_raw:
                continue
            condition = _canon_key(condition_raw) if normalize_keys else condition_raw
            if condition not in common:
                continue

            label_name = _resolve_label_name(row)
            if not label_name:
                continue

            gene_idx_raw = str(row.get("gene_idx", "")).strip()
            if not gene_idx_raw:
                continue
            try:
                gene_idx = int(gene_idx_raw)
            except ValueError:
                continue

            truth_vec = truth[condition]
            base_vec = base[condition]
            cfy_vec = cfy[condition]
            if gene_idx < 0 or gene_idx >= len(truth_vec):
                continue

            se_base = float((base_vec[gene_idx] - truth_vec[gene_idx]) ** 2)
            se_cfy = float((cfy_vec[gene_idx] - truth_vec[gene_idx]) ** 2)
            sum_base[label_name] += se_base
            sum_cfy[label_name] += se_cfy
            count_pairs[label_name] += 1
            class_conditions[label_name].add(condition)

    class_order = ["Additive", "Synergy", "Buffering", "Opposite", "Other"]
    seen = set(class_order)
    class_order.extend(sorted(k for k in count_pairs.keys() if k not in seen))

    out: dict[str, object] = {}
    for label_name in class_order:
        n_pairs = count_pairs.get(label_name, 0)
        if n_pairs == 0:
            continue
        mse_base = sum_base[label_name] / n_pairs
        mse_cfy = sum_cfy[label_name] / n_pairs
        out[label_name] = {
            "pairs": n_pairs,
            "conditions": len(class_conditions[label_name]),
            "mse_base": mse_base,
            "mse_cfy": mse_cfy,
            "delta": mse_cfy - mse_base,
            "rel_change_pct": _safe_rel_change(mse_cfy, mse_base),
        }

    return out


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
        "rel_change_all_pct": _safe_rel_change(mse_cfy_all, mse_base_all),
        "mse_base_dual": mse_base_dual,
        "mse_cfy_dual": mse_cfy_dual,
        "delta_dual": mse_cfy_dual - mse_base_dual,
        "rel_change_dual_pct": _safe_rel_change(mse_cfy_dual, mse_base_dual),
        "normalized_keys": normalize,
    }

    if args.label_csv:
        label_csv = Path(args.label_csv)
        if not label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        out["class_mse"] = _eval_labeled_class_mse(
            base_sub,
            cfy_sub,
            gt_sub,
            label_csv=label_csv,
            normalize_keys=normalize,
        )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
