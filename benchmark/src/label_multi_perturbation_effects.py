import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label multi-perturbation co-effects (Additive/Synergy/Buffering/Opposite/Other) "
            "using benchmark/src/Pdata_Description.py and export a compact JSON mapping."
        )
    )
    parser.add_argument(
        "--perturblib_root",
        type=str,
        default="",
        help=(
            "Optional path to an external perturblib checkout to import perturb_lib from. "
            "If omitted, relies on the current Python environment."
        ),
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="perturb_lib context name (e.g., norman or your registered dataset name)",
    )
    parser.add_argument(
        "--split_id",
        type=str,
        default="",
        help="Optional split id. If set, uses perturb_lib.split_plibdata_3fold(context, split_id=...) when available.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Output JSON path (e.g., results/<run_id>/co_effect_labels.json)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help=(
            "Override PLIB_CHECK_CACHE_DIR to control where parquet/pkl caches are written. "
            "Default: perturb_lib/checks/<dataset_name>/ under current working dir."
        ),
    )

    args = parser.parse_args()

    if args.cache_dir:
        import os

        os.environ["PLIB_CHECK_CACHE_DIR"] = args.cache_dir

    if args.perturblib_root:
        import sys

        sys.path.insert(0, args.perturblib_root)

    import perturb_lib as plib

    from Pdata_Description import Check_Perturbation

    plibdata = plib.load_plibdata(args.context)

    # Optional: apply a split for reproducibility (if the context supports it)
    if args.split_id:
        try:
            splits = plib.split_plibdata_3fold(plibdata, split_id=args.split_id)
            # For labeling we want full dataset labels; merge splits back if provided.
            # Some plib versions return dict with train/val/test.
            if isinstance(splits, dict):
                import polars as pl

                parts = []
                for part in splits.values():
                    if hasattr(part, "_data"):
                        parts.append(part._data)
                if parts:
                    plibdata._data = pl.concat(parts, how="vertical")
        except Exception:
            # If split API signature differs, ignore split.
            pass

    checker = Check_Perturbation(plibdata)
    checker.check_multi_perturbation_coeffect()

    out_path = Path(args.out_json)
    checker.export_coeffect_labels_json(out_path)


if __name__ == "__main__":
    main()
