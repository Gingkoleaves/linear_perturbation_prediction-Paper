from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[cmd] {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pure-Python GEARS + CFY pipeline runner")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--test_train_config_id", required=True)
    p.add_argument("--working_dir", required=True, help="benchmark directory")
    p.add_argument("--baseline_result_id", default="gears_baseline")
    p.add_argument("--cfy_result_id", default="gears_cfy_plugin")
    p.add_argument("--baseline_epochs", type=int, default=20)
    p.add_argument("--cfy_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--force_baseline", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    working_dir = Path(args.working_dir).resolve()
    results_dir = working_dir / "results"
    base_dir = results_dir / args.baseline_result_id

    if not args.skip_baseline:
        if args.force_baseline or not base_dir.exists():
            _run(
                [
                    sys.executable,
                    "src/run_gears.py",
                    "--dataset_name",
                    args.dataset_name,
                    "--test_train_config_id",
                    args.test_train_config_id,
                    "--working_dir",
                    ".",
                    "--result_id",
                    args.baseline_result_id,
                    "--epochs",
                    str(args.baseline_epochs),
                    "--seed",
                    str(args.seed),
                ],
                cwd=working_dir,
            )
        else:
            print(f"[skip] baseline exists: {base_dir}")

    _run(
        [
            sys.executable,
            "src/run_gears_cfy_plugin.py",
            "--dataset_name",
            args.dataset_name,
            "--test_train_config_id",
            args.test_train_config_id,
            "--working_dir",
            ".",
            "--base_result_id",
            args.baseline_result_id,
            "--result_id",
            args.cfy_result_id,
            "--epochs",
            str(args.cfy_epochs),
            "--model_name",
            "gears",
            "--seed",
            str(args.seed),
        ],
        cwd=working_dir,
    )

    print(f"[done] baseline: {results_dir / args.baseline_result_id}")
    print(f"[done] cfy: {results_dir / args.cfy_result_id}")


if __name__ == "__main__":
    main()
