import faulthandler
import runpy
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=False)

sys.argv = [
    "run_scgpt.py",
    "--dataset_name",
    "norman",
    "--test_train_config_id",
    "seed_1_norman_split",
    "--working_dir",
    "benchmark",
    "--result_id",
    "scgpt_smoke_trace",
    "--epochs",
    "1",
    "--max_train_steps",
    "1",
    "--predict_max_conds",
    "1",
    "--predict_progress_every",
    "1",
]

runpy.run_path("benchmark/src/run_scgpt.py", run_name="__main__")
