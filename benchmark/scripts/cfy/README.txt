CFY Plugin 运行速查（scFoundation / GEARS / Additive）

这些脚本是 `benchmark/src` 的薄封装，不会自动激活 conda 环境。
以下命令都在 `benchmark/` 目录执行。

通用模式（两阶段）
1) 先跑 baseline，生成 `results/<base_result_id>/all_predictions.json`
2) 再跑 CFY post-hoc，生成 `results/<cfy_result_id>/all_predictions.json`

------------------------------------------------------------
scFoundation + CFY
------------------------------------------------------------
环境:
  conda activate scfoundation_env

baseline:
  bash scripts/cfy/run_scfoundation_baseline.sh \
    norman_from_scfoundation seed_1_norman_from_scfoundation_split \
    scf_norman_e15 15

CFY:
  bash scripts/cfy/run_scfoundation_cfy_plugin.sh \
    norman_from_scfoundation seed_1_norman_from_scfoundation_split \
    scf_norman_e15 scf_norman_e15_cfy 5

等价 Python 入口:
  python3 src/run_scfoundation_cfy_plugin.py \
    --dataset_name norman_from_scfoundation \
    --test_train_config_id seed_1_norman_from_scfoundation_split \
    --working_dir . \
    --base_result_id scf_norman_e15 \
    --result_id scf_norman_e15_cfy \
    --epochs 5 \
    --model_name scfoundation

------------------------------------------------------------
GEARS + CFY
------------------------------------------------------------
环境:
  conda activate gears_env2

baseline:
  bash scripts/cfy/run_gears_baseline.sh \
    norman seed_1_norman_split \
    gears_norman_e20 20

CFY:
  bash scripts/cfy/run_gears_cfy_plugin.sh \
    norman seed_1_norman_split \
    gears_norman_e20 gears_norman_e20_cfy 5

一键 Python 流水线（baseline + CFY）:
  python3 src/run_gears_cfy_pipeline.py \
    --dataset_name norman \
    --test_train_config_id seed_1_norman_split \
    --working_dir . \
    --baseline_result_id gears_norman_e20 \
    --cfy_result_id gears_norman_e20_cfy \
    --baseline_epochs 20 \
    --cfy_epochs 5

若 baseline 已存在，仅跑 CFY:
  python3 src/run_gears_cfy_pipeline.py \
    --dataset_name gse220974 \
    --test_train_config_id seed_1_gse220974_split \
    --working_dir . \
    --baseline_result_id gears_gse220974_e20 \
    --cfy_result_id gears_gse220974_e20_cfy_overall_best_v2 \
    --cfy_epochs 10 \
    --skip_baseline

gse220974 推荐 preset:
  整体 MSE 最优:
  python3 src/run_gears_cfy_plugin.py \
    --dataset_name gse220974 \
    --test_train_config_id seed_1_gse220974_split \
    --working_dir . \
    --base_result_id gears_gse220974_e20 \
    --result_id gears_gse220974_e20_cfy_overall_best_v2 \
    --model_name gears \
    --preset gears_gse220974_overall

  非 additive 类更稳（历史配置，当前结果目录未保留）:
  python3 src/run_gears_cfy_plugin.py \
    --dataset_name gse220974 \
    --test_train_config_id seed_1_gse220974_split \
    --working_dir . \
    --base_result_id gears_gse220974_e20 \
    --result_id gears_gse220974_e20_cfy_nonadd_best \
    --model_name gears \
    --preset gears_gse220974_nonadd

------------------------------------------------------------
Additive + CFY
------------------------------------------------------------
环境:
  conda activate gears_env2

baseline（如需重跑）:
  python3 src/run_additive_model.py \
    --dataset_name gse220974 \
    --test_train_config_id seed_1_gse220974_split \
    --working_dir . \
    --result_id addi_gse220974_results

当前最优 Python 入口:
  python3 src/run_additive_cfy_plugin.py \
    --dataset_name gse220974 \
    --test_train_config_id seed_1_gse220974_split \
    --working_dir . \
    --base_result_id addi_gse220974_results \
    --result_id addi_gse220974_cfy_regonly_e20 \
    --model_name additive \
    --disable_label_loss \
    --epochs 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --conditions_per_batch 4 \
    --gene_embedding_dim 128 \
    --hidden_dim 128 \
    --dropout 0.1 \
    --cls_loss_weight 0 \
    --additive_anchor_weight 0

------------------------------------------------------------
其他
------------------------------------------------------------
GEARS forward embeddings:
  conda activate gears_env2
  bash scripts/cfy/extract_gears_forward_embeddings.sh \
    norman seed_1_norman_split gears_forward_emb transform 0

评测 baseline vs CFY（MSE）:
  conda activate gears_env2
  python3 src/evaluate_predictions_mse.py \
    --base_result_dir results/gears_norman_e20_cu128 \
    --cfy_result_dir results/gears_norman_e20_cu128_cfy_e5 \
    --ground_truth_dir results/ground_truth_norman_results

  conda activate gears_env2
  python3 src/evaluate_predictions_mse.py \
    --base_result_dir results/gears_gse220974_e20 \
    --cfy_result_dir results/gears_gse220974_e20_cfy_overall_best_v2 \
    --ground_truth_dir results/ground_truth_gse220974_results

 python3 src/evaluate_predictions_mse.py \
    --base_result_dir results/gears_gse220974_e20 \
    --cfy_result_dir results/gears_gse220974_e20_cfy_overall_best_v2 \
    --ground_truth_dir results/ground_truth_gse220974_results \
    --label_csv data/gears_pert_data/gse220974/perturb_processed_with_coeffect_gene_level.csv

 python3 src/evaluate_predictions_mse.py \
    --base_result_dir results/addi_gse220974_results \
    --cfy_result_dir results/addi_gse220974_cfy_regonly_e20 \
    --ground_truth_dir results/ground_truth_gse220974_results \
    --label_csv data/gears_pert_data/gse220974/perturb_processed_with_coeffect_gene_level.csv

  说明: 
  - 默认读取每个目录下的 `all_predictions.json`
  - 默认会统一 condition key（`_`/`+`、去掉 `ctrl`、双扰动排序）
  - 如需关闭 key 归一化，追加 `--no_normalize_keys`

输出目录:
  `benchmark/results/<result_id>/`

注意:
- `norman_from_scfoundation` 会加载很大的 data_pyg，耗时较长。
- 如果出现 `BrokenPipeError`（例如错误管道到不存在命令），请改用 `grep`。


# 1) scFoundation baseline vs CFY
cd /home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/benchmark
conda activate scfoundation_env
python3 src/evaluate_predictions_mse.py \
  --base_result_dir results/scfoundation_norman_results \
  --cfy_result_dir results/scfoundation_norman_results_cfy_e5 \
  --ground_truth_dir results/ground_truth_norman_results

# 2) GEARS baseline vs CFY
cd /home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/benchmark
conda activate gears_env2
python3 src/evaluate_predictions_mse.py \
  --base_result_dir results/gears_norman_e20_cu128 \
  --cfy_result_dir results/gears_norman_e20_cu128_cfy_e5 \
  --ground_truth_dir results/ground_truth_norman_results

# 3) Additive baseline vs CFY (20 epochs version)
cd /home/gingkoleaves/Documents/linear_perturbation_prediction-Paper/benchmark
conda activate gears_env2
python3 src/evaluate_predictions_mse.py \
  --base_result_dir results/addi_norman_results \
  --cfy_result_dir results/addi_norman_results_cfy_e20 \
  --ground_truth_dir results/ground_truth_norman_results
