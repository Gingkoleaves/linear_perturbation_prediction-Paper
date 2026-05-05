import perturb_lib as plib
import pandas as pd
import numpy as np
import polars as pl
import pickle
from pathlib import Path
from scipy.stats import norm
from scipy.stats import gaussian_kde
from tqdm import tqdm

class Check_Perturbation:
    def __init__(self, pdata):
        self.pdata = pdata
        self.control_samples = None
        self.control_means = None
        self.single_perturbation_list = None
        self.multi_perturbation_list = None  
        if hasattr(pdata, '_data') and 'context' in pdata._data.columns:
            self.contexts = pdata._data['context'].unique().to_list()
        else:
            self.contexts = None
        if hasattr(pdata, '_data') and 'readout' in pdata._data.columns:
            self.readouts = pdata._data['readout'].unique().to_list()
        else:
            self.readouts = None

        # 根据 context 列自动生成数据集名称，用于缓存路径
        if self.contexts:
            self._dataset_name = "_".join(sorted(self.contexts))
        else:
            self._dataset_name = "unknown_dataset"

    def _get_cache_dir(self) -> Path:
        """获取当前数据集的缓存目录 perturb_lib/checks/{dataset_name}/，自动创建。"""
        # Cache lives under this repo by default to avoid depending on the external perturblib checkout.
        # Can be overridden (e.g. to share caches across repos) via PLIB_CHECK_CACHE_DIR.
        import os
        base = os.environ.get("PLIB_CHECK_CACHE_DIR", "perturb_lib")
        cache_dir = Path(base) / "checks" / self._dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def export_coeffect_labels_json(self, out_path: str | Path) -> None:
        """Export (multi, readout, context) -> co_effect_type mapping as JSON.

        This is a lightweight artifact for downstream consumers (e.g. CFY) that
        don't want the full row-level pdata.
        """
        import json
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not hasattr(self.pdata, "_data"):
            raise ValueError("pdata has no _data attribute")

        df = self.pdata._data.to_pandas()
        if "co_effect_type" not in df.columns:
            raise ValueError("co_effect_type missing; run check_multi_perturbation_coeffect() first")

        # Keep only multi-pert rows; label is per (multi, readout, context)
        df_multi = df[df["perturbation"].astype(str).str.contains("+")].copy()
        if df_multi.empty:
            payload = {"schema": "co_effect_type@v1", "labels": {}}
        else:
            df_multi["multi"] = df_multi["perturbation"].astype(str)
            df_multi = df_multi[["multi", "readout", "context", "co_effect_type"]].dropna()
            df_multi = df_multi.drop_duplicates(subset=["multi", "readout", "context"], keep="first")

            labels: dict[str, str] = {}
            for row in df_multi.itertuples(index=False):
                key = f"{row.multi}|{row.readout}|{row.context}"
                labels[key] = str(row.co_effect_type)
            payload = {"schema": "co_effect_type@v1", "labels": labels}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote co-effect labels JSON to: {out_path}")

    def clean_df(self,df:pd.DataFrame):
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        clean_df = df[
            (df['value'] >= lower) &
            (df['value'] <= upper)
        ]

        print("Before:", len(df))
        print("After:", len(clean_df))
        return clean_df
        # 可能导致清空某些perturbation的样本，后续分析时需要注意
    
    def remove_control_from_multi(self,x):
        # print("Removing 'Control' from multi-perturbation:", x)
        parts = str(x).split('+')
        
        # 如果是单独 Control，不动
        if len(parts) == 1 and parts[0] == plib.ControlSymbol:
            return x
        
        # 多扰动才去掉 Control
        cleaned = [p for p in parts if p != plib.ControlSymbol]
        
        return '+'.join(cleaned)

    def clean_pdata(self):
        print("Deleting 'control' in multi-perturbation samples")

        # 使用 _data (polars DataFrame) 获取 perturbation 列
        if hasattr(self.pdata, '_data') and 'perturbation' in self.pdata._data.columns:
            perturbations = self.pdata._data['perturbation'].to_pandas()
        else:
            print("No 'perturbation' column found in pdata.")
            return
        
        # 删除包含 "Control" 的多扰动样本中的 "Control" 字符串
        perturbations_cleaned = perturbations.apply(self.remove_control_from_multi)
        # 更新pdata中的perturbation列        
        if hasattr(self.pdata, '_data'):
            import polars as pl
            self.pdata._data = self.pdata._data.with_columns(
                pl.Series("perturbation", perturbations_cleaned.values)
            )
        
        return self.pdata

    def check_control(self):
        print("Checking for control samples...")

        if not hasattr(self.pdata, '_data') or 'perturbation' not in self.pdata._data.columns:
            print("No perturbation column found in pdata.")
            return

        df = self.pdata._data.to_pandas()

        if 'value' not in df.columns:
            print("Required 'value' column not found.")
            return

        # 筛选 control
        control_samples = df[df['perturbation'] == plib.ControlSymbol]
        self.control_samples = control_samples

        # 对相同readout\context的control样本取平均值，得到一个代表性的control表达水平
        control_means = control_samples.groupby(['readout', 'context'])['value'].mean().reset_index()
        self.control_means = control_means

        print("Number of control samples:", len(self.control_samples))
        print("Control samples:\n", self.control_samples)
        
    def check_multi_perturbation(self):
        # Check if there are multiple perturbations per cell
        print("Checking for multi-perturbation per cell...")
        
        # 获取perturbation列，无论pdata是DataFrame还是PlibData
        if hasattr(self.pdata, '_data') and 'perturbation' in self.pdata._data.columns:
            perturbations = self.pdata._data['perturbation'].to_pandas()
        elif hasattr(self.pdata, 'columns') and 'perturbation' in self.pdata.columns:
            try:
                perturbations = self.pdata['perturbation'].to_pandas()
            except Exception:
                print("无法获取perturbation列。")
                return
        else:
            print("No 'perturbation' column found in pdata.")
            return

        print("Number of unique perturbations:", perturbations.nunique())
        unique_perturbations = perturbations.unique()
        print("Unique perturbations:", unique_perturbations)
        # 转为 pandas Series 以便使用 apply
        unique_perturbations_pd = pd.Series(unique_perturbations)
        multi_perturbation = unique_perturbations_pd[unique_perturbations_pd.apply(lambda x: len(str(x).split('+')) > 1)]
        # 删除含有Control的多重扰动
        multi_perturbation = multi_perturbation[~multi_perturbation.str.contains("Control")]
        print("Number of unique multi-perturbations:", len(multi_perturbation))
        print("Multi-perturbation types:\n", multi_perturbation)

    def check_control_completeness(self):
        # Check if control samples cover all readout/context combinations
        print("Checking control sample completeness across readout/context combinations...")
        
        if self.control_means is None:
            print("Control means not calculated. Please run check_control first.")
            return 0
        
        # 获取所有的readout/context组合
        all_combinations = self.pdata._data.select(['readout','context']).unique().to_pandas()
        control_combinations = self.control_means[['readout', 'context']]
        
        merged = pd.merge(all_combinations, control_combinations, on=['readout', 'context'], how='left', indicator=True)
        missing_controls = merged[merged['_merge'] == 'left_only']
        
        if not missing_controls.empty:
            print("Missing control samples for the following readout/context combinations:")
            print(missing_controls[['readout', 'context']])
            return -1
        else:
            print("All readout/context combinations have control samples.")
        return 0
    
    # 检查是否每个[multi-perturbation, readout, context]组合的组成部分的单重扰动都存在，如果不完整则无法进行后续分析
    def check_single_in_multi(self):
        # Check if all single perturbations that compose the multi-perturbations are present in the dataset
        print("Checking if all single perturbations that compose the multi-perturbations are present...")
        
        if self.control_means is None:
            print("Control means not calculated. Please run check_control first.")
            return 0
        
        # 逐个复合扰动检查其组成部分的单重扰动是否存在
        df = self.pdata._data.to_pandas()
        multi_perturbations = df[df['perturbation'].apply(lambda x: len(str(x).split('+')) > 1)]['perturbation'].unique()
        single_perturbations = df[df['perturbation'].apply(lambda x: len(str(x).split('+')) == 1)]['perturbation'].unique()
        
        incomplete_multi = []
        single_set = set(single_perturbations)
        for multi in multi_perturbations:
            components = str(multi).split('+')
            missing_comps = [c for c in components if c not in single_set]
            if missing_comps:
                contexts = df.loc[df['perturbation'] == multi, 'context'].unique()
                for context in contexts:
                    for comp in missing_comps:
                        incomplete_multi.append((multi, context, comp))

        if incomplete_multi:
            print(f"Incomplete multi-perturbations found: {len(incomplete_multi)} missing entries")
            for item in incomplete_multi[:20]:
                print(item)
            if len(incomplete_multi) > 20:
                print(f"... and {len(incomplete_multi) - 20} more")

        return 0

                
    # 本函数的目标是为pdata添加一列co_effect_type，标明每个多重扰动样本的协同效应类型
    def check_multi_perturbation_coeffect(self):
        # Check which types of co-effects exist in multi-perturbation samples
        print("Checking for co-effect types in multi-perturbation samples...")

        cache_dir = self._get_cache_dir()

        # 删除control+perturbation中的control，避免干扰后续分析
        self.clean_pdata()
        # 提取所有readout/context下的control平均表达水平，作为后续分析的基线
        self.check_control()
        
        # ========== 检查是否已有标注过 co_effect_type 的 pdata 缓存 ==========
        pdata_cache_path = cache_dir / "pdata_with_coeffect.parquet"
        if pdata_cache_path.exists():
            print(f"Found cached labeled pdata at {pdata_cache_path}, loading...")
            self.pdata._data = pl.read_parquet(pdata_cache_path)
            return

        """

        # 检查control是否包括所有情况的control
        if self.check_control_completeness() == -1:
            print("Control samples are incomplete. Cannot proceed with co-effect analysis.")
            return

        """

        # 检查单扰动是否包括全部多扰动的组成部分，如果不完整则无法进行后续分析
        # self.check_single_in_multi()
        # 太慢了，先不检查了，后续分析时如果发现某些多扰动的组成部分单扰动缺失了，再具体分析原因

        # co-effect types: addictive, buffering, synergy, opposite
        # 最终目标是为pdata添加一列co_effect_type，标明每个多重扰动样本的协同效应类型
        # 根据已有的单重扰动数据，分析多重扰动样本的表达模式，判断其协同效应类型
        # 逐个遍历多重扰动样本，比较其表达模式与对应单重扰动样本的表达模式，判断其协同效应类型

        # ========== 计算或加载 value_lookup ==========
        df_all = self.pdata._data.to_pandas()
        grouped = df_all.groupby(['perturbation', 'readout', 'context']).agg({'value': 'mean'}).reset_index()

        lookup_cache_path = cache_dir / "value_lookup.pkl"
        if lookup_cache_path.exists():
            print(f"Found cached value_lookup at {lookup_cache_path}, loading...")
            with open(lookup_cache_path, 'rb') as f:
                value_lookup = pickle.load(f)
        else:
            # 向量化构建 (perturbation, readout, context) -> mean value 的字典，避免逐行迭代
            keys = list(zip(grouped['perturbation'], grouped['readout'], grouped['context']))
            value_lookup = dict(zip(keys, grouped['value']))

            # 缓存 value_lookup 到磁盘
            with open(lookup_cache_path, 'wb') as f:
                pickle.dump(value_lookup, f)
            print(f"Saved value_lookup cache to {lookup_cache_path}")

        # 向量化收集 (multi-perturbation, readout, context) 组合，避免 context/readout/perturbation 三重循环
        """
        ctrl_mask = grouped['perturbation'] == plib.ControlSymbol
        grouped_unique = df_all[['perturbation', 'readout', 'context']].drop_duplicates()
        # 对齐论文评估口径：优先在 control 条件下表达最高的前 1,000 个 readout 上定义交互类型。
        if ctrl_mask.any():
            top_readouts = (
                grouped.loc[ctrl_mask]
                .groupby('readout', as_index=False)['value']
                .mean()
                .sort_values('value', ascending=False)
                .to_numpy()
            )
            grouped_unique = grouped_unique[grouped_unique['readout'].isin(top_readouts)]
        """
        multi_rows = grouped[grouped['perturbation'].astype(str).str.contains('+', regex=False)]
        tasks = list(zip(multi_rows['perturbation'], multi_rows['readout'], multi_rows['context']))

        print(f"Total multi-perturbation combinations to analyze: {len(tasks)}")

        if len(tasks) == 0:
            df_pd = self.pdata._data.to_pandas()
            if "co_effect_type" not in df_pd.columns:
                df_pd["co_effect_type"] = np.nan
            self.pdata._data = pl.from_pandas(df_pd)
            self.pdata._data.write_parquet(pdata_cache_path)
            print("No multi-perturbation combinations found. Saved pdata with empty co_effect_type labels.")
            return

        results = []
        # 使用 tqdm 显示进度条，逐个分析多重扰动
        for multi, readout, context in tqdm(tasks, desc="Analyzing co-effects", unit="combo"):
            # O(1) 查找多重扰动的表达水平
            multi_value = value_lookup.get((multi, readout, context))
            if multi_value is None:
                print("Error: Missing multi-perturbation value for", multi, readout, context)
                continue

            ctrl_value = value_lookup.get((plib.ControlSymbol, readout, context))
            if ctrl_value is None:
                continue

            # 获取组成这个 multi-perturbation 的单重扰动
            single_components = str(multi).split('+')
            single_values = []
            single_lfcs = []
            all_found = True
            for single in single_components:
                sv = value_lookup.get((single, readout, context))
                if sv is None:
                    all_found = False
                    break
                single_values.append(sv)
                single_lfcs.append(float(sv) - float(ctrl_value))

            if not all_found:
                continue

            # 对齐论文：使用 control-centered LFC 与 additive expectation。
            lfc_obs = float(multi_value) - float(ctrl_value)
            lfc_add = float(np.sum(single_lfcs))
            # interaction effect = |observed - additive|
            interaction_effect = abs(lfc_obs - lfc_add)

            results.append({
                'multi': multi,
                'readout': readout,
                'context': context,
                'LFC_observed': lfc_obs,
                'LFC_additive': lfc_add,
                'interaction': interaction_effect,
                'single_values': single_lfcs
            })

        results_df = pd.DataFrame(results)

        if results_df.empty:
            df_pd = self.pdata._data.to_pandas()
            if "co_effect_type" not in df_pd.columns:
                df_pd["co_effect_type"] = np.nan
            self.pdata._data = pl.from_pandas(df_pd)
            self.pdata._data.write_parquet(pdata_cache_path)
            print("No valid co-effect records after filtering. Saved pdata with empty co_effect_type labels.")
            return

        # 拟合一个零分布，并设定5%的FDR，判断差距是否显著
        results_df = self.check_Significant_localFDR(results_df)

        # 按论文口径分型：先判显著，再按同号前提区分 buffering/synergy/opposite，异号归为 other。
        sig = results_df["Significant"].values
        lfc_add = results_df["LFC_additive"].values
        lfc_obs = results_df["LFC_observed"].values

        co_types = np.full(results_df.shape[0], "Additive", dtype=object)
        for i in range(results_df.shape[0]):
            if not sig[i]:
                continue

            singles = results_df.iloc[i]["single_values"]
            if singles is None or len(singles) < 2:
                co_types[i] = "Other"
                continue

            single_signs = np.sign(np.asarray(singles, dtype=float))
            non_zero_signs = single_signs[single_signs != 0]
            if non_zero_signs.size == 0:
                co_types[i] = "Other"
                continue

            # 只有当两个单扰动方向一致时才做 buffering/synergy/opposite 细分。
            if not np.all(non_zero_signs == non_zero_signs[0]):
                co_types[i] = "Other"
                continue

            single_sign = non_zero_signs[0]
            obs = float(lfc_obs[i])
            add = float(lfc_add[i])

            if np.sign(obs) != single_sign and obs != 0:
                co_types[i] = "Opposite"
                continue

            if single_sign > 0:
                if 0 <= obs <= add:
                    co_types[i] = "Buffering"
                elif obs > add:
                    co_types[i] = "Synergy"
                else:
                    co_types[i] = "Opposite"
            else:
                if add <= obs <= 0:
                    co_types[i] = "Buffering"
                elif obs < add:
                    co_types[i] = "Synergy"
                else:
                    co_types[i] = "Opposite"

        results_df["CoEffect_Type"] = co_types

        effect_df = results_df[["multi", "readout", "context", "CoEffect_Type"]].drop_duplicates(
            subset=["multi", "readout", "context"],
            keep="last",
        )
        df_pd = self.pdata._data.to_pandas()
        df_pd = df_pd.merge(
            effect_df,
            how="left",
            left_on=["perturbation", "readout", "context"],
            right_on=["multi", "readout", "context"],
        )
        df_pd["co_effect_type"] = df_pd["CoEffect_Type"]
        df_pd = df_pd.drop(columns=["multi", "CoEffect_Type"], errors="ignore")
        self.pdata._data = pl.from_pandas(df_pd)

        # ========== 缓存标注后的 pdata 到磁盘 ==========
        self.pdata._data.write_parquet(pdata_cache_path)
        print(f"Saved labeled pdata cache to {pdata_cache_path}")

    def check_Significant_localFDR(self, results_df, lfdr_threshold=0.05):
        print("Running local FDR significance test...")
        deltas = results_df["interaction"].values
        n = len(deltas)

        # ---------- Step 1: Robust empirical null ----------
        print(f"  Step 1/5: Computing empirical null (n={n})...")
        mu0 = np.median(deltas)
        sigma0 = 1.4826 * np.median(np.abs(deltas - mu0))

        if sigma0 == 0:
            results_df["localFDR"] = 1.0
            results_df["Significant"] = False
            return results_df

        z = (deltas - mu0) / sigma0

        # ---------- Step 2: Estimate overall density f(z) ----------
        print("  Step 2/5: Fitting KDE on grid...")
        kde = gaussian_kde(z)
        n_grid = 10000
        z_grid = np.linspace(z.min() - 1, z.max() + 1, n_grid)
        # 分批评估 KDE 网格并显示进度
        batch_size = 2000
        f_grid = np.empty(n_grid)
        for i in tqdm(range(0, n_grid, batch_size), desc="  KDE grid eval", unit="batch"):
            end = min(i + batch_size, n_grid)
            f_grid[i:end] = kde(z_grid[i:end])
        f_z = np.interp(z, z_grid, f_grid)

        # ---------- Step 3: Null density f0(z) ----------
        print("  Step 3/5: Computing null density...")
        f0_z = norm.pdf(z)

        # ---------- Step 4: Estimate pi0 ----------
        print("  Step 4/5: Estimating pi0...")
        central = np.abs(z) < 1
        pi0 = np.mean(central) / 0.682  # 0.682 = P(|Z|<1) for N(0,1)
        pi0 = min(pi0, 1.0)

        # ---------- Step 5: local FDR ----------
        print("  Step 5/5: Computing local FDR and marking significance...")
        local_fdr = (pi0 * f0_z) / f_z
        local_fdr = np.clip(local_fdr, 0, 1)

        results_df["localFDR"] = local_fdr
        results_df["Significant"] = local_fdr < lfdr_threshold

        sig_count = results_df["Significant"].sum()
        print(f"  Done: {sig_count}/{n} ({sig_count/n:.1%}) marked as significant (localFDR < {lfdr_threshold})")

        return results_df

    def check_Significant(self, results_df, fdr_threshold=0.05):
        deltas = results_df["interaction"].values

        # empirical null
        mu = np.median(deltas)
        sigma = 1.4826 * np.median(np.abs(deltas - mu))

        if sigma == 0:
            results_df["Significant"] = False
            return results_df

        # Z-score
        z_scores = (deltas - mu) / sigma
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        # BH-FDR
        order = np.argsort(p_values)
        ranked_p = p_values[order]
        m = len(p_values)

        bh_thresholds = np.arange(1, m+1) / m * fdr_threshold
        below = ranked_p <= bh_thresholds

        significant = np.zeros(m, dtype=bool)

        if np.any(below):
            max_i = np.max(np.where(below))
            significant[order[:max_i+1]] = True

        results_df["Significant"] = significant

        return results_df
            
    def check_perturbation_types(self):
        CoEffect_Type = ["Additive", "Buffering", "Synergy", "Opposite", "Other"]

        # 对types做描述性统计
        if "co_effect_type" not in self.pdata._data.columns:
            print("No co_effect_type column found. Please run check_multi_perturbation_coeffect first.")
            return
        
        df_pd = self.pdata._data.to_pandas()
        for t in CoEffect_Type:
            count = df_pd["co_effect_type"].value_counts().get(t, 0)
            print(f"Number of {t} effects: {count}, proportion: {count / len(df_pd):.2%}")