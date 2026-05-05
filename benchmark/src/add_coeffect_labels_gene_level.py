"""
Add co-effect type labels at [dual_perturbation, gene] level.
Each (condition, gene) pair gets its own co-effect type label.
Following perturblib's approach: keep gene-level labels, don't aggregate.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from tqdm import tqdm


def classify_coeffect_gene_level(adata_path, output_csv_path=None):
    """
    Add co_effect_type at gene level for each (condition, gene) pair.

    Output format:
    - results_df: DataFrame with columns [condition, gene_name, gene_idx, CoEffect_Type, LFC_observed, LFC_additive, interaction, localFDR]
    - Also saves to CSV for easy inspection
    """
    print(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    print(f"Data shape: {adata.shape}")
    print(f"Conditions: {adata.obs['condition'].nunique()}")
    print(f"Genes: {adata.shape[1]}")

    # Clean up: remove 'ctrl' from multi-perturbations
    print("\nCleaning multi-perturbations (removing '+ctrl')...")
    def remove_ctrl(cond):
        if '+' not in str(cond):
            return cond
        parts = str(cond).split('+')
        if len(parts) == 1:
            return cond
        cleaned = [p for p in parts if p != 'ctrl']
        if len(cleaned) == 0:
            return 'ctrl'
        return '+'.join(cleaned)

    adata.obs['condition'] = adata.obs['condition'].apply(remove_ctrl)

    # Get control mean expression for each gene
    print("\nCalculating control mean expression...")
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    ctrl_data = adata[ctrl_mask].X
    if hasattr(ctrl_data, 'toarray'):
        ctrl_data = ctrl_data.toarray()
    ctrl_mean = ctrl_data.mean(axis=0)  # (n_genes,)

    # Build lookup: condition -> mean expression
    print("\nBuilding expression lookup table...")
    conditions = adata.obs['condition'].unique()
    n_genes = adata.shape[1]
    gene_names = adata.var_names.tolist()

    # Calculate mean expression for each condition
    condition_means = {}
    for cond in tqdm(conditions, desc="Computing condition means"):
        mask = adata.obs['condition'] == cond
        data = adata[mask].X
        if hasattr(data, 'toarray'):
            data = data.toarray()
        condition_means[cond] = data.mean(axis=0)  # (n_genes,)

    # Identify multi-perturbations
    multi_conditions = [c for c in conditions if '+' in str(c) and c != 'ctrl']
    print(f"\nFound {len(multi_conditions)} multi-perturbation conditions")

    # Analyze each (multi-perturbation, gene) pair
    print("\nAnalyzing co-effects at gene level...")
    results = []

    for multi_cond in tqdm(multi_conditions, desc="Analyzing multi-perturbations"):
        multi_expr = condition_means[multi_cond]

        # Get single components
        singles = str(multi_cond).split('+')

        # Check if all singles exist
        if not all(s in condition_means for s in singles):
            continue

        single_exprs = [condition_means[s] for s in singles]

        # For each gene, calculate LFC and interaction
        for gene_idx in range(n_genes):
            gene_name = gene_names[gene_idx]
            ctrl_val = ctrl_mean[gene_idx]
            multi_val = multi_expr[gene_idx]

            # LFC for each single perturbation
            single_lfcs = [single_exprs[i][gene_idx] - ctrl_val for i in range(len(singles))]

            # Observed LFC
            lfc_obs = multi_val - ctrl_val

            # Additive expectation
            lfc_add = sum(single_lfcs)

            # Interaction effect
            interaction = abs(lfc_obs - lfc_add)

            results.append({
                'condition': multi_cond,
                'gene_name': gene_name,
                'gene_idx': gene_idx,
                'LFC_observed': lfc_obs,
                'LFC_additive': lfc_add,
                'interaction': interaction,
                'single_lfcs': single_lfcs
            })

    results_df = pd.DataFrame(results)
    print(f"\nTotal (condition, gene) pairs: {len(results_df)}")

    if len(results_df) == 0:
        print("No multi-perturbation records found!")
        return None

    # Local FDR significance test
    print("\nRunning local FDR significance test...")
    results_df = compute_local_fdr(results_df)

    # Classify co-effect types
    print("\nClassifying co-effect types at gene level...")
    results_df = classify_types(results_df)

    # Print statistics
    print("\n" + "="*60)
    print("Co-effect Type Distribution (Gene Level):")
    print("="*60)
    type_counts = results_df['CoEffect_Type'].value_counts()
    for ctype, count in type_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {ctype}: {count} (condition, gene) pairs ({pct:.2f}%)")

    # Show distribution by condition
    print("\n" + "="*60)
    print("Co-effect Type Distribution by Condition:")
    print("="*60)
    condition_type_dist = results_df.groupby('condition')['CoEffect_Type'].value_counts().unstack(fill_value=0)
    print(condition_type_dist.head(10))

    # Save gene-level results
    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nGene-level results saved to {output_csv_path}")

        parquet_path = output_csv_path.with_suffix(".parquet")
        try:
            results_df.to_parquet(parquet_path, index=False)
            print(f"Gene-level results saved to {parquet_path}")
        except Exception as e:
            print(f"Skipping parquet export ({type(e).__name__}: {e})")

    return results_df


def compute_local_fdr(results_df, lfdr_threshold=0.05):
    """Compute local FDR following perturblib's method."""
    deltas = results_df['interaction'].values
    n = len(deltas)

    print(f"  Computing empirical null (n={n})...")
    mu0 = np.median(deltas)
    sigma0 = 1.4826 * np.median(np.abs(deltas - mu0))

    if sigma0 == 0:
        results_df['localFDR'] = 1.0
        results_df['Significant'] = False
        return results_df

    z = (deltas - mu0) / sigma0

    print("  Fitting KDE...")
    kde = gaussian_kde(z)
    n_grid = 10000
    z_grid = np.linspace(z.min() - 1, z.max() + 1, n_grid)

    batch_size = 2000
    f_grid = np.empty(n_grid)
    for i in tqdm(range(0, n_grid, batch_size), desc="  KDE grid eval", leave=False):
        end = min(i + batch_size, n_grid)
        f_grid[i:end] = kde(z_grid[i:end])

    f_z = np.interp(z, z_grid, f_grid)
    f0_z = norm.pdf(z)

    print("  Estimating pi0...")
    central = np.abs(z) < 1
    pi0 = np.mean(central) / 0.682
    pi0 = min(pi0, 1.0)

    print("  Computing local FDR...")
    local_fdr = (pi0 * f0_z) / f_z
    local_fdr = np.clip(local_fdr, 0, 1)

    results_df['localFDR'] = local_fdr
    results_df['Significant'] = local_fdr < lfdr_threshold

    sig_count = results_df['Significant'].sum()
    print(f"  {sig_count}/{n} ({sig_count/n:.1%}) marked as significant")

    return results_df


def classify_types(results_df):
    """Classify co-effect types following perturblib's logic."""
    sig = results_df['Significant'].values
    lfc_add = results_df['LFC_additive'].values
    lfc_obs = results_df['LFC_observed'].values

    co_types = np.full(len(results_df), "Additive", dtype=object)

    for i in range(len(results_df)):
        if not sig[i]:
            continue

        singles = results_df.iloc[i]['single_lfcs']
        if singles is None or len(singles) < 2:
            co_types[i] = "Other"
            continue

        single_signs = np.sign(np.asarray(singles, dtype=float))
        non_zero_signs = single_signs[single_signs != 0]

        if non_zero_signs.size == 0:
            co_types[i] = "Other"
            continue

        # Only classify if single perturbations have consistent direction
        if not np.all(non_zero_signs == non_zero_signs[0]):
            co_types[i] = "Other"
            continue

        single_sign = non_zero_signs[0]
        obs = float(lfc_obs[i])
        add = float(lfc_add[i])

        # Opposite direction
        if np.sign(obs) != single_sign and obs != 0:
            co_types[i] = "Opposite"
            continue

        # Buffering vs Synergy
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

    results_df['CoEffect_Type'] = co_types
    return results_df


def _default_output_csv(adata_path: str) -> Path:
    adata_path = Path(adata_path)
    return adata_path.with_name("perturb_processed_with_coeffect_gene_level.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate gene-level co-effect labels for CFY from a benchmark "
            "perturb_processed.h5ad file."
        )
    )
    parser.add_argument(
        "--adata_path",
        required=True,
        help="Path to perturb_processed.h5ad",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help=(
            "Output CSV path. Default: sibling file named "
            "perturb_processed_with_coeffect_gene_level.csv"
        ),
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv) if args.output_csv else _default_output_csv(args.adata_path)
    results_df = classify_coeffect_gene_level(args.adata_path, str(output_csv))

    if results_df is not None:
        print("\n" + "=" * 60)
        print("Sample Results:")
        print("=" * 60)
        print(
            results_df[
                [
                    "condition",
                    "gene_name",
                    "CoEffect_Type",
                    "LFC_observed",
                    "LFC_additive",
                    "interaction",
                ]
            ].head(20)
        )


if __name__ == "__main__":
    main()
