"""Layer 3a population direction and 3b automatic within-cell-type annotation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .layer2_association import resolve_disease_contrast


def compute_population_direction(gene_tables: dict, config: dict) -> pd.DataFrame:
    """Use Layer 0 cached medians/QC and core flags, without rescreening genes.

    Background candidates are the reliable PAC universe. By default remove
    the foreground, preserving the original disjoint MWU comparison. Set
    direction_background_mode='full_universe' for the overlapping comparison.
    expected_direction describes wPSI when *usage increases*, not group B-A.
    """
    mode = config.get('direction_background_mode', 'complement')
    if mode not in ('complement', 'full_universe'):
        raise ValueError('direction_background_mode must be complement or full_universe')
    minimum = int(config.get('min_fg_genes', 5))
    if minimum < 1:
        raise ValueError('min_fg_genes must be positive')
    rows = []
    for program, table in gene_tables.items():
        eligible = table['is_wpsi_reliable'].fillna(False) & np.isfinite(table['gene_wpsi_median'])
        foreground = table.loc[table['is_core'] & eligible]
        background = table.loc[eligible & (~table['is_core'] if mode == 'complement' else True)]
        row = dict(program=program, n_fg_genes=len(foreground), n_bg_genes=len(background),
                   n_core_genes=int(table['is_core'].sum()), fg_median=np.nan, bg_median=np.nan,
                   delta_median=np.nan, mwu_stat=np.nan, pval=np.nan, padj=np.nan,
                   direction='NA', expected_direction='unknown', status='missing_wpsi',
                   background_mode=mode)
        if table['is_wpsi_reliable'].notna().any():
            row['status'] = 'insufficient_foreground' if len(foreground) < minimum else 'insufficient_background'
            if len(foreground) >= minimum and len(background) >= minimum:
                statistic, p = mannwhitneyu(foreground['gene_wpsi_median'], background['gene_wpsi_median'], alternative='two-sided')
                fg = float(foreground['gene_wpsi_median'].median())
                bg = float(background['gene_wpsi_median'].median())
                delta = fg - bg
                row.update(fg_median=fg, bg_median=bg, delta_median=delta, mwu_stat=float(statistic),
                           pval=float(p), direction='Distal' if delta > 0 else 'Proximal' if delta < 0 else 'Unchanged',
                           expected_direction='increase' if delta > 0 else 'decrease' if delta < 0 else 'unknown', status='tested')
        rows.append(row)
    result = pd.DataFrame(rows).set_index('program')
    valid = result['pval'].notna()
    if valid.any():
        result.loc[valid, 'padj'] = multipletests(result.loc[valid, 'pval'], method='fdr_bh')[1]
    result['is_significant'] = result['padj'].lt(config.get('direction_padj_threshold', 0.05))
    return result


def annotate_within_celltype(gene_tables: dict, data: dict, config: dict,
                             association_summary: pd.DataFrame, population_direction: pd.DataFrame,
                             *, inplace: bool = True) -> dict:
    """Write 3b results into the Layer 0 tables; test *all* core genes.

    mean_diff is always disease group B minus A. Its expected sign is the
    usage-increase expectation times sign(mean usage B-A) *within the chosen
    cell type*. Thus a disease with lower usage cannot invert concordance.
    Failed prerequisites have explicit statuses and never become discordant.
    """
    tables = gene_tables if inplace else {p: t.copy(deep=True) for p, t in gene_tables.items()}
    disease = resolve_disease_contrast(config)
    minimum = int(config.get('within_min_cells_per_group', config.get('min_cells_per_group', 20)))
    min_samples = int(config.get('min_samples_per_group') or 1)
    effect = float(config.get('within_effect_threshold', 0.01))
    alpha = float(config.get('within_p_threshold', 0.05))
    significance = config.get('within_significance', 'p')
    if minimum < 1 or min_samples < 1 or effect < 0 or not 0 < alpha <= 1:
        raise ValueError('Invalid within-cell-type sample/effect/significance thresholds')
    if significance not in ('p', 'padj'):
        raise ValueError('within_significance must be p or padj')
    numeric = ['mean_a', 'mean_b', 'mean_diff', 'p', 'padj', 'detection_rate_a', 'detection_rate_b',
               'nonzero_frac_a', 'nonzero_frac_b', 'n_a', 'n_b', 'n_samples_a', 'n_samples_b', 'usage_mean_diff']
    meta, wpsi = data['meta'], data.get('wpsi')
    if not meta.index.is_unique:
        raise ValueError('metadata cell labels must be unique')
    if wpsi is not None and (not wpsi.index.is_unique or not wpsi.columns.is_unique):
        raise ValueError('wPSI gene and cell labels must be unique')
    celltype_col = config['meta_columns']['celltype']
    sample_col = config['meta_columns'].get('sample')
    for program, table in tables.items():
        for column in numeric:
            table[f'within_celltype_{column}'] = np.nan
        table['within_celltype'] = pd.Series(None, index=table.index, dtype=object)
        table['within_celltype_group_a'] = disease['group_a'] if disease else None
        table['within_celltype_group_b'] = disease['group_b'] if disease else None
        table['expected_direction'] = population_direction.loc[program, 'expected_direction']
        table['within_celltype_expected_direction'] = 'unknown'
        table['within_celltype_status'] = 'not_core'
        table['concordance'] = 'not_tested'
        core = table['is_core']

        def skip(reason):
            table.loc[core, 'within_celltype_status'] = reason

        if disease is None:
            skip('missing_disease_contrast')
            continue
        if wpsi is None:
            skip('missing_wpsi')
            continue
        if program not in association_summary.index or pd.isna(association_summary.loc[program, 'best_celltype']):
            skip('missing_celltype_association')
            continue
        selected = association_summary.loc[program]
        celltype = selected['best_celltype']
        table.loc[core, 'within_celltype'] = celltype
        if selected['best_celltype_n_cells'] < 2 * minimum:
            skip('insufficient_cells')
            continue
        subset = meta.loc[(meta[celltype_col] == celltype) & meta[disease['column']].isin([disease['group_a'], disease['group_b']])]
        if sample_col is None or sample_col not in subset:
            if min_samples > 1:
                raise ValueError('Sample column is required for min_samples_per_group > 1')
        usage = data['usage'][program]
        usage_cells = subset.index.intersection(usage.index)
        usage_groups = subset.loc[usage_cells, disease['column']]
        usage_a = usage.loc[usage_cells[usage_groups == disease['group_a']]].dropna()
        usage_b = usage.loc[usage_cells[usage_groups == disease['group_b']]].dropna()
        usage_diff = float(usage_b.mean() - usage_a.mean()) if len(usage_a) and len(usage_b) else np.nan
        table.loc[core, 'within_celltype_usage_mean_diff'] = usage_diff
        expected = {'increase': 1, 'decrease': -1}.get(population_direction.loc[program, 'expected_direction'], 0)
        expected_sign = expected * np.sign(usage_diff) if np.isfinite(usage_diff) and abs(usage_diff) > 1e-12 else 0
        expected_group = 'increase' if expected_sign > 0 else 'decrease' if expected_sign < 0 else 'unknown'
        table.loc[core, 'within_celltype_expected_direction'] = expected_group
        common = subset.index.intersection(wpsi.columns)
        groups = subset.loc[common, disease['column']]
        tested = []
        for index in table.index[core]:
            gid = table.at[index, 'gene_id']
            if gid not in wpsi.index:
                table.at[index, 'within_celltype_status'] = 'missing_gene_wpsi'
                continue
            values = wpsi.loc[gid, common].replace([np.inf, -np.inf], np.nan)
            a = values.loc[groups == disease['group_a']].dropna()
            b = values.loc[groups == disease['group_b']].dropna()
            nsa = subset.loc[a.index, sample_col].nunique() if sample_col in subset else 1
            nsb = subset.loc[b.index, sample_col].nunique() if sample_col in subset else 1
            statistics = dict(n_a=len(a), n_b=len(b), n_samples_a=nsa, n_samples_b=nsb,
                              mean_a=a.mean(), mean_b=b.mean(), mean_diff=b.mean() - a.mean(),
                              detection_rate_a=((a > 0) & (a < 1)).mean(), detection_rate_b=((b > 0) & (b < 1)).mean(),
                              nonzero_frac_a=(a > 0).mean(), nonzero_frac_b=(b > 0).mean())
            for key, value in statistics.items():
                table.at[index, f'within_celltype_{key}'] = value
            if len(a) < minimum or len(b) < minimum:
                table.at[index, 'within_celltype_status'] = 'insufficient_cells'
            elif nsa < min_samples or nsb < min_samples:
                table.at[index, 'within_celltype_status'] = 'insufficient_samples'
            else:
                table.at[index, 'within_celltype_p'] = mannwhitneyu(a, b, alternative='two-sided').pvalue
                table.at[index, 'within_celltype_status'] = 'tested'
                tested.append(index)
        if tested:
            table.loc[tested, 'within_celltype_padj'] = multipletests(table.loc[tested, 'within_celltype_p'], method='fdr_bh')[1]
            for index in tested:
                difference = table.at[index, 'within_celltype_mean_diff']
                p = table.at[index, f'within_celltype_{significance}']
                label = 'inconclusive'
                if expected_sign and np.isfinite(expected_sign) and p < alpha and abs(difference) >= effect and difference != 0:
                    label = 'concordant' if difference * expected_sign > 0 else 'discordant'
                table.at[index, 'concordance'] = label
                if not expected_sign or not np.isfinite(expected_sign):
                    table.at[index, 'within_celltype_status'] = 'unknown_expected_direction'
    return tables
