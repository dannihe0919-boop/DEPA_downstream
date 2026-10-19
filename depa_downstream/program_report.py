"""Reusable per-program report supplement. Run from a Notebook after Layer 3.

generate_program_report is offline; run_directional_enrichment optionally queries
Enrichr for the two length-change groups. Neither changes shared Layer 0 tables.
All wPSI values, including 0 and 1, are retained. NaN is never replaced by zero.
"""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import warnings
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from depa_downstream.layer2_association import resolve_disease_contrast
from depa_downstream.layer3_direction import annotate_within_celltype
from depa_downstream.layer0_identity import pac_universe
from depa_downstream.layer1_enrichment import (
    BackgroundEnrichrClient, DEFAULT_ENRICHR_DATABASES, ENRICHMENT_COLUMNS, _symbols,
)

DEFAULT_KEY_GENES = {'P02': ['CENPM', 'NAP1L1', 'TPX2', 'MPC1', 'SOX4', 'STMN1',
                 'PTP4A2', 'PCLAF', 'HNRNPA1', 'HNRNPA0', 'CACYBP', 'PSIP1']}
COLORS = {'shorter': '#3575AD', 'longer': '#DE8744',
          'inconclusive': '#A5A5A5', 'not_tested': '#DDDDDD'}
GROUP_COLORS = ['#526A91', '#B64852']
REPORT_FIGURES = (
    'A_identity_usage', 'B_pooled_preference', 'C_core_gene_length_changes',
    'D_key_gene_distributions', 'E_key_gene_sample_means', 'F_usage_wpsi_trends',
    'G_enrichment_length_overlap',
)


def cleanup_report_figures(output_dir, *, program='P02', remove_formats=('png', 'svg')):
    """Remove only known report PNG/SVG duplicates with an existing PDF.

    Never glob-delete unrelated figures, another program, tables or PDFs.
    A missing/empty/non-PDF counterpart preserves the image as its sole copy.
    May be called on its own without loading data or rerunning statistics.
    """
    if not isinstance(program, str) or not program.strip():
        raise ValueError('program must be a nonempty string')
    if set(remove_formats) - {'png', 'svg'}:
        raise ValueError('Only redundant png/svg formats may be removed')
    out, prefix = Path(output_dir), quote(program, safe='')
    removed = []
    for name in (*REPORT_FIGURES, 'G_directional_enrichment_bubble'):
        pdf = out / f'{prefix}_{name}.pdf'
        if not pdf.is_file():
            continue
        with pdf.open('rb') as stream:
            if stream.read(5) != b'%PDF-':
                continue
        for extension in remove_formats:
            path = out / f'{prefix}_{name}.{extension}'
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(str(path))
    return removed


def _bh(values):
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if valid.any():
        result.loc[valid] = multipletests(values.loc[valid], method='fdr_bh')[1]
    return result


def _preference(delta, p, eligible, alpha, effect):
    result = pd.Series('not_tested', index=delta.index, dtype=object)
    tested = eligible & delta.notna() & p.notna()
    result.loc[tested] = 'inconclusive'
    significant = tested & p.lt(alpha) & delta.abs().ge(effect)
    result.loc[significant & delta.lt(0)] = 'shorter'
    result.loc[significant & delta.gt(0)] = 'longer'
    return result


def _audit_wpsi_range(matrix, gene_table, output_dir, prefix, tolerance):
    """Audit without changing any values; export counts and up to 1000 examples.

    Noncore cell-level values are not consumed by the within-program report.
    Cached medians displayed for core genes or used as reliable population
    background are audited separately.
    """
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-4:
        raise ValueError('wpsi_tolerance must be finite and between 0 and 1e-4')
    array = matrix.to_numpy(dtype=float)
    finite = np.isfinite(array)
    strict = finite & ((array < 0) | (array > 1))
    material = finite & ((array < -tolerance) | (array > 1 + tolerance))
    lookup = gene_table.set_index('gene_id').reindex(matrix.index)
    core = lookup.is_core.fillna(False).to_numpy(dtype=bool)
    reliable = lookup.is_wpsi_reliable.fillna(False).to_numpy(dtype=bool)
    medians = lookup.gene_wpsi_median.to_numpy(dtype=float)
    bad_cached = (core | reliable) & np.isfinite(medians) & ((medians < -tolerance) | (medians > 1 + tolerance))
    per_gene = pd.DataFrame({
        'gene_id': matrix.index, 'symbol': lookup.symbol.to_numpy(), 'is_core': core,
        'n_finite': finite.sum(axis=1), 'n_missing_or_nonfinite': (~finite).sum(axis=1),
        'minimum': matrix.min(axis=1).to_numpy(), 'maximum': matrix.max(axis=1).to_numpy(),
        'n_strict_outside_0_1': strict.sum(axis=1),
        'n_outside_tolerance': material.sum(axis=1),
        'cached_population_median': medians,
        'cached_population_median_outside_tolerance': bad_cached,
    })
    ii, jj = np.nonzero(strict)
    written = min(len(ii), 1000)
    examples = pd.DataFrame({
        'gene_id': matrix.index.to_numpy()[ii[:written]],
        'symbol': lookup.symbol.to_numpy()[ii[:written]],
        'is_core': core[ii[:written]],
        'cell_id': matrix.columns.to_numpy()[jj[:written]],
        'wpsi': array[ii[:written], jj[:written]],
        'outside_tolerance': material[ii[:written], jj[:written]],
    })
    values = array[finite]
    summary = dict(
        tolerance=tolerance, minimum=float(values.min()) if len(values) else None,
        maximum=float(values.max()) if len(values) else None,
        n_finite=int(finite.sum()), n_strict_outside_0_1=int(strict.sum()),
        n_outside_tolerance=int(material.sum()),
        n_core_outside_tolerance=int(material[core].sum()),
        n_noncore_outside_tolerance=int(material[~core].sum()),
        n_used_cached_medians_outside_tolerance=int(bad_cached.sum()),
        examples_written=written, values_modified=False,
    )
    blocked = summary['n_core_outside_tolerance'] > 0 or bad_cached.any()
    summary['status'] = ('blocked_material_outliers' if blocked else
                         'noncore_outliers_only' if material.any() else
                         'within_floating_tolerance' if strict.any() else 'within_range')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(out / f'{prefix}_wpsi_range_by_gene.csv', index=False, encoding='utf-8-sig')
    examples.to_csv(out / f'{prefix}_wpsi_range_examples.csv', index=False, encoding='utf-8-sig')
    (out / f'{prefix}_wpsi_range_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if blocked:
        raise ValueError(
            f'wPSI range audit: min={summary["minimum"]}, max={summary["maximum"]}; '
            f'{summary["n_core_outside_tolerance"]} core-gene values and '
            f'{summary["n_used_cached_medians_outside_tolerance"]} cached medians used by the report '
            f'exceed [0,1] by more than {tolerance}. No values were clipped or rescaled. '
            f'Diagnostics saved to {out / (prefix + "_wpsi_range_summary.json")} '
            f'and {prefix}_wpsi_range_by_gene.csv / {prefix}_wpsi_range_examples.csv. '
            'Check the upstream scale/missing-value encoding; do not automatically divide by 100.')
    if strict.any():
        warnings.warn(
            f'{prefix}: wPSI audit status={summary["status"]}; '
            f'min={summary["minimum"]}, max={summary["maximum"]}. '
            'Original values retained; see range diagnostic files.', UserWarning, stacklevel=2)
    return summary


def generate_program_report(data, config, gene_tables, association_summary,
                        population_direction, *, program='P02', output_dir=None, key_genes=None,
                        enrichment=None,
                        significance='padj', alpha=0.05, effect_threshold=0.01,
                        min_cells_per_sample=10, n_bins=5, formats=('pdf',),
                        wpsi_tolerance=1e-6, cleanup_existing=True):
    """Export one program's tables and figures into the existing layer3 directory.

    Reuse Layer 0 core membership and Layer 2 selected cell type. Recompute only
    the selected program's within-cell-type statistics on a copy to avoid stale Notebook outputs.
    Main plot defaults to within-program BH q<.05, |mean B-A|>=.01; raw-p labels and
    original Layer 3 concordance are also exported. Sample-level tests compare
    unpaired sample means, with at least two usable samples in each group.

    Return a dict with DataFrames, summary metadata and saved figure paths.
    This function assumes increasing wPSI means distal/longer 3'UTR preference;
    confirm this and the upstream meaning of zero before biological reporting.
    Boundary deviations <= wpsi_tolerance are retained, not clipped. Material
    core-gene or used cached-median outliers stop the report after exporting
    diagnostic files. Noncore cell-level outliers are recorded separately.
    Figures default to PDF only. After successful figure generation, remove
    this program's known redundant PNG/SVG files with a PDF counterpart.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not isinstance(program, str) or not program.strip():
        raise ValueError('program must be a nonempty string')
    prefix = quote(program, safe='')
    if significance not in ('p', 'padj') or not 0 < alpha < 1:
        raise ValueError('Use significance=p or padj and 0<alpha<1')
    if (not np.isfinite(effect_threshold) or effect_threshold < 0
            or min_cells_per_sample < 1 or n_bins < 2):
        raise ValueError('Invalid effect, sample-cell or bin threshold')
    if not formats or set(formats) - {'png', 'pdf', 'svg'}:
        raise ValueError('formats must contain png, pdf and/or svg')
    if program not in gene_tables or program not in data['usage']:
        raise ValueError(f'{program} is missing from gene tables or usage')
    if data.get('wpsi') is None:
        raise ValueError('wPSI is required; no values will be imputed')
    disease = resolve_disease_contrast(config)
    if disease is None:
        raise ValueError('A disease contrast with group_a/group_b is required')
    celltype = association_summary.loc[program, 'best_celltype']
    if pd.isna(celltype):
        raise ValueError(f'Layer 2 has no selected {program} cell type')
    a, b, group_col = disease['group_a'], disease['group_b'], disease['column']
    type_col = config['meta_columns']['celltype']
    sample_col = config['meta_columns'].get('sample')
    meta = data['meta']
    subset = meta.loc[meta[type_col].eq(celltype) & meta[group_col].isin([a, b])].copy()
    subset = subset.loc[subset.index.intersection(data['wpsi'].columns)]
    if not all(subset[group_col].eq(g).any() for g in (a, b)):
        raise ValueError('Both disease groups must occur inside the selected cell type')
    ids = gene_tables[program]['gene_id']
    if ids.duplicated().any():
        raise ValueError('Layer 0 gene IDs must be unique')
    out = Path(output_dir if output_dir is not None else Path(config['output_root']) / 'layer3')
    matrix = data['wpsi'].reindex(index=ids, columns=subset.index).replace([np.inf, -np.inf], np.nan)
    range_audit = _audit_wpsi_range(matrix, gene_tables[program], out, prefix, wpsi_tolerance)

    tables = annotate_within_celltype({program: gene_tables[program]}, data, config,
                                     association_summary, population_direction, inplace=False)
    genes = tables[program].loc[lambda t: t['is_core']].copy().reset_index(drop=True)
    if genes.empty:
        raise ValueError(f'{program} has no Layer 0 core genes')
    eligible = genes.within_celltype_status.isin(['tested', 'unknown_expected_direction'])
    # Also supports the earlier Layer3 module when this file is downloaded alone.
    # An undefined population expectation must not prevent a direct B-A analysis.
    for index, row in genes.loc[eligible].iterrows():
        values = matrix.loc[row.gene_id].dropna()
        if len(values) and values.nunique() == 1:
            genes.at[index, 'within_celltype_p'] = 1.0
    genes['within_celltype_padj'] = _bh(genes.within_celltype_p)
    genes['report_length_preference_raw_p'] = _preference(
        genes.within_celltype_mean_diff, genes.within_celltype_p,
        eligible, alpha, effect_threshold)
    genes['report_length_preference_bh'] = _preference(
        genes.within_celltype_mean_diff, genes.within_celltype_padj,
        eligible, alpha, effect_threshold)
    genes['report_length_preference'] = genes['report_length_preference_bh' if significance == 'padj'
                                             else 'report_length_preference_raw_p']
    genes['pooled_delta_from_background'] = genes.gene_wpsi_median - population_direction.loc[program, 'bg_median']
    # Explicitly named interior fraction: it is NOT an expression-detection rate.
    genes['wpsi_interior_fraction_a'] = genes.within_celltype_detection_rate_a
    genes['wpsi_interior_fraction_b'] = genes.within_celltype_detection_rate_b

    usage = data['usage'][program].reindex(subset.index).replace([np.inf, -np.inf], np.nan)
    sample_rows, usage_rows, bin_rows, corr_rows = [], [], [], []
    has_samples = sample_col is not None and sample_col in subset
    sample_groups = (list(subset.dropna(subset=[sample_col]).groupby([group_col, sample_col], observed=True))
                     if has_samples else [])
    for group in (a, b):
        vals = usage.loc[subset[group_col].eq(group)].dropna()
        usage_rows.append(dict(level='cell', group=group, sample=None, n_cells=len(vals),
                               usage_mean=vals.mean(), usage_median=vals.median()))
    for (group, sample), sample_meta in sample_groups:
        vals = usage.reindex(sample_meta.index).dropna()
        usage_rows.append(dict(level='sample', group=group, sample=sample, n_cells=len(vals),
                               usage_mean=vals.mean(), usage_median=vals.median()))
        for row in genes.itertuples():
            values = matrix.loc[row.gene_id, sample_meta.index].dropna()
            sample_rows.append(dict(gene_id=row.gene_id, symbol=row.symbol, group=group,
                                    sample=sample, n_cells_total=len(sample_meta), n_valid=len(values),
                                    wpsi_mean=values.mean(), wpsi_median=values.median(),
                                    wpsi_nonzero_fraction=values.gt(0).mean(),
                                    wpsi_zero_fraction=values.eq(0).mean(),
                                    wpsi_one_fraction=values.eq(1).mean(),
                                    eligible=len(values) >= min_cells_per_sample))
    sample_table = pd.DataFrame(sample_rows, columns=[
        'gene_id', 'symbol', 'group', 'sample', 'n_cells_total', 'n_valid', 'wpsi_mean',
        'wpsi_median', 'wpsi_nonzero_fraction', 'wpsi_zero_fraction', 'wpsi_one_fraction', 'eligible'])
    sample_effects = []
    for row in genes.itertuples():
        valid = sample_table.loc[sample_table.gene_id.eq(row.gene_id) & sample_table.eligible.eq(True)]
        va = valid.loc[valid.group.eq(a), 'wpsi_mean'].dropna()
        vb = valid.loc[valid.group.eq(b), 'wpsi_mean'].dropna()
        shared = set(valid.loc[valid.group.eq(a), 'sample']) & set(valid.loc[valid.group.eq(b), 'sample'])
        status = ('missing_sample_metadata' if not has_samples else 'shared_sample_ids_require_paired_design'
                  if shared else 'tested' if len(va) >= 2 and len(vb) >= 2 else 'insufficient_samples')
        p = np.nan
        if status == 'tested':
            p = (1.0 if va.nunique() == 1 and vb.eq(va.iloc[0]).all()
                 else mannwhitneyu(va, vb, alternative='two-sided').pvalue)
        sample_effects.append(dict(gene_id=row.gene_id, sample_n_a=len(va), sample_n_b=len(vb),
                                   sample_mean_a=va.mean(), sample_mean_b=vb.mean(),
                                   sample_mean_diff=vb.mean() - va.mean(), sample_p=p, sample_status=status))
    sample_effects = pd.DataFrame(sample_effects)
    sample_effects['sample_padj'] = _bh(sample_effects.sample_p)
    sample_effects['sample_length_preference_bh'] = _preference(
        sample_effects.sample_mean_diff, sample_effects.sample_padj,
        sample_effects.sample_status.eq('tested'), alpha, effect_threshold)
    genes = genes.merge(sample_effects, on='gene_id', validate='one_to_one')
    genes = genes.sort_values(['spectra_z', 'gene_id'], ascending=[False, True]).reset_index(drop=True)

    # Shared usage cutpoints, but separate disease groups and separate samples.
    # Bin means are descriptive, not tests; no selection for favorable slopes.
    bins = pd.Series(np.nan, index=subset.index)
    if usage.dropna().nunique() > 1:
        bins.loc[usage.notna()] = pd.qcut(usage.dropna(), n_bins, labels=False, duplicates='drop').astype(float)
    for row in genes.itertuples():
        frame = pd.DataFrame({'wpsi': matrix.loc[row.gene_id], 'usage': usage,
                              'group': subset[group_col], 'bin': bins})
        if has_samples:
            frame['sample'] = subset[sample_col]
        for group in (a, b):
            group_frame = frame.loc[frame.group.eq(group)].dropna(subset=['usage', 'wpsi'])
            rho = (spearmanr(group_frame.usage, group_frame.wpsi).statistic
                   if len(group_frame) >= 3 and group_frame.usage.nunique() > 1
                   and group_frame.wpsi.nunique() > 1 else np.nan)
            corr_rows.append(dict(gene_id=row.gene_id, symbol=row.symbol, group=group,
                                  n_cells=len(group_frame), spearman_rho=rho,
                                  interpretation='descriptive cell correlation; not independent-sample inference'))
        if has_samples:
            for (group, sample, bin_id), sub in frame.dropna().groupby(['group', 'sample', 'bin'], observed=True):
                bin_rows.append(dict(gene_id=row.gene_id, symbol=row.symbol, group=group,
                                     sample=sample, usage_bin=int(bin_id), n_cells=len(sub),
                                     usage_mean=sub.usage.mean(), wpsi_mean=sub.wpsi.mean()))
    bin_table = pd.DataFrame(bin_rows, columns=['gene_id', 'symbol', 'group', 'sample',
                                               'usage_bin', 'n_cells', 'usage_mean', 'wpsi_mean'])
    usage_table = pd.DataFrame(usage_rows)
    corr_table = pd.DataFrame(corr_rows)
    requested = list(dict.fromkeys(DEFAULT_KEY_GENES.get(program, []) if key_genes is None else key_genes))
    selected = genes.loc[genes.symbol.isin(requested)].copy()
    if selected.empty:
        selected = genes.head(8).copy()  # Fixed weight ranking, not significance selection.
    else:
        selected['_order'] = selected.symbol.map({g: i for i, g in enumerate(requested)})
        selected = selected.sort_values(['_order', 'gene_id']).drop(columns='_order')
    # Keep symbol collisions visible by including gene ID in the label.
    duplicates = genes.symbol.duplicated(keep=False)
    genes['plot_label'] = genes.symbol.astype(str)
    genes.loc[duplicates, 'plot_label'] += ' [' + genes.loc[duplicates, 'gene_id'].astype(str) + ']'
    selected = selected.drop(columns='plot_label', errors='ignore').merge(
        genes[['gene_id', 'plot_label']], on='gene_id', validate='one_to_one')

    out.mkdir(parents=True, exist_ok=True)
    figures = {}
    def save(fig, name):
        paths = []
        for extension in formats:
            path = out / f'{prefix}_{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(str(path))
        plt.close(fig)
        figures[name] = paths

    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False}):
        # A. Program localization and within-cell-type usage. No significance stars.
        coords = config['meta_columns']
        umap = all(coords.get(k) in meta for k in ('umap1', 'umap2'))
        fig, axes = plt.subplots(1, 3 if umap else 1, figsize=(14 if umap else 5, 4), squeeze=False)
        ax = axes[0, -1]
        if umap:
            xy = meta[[coords['umap1'], coords['umap2']]].apply(pd.to_numeric, errors='coerce')
            valid = xy.notna().all(axis=1)
            for axis in axes[0, :2]:
                axis.scatter(xy.loc[valid].iloc[:, 0], xy.loc[valid].iloc[:, 1], s=2, c='#DCDCDC', rasterized=True)
                axis.set(xticks=[], yticks=[], xlabel='UMAP 1', ylabel='UMAP 2')
            mask = valid & meta[type_col].eq(celltype)
            axes[0, 0].scatter(xy.loc[mask].iloc[:, 0], xy.loc[mask].iloc[:, 1], s=3, c='#526A91', rasterized=True)
            axes[0, 0].set_title(str(celltype))
            uv = data['usage'][program].reindex(meta.index)
            mask = valid & uv.notna()
            cells = uv.loc[mask].sort_values().index
            sc = axes[0, 1].scatter(xy.loc[cells].iloc[:, 0], xy.loc[cells].iloc[:, 1], c=uv.loc[cells],
                                    s=3, cmap='viridis', rasterized=True)
            fig.colorbar(sc, ax=axes[0, 1], label=f'{program} usage', fraction=.04)
            axes[0, 1].set_title(f'{program} usage (all cells)')
        ua, ub = [usage.loc[subset[group_col].eq(g)].dropna() for g in (a, b)]
        box = ax.boxplot([ua, ub], patch_artist=True, showfliers=False)
        for patch, color in zip(box['boxes'], GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(.4)
        su = usage_table.loc[usage_table.level.eq('sample')]
        for i, (group, color) in enumerate(zip((a, b), GROUP_COLORS), 1):
            vals = su.loc[su.group.eq(group), 'usage_mean'].dropna()
            ax.scatter(i + np.linspace(-.08, .08, len(vals)), vals, color=color, edgecolor='white', s=45, zorder=3)
        auc = mannwhitneyu(ub, ua, alternative='two-sided').statistic / (len(ua) * len(ub)) if len(ua) and len(ub) else np.nan
        ax.set(xticks=[1, 2], xticklabels=[f'{a}\nn={len(ua)}', f'{b}\nn={len(ub)}'], ylabel=f'{program} usage',
               title=f'Within selected cell type\nAUC(B vs A)={auc:.3f}; dots: sample means')
        fig.tight_layout()
        save(fig, 'A_identity_usage')

        # B. Pooled absolute preference is kept separate from disease change.
        ordered = genes.sort_values(['gene_wpsi_median', 'gene_id'], na_position='last')
        fig, ax = plt.subplots(figsize=(7, max(4, .24 * len(ordered))))
        bg = population_direction.loc[program, 'bg_median']
        for y, row in enumerate(ordered.itertuples()):
            marker = '*' if row.is_specific else 'o'
            reliable = pd.notna(row.is_wpsi_reliable) and bool(row.is_wpsi_reliable)
            ax.scatter(row.gene_wpsi_median, y, marker=marker, s=80 if row.is_specific else 25,
                       color='#526A91' if reliable else '#C4C4C4')
        if np.isfinite(bg):
            ax.axvline(bg, color='black', ls='--', label=f'Background median={bg:.3f}')
        ax.set(yticks=range(len(ordered)), yticklabels=ordered.plot_label, xlim=(-.02, 1.02),
               xlabel='Across-cell gene wPSI median', title='Pooled preference; not a disease difference\nStar: specific marker; grey: fails wPSI reliability flag')
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, 'B_pooled_preference')

        # C. All core genes, group means + direct length interpretation + BH q.
        ordered = genes.sort_values(['within_celltype_mean_diff', 'gene_id'], na_position='last')
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, max(5, .27 * len(ordered))), sharey=True)
        for y, row in enumerate(ordered.itertuples()):
            left.plot([row.within_celltype_mean_a, row.within_celltype_mean_b], [y, y], c='#BBBBBB', lw=1)
            for value, color in zip([row.within_celltype_mean_a, row.within_celltype_mean_b], GROUP_COLORS):
                left.scatter(value, y, s=24, color=color)
            color = COLORS[row.report_length_preference]
            right.barh(y, row.within_celltype_mean_diff, color=color, height=.7)
            q = row.within_celltype_padj
            right.text(1.02, y, f'q={q:.2g}' if np.isfinite(q) else row.within_celltype_status,
                       transform=right.get_yaxis_transform(), va='center', fontsize=7)
        left.set(yticks=range(len(ordered)), yticklabels=ordered.plot_label, xlim=(-.02, 1.02),
                 xlabel='Mean wPSI', title=f'Within {celltype}')
        left.legend(handles=[Patch(color=c, label=str(g)) for g, c in zip((a, b), GROUP_COLORS)], fontsize=8)
        right.axvline(0, c='black', lw=.8)
        for x in (-effect_threshold, effect_threshold):
            right.axvline(x, c='#888888', lw=.7, ls=':')
        right.set(xlabel=f'Mean wPSI difference ({b} - {a})',
                  title=f'Blue: shorter; orange: longer\n{significance}<{alpha}, |difference|>={effect_threshold}')
        right.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc='lower right', fontsize=8)
        fig.tight_layout(rect=(0, 0, .91, 1))
        save(fig, 'C_core_gene_length_changes')

        # D. Empirical CDF shows exact 0/1 mass without violin smoothing.
        ncols = min(3, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
        for ax, row in zip(axes.flat, selected.itertuples()):
            for group, color in zip((a, b), GROUP_COLORS):
                vals = matrix.loc[row.gene_id, subset.index[subset[group_col].eq(group)]].dropna().sort_values().to_numpy()
                if len(vals):
                    ax.step(np.r_[vals[0], vals], np.r_[0, np.arange(1, len(vals)+1)/len(vals)],
                            where='post', c=color, label=f'{group} (n={len(vals)})')
                    ax.axvline(vals.mean(), c=color, lw=.8, ls=':')
            ax.set(xlim=(-.02, 1.02), ylim=(0, 1.03), xlabel='wPSI (0 and 1 retained)', ylabel='Cumulative cell fraction',
                   title=f'{row.plot_label}: {row.report_length_preference}\n'
                         f'diff={row.within_celltype_mean_diff:.3f}, q={row.within_celltype_padj:.2g}')
            ax.legend(fontsize=7, loc='lower right')
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f'Within {celltype}; dotted lines: means', y=1.01)
        fig.tight_layout()
        save(fig, 'D_key_gene_distributions')

        # E/F always exist; absence of usable sample metadata is explicit.
        for suffix, mode in [('E_key_gene_sample_means', 'sample'), ('F_usage_wpsi_trends', 'bins')]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
            for ax, row in zip(axes.flat, selected.itertuples()):
                if mode == 'sample':
                    for i, (group, color) in enumerate(zip((a, b), GROUP_COLORS), 1):
                        sub = sample_table.loc[sample_table.gene_id.eq(row.gene_id) & sample_table.group.eq(group)]
                        sub = sub.sort_values('sample')
                        for x, val in zip(i + np.linspace(-.12, .12, len(sub)), sub.itertuples()):
                            ax.scatter(x, val.wpsi_mean, s=40, edgecolor=color,
                                       facecolor=color if val.eligible else 'none')
                            ax.annotate(str(val.sample), (x, val.wpsi_mean), xytext=(3, 2),
                                        textcoords='offset points', fontsize=6)
                        usable = sub.loc[sub.eligible.eq(True), 'wpsi_mean'].dropna()
                        if len(usable):
                            ax.plot([i-.2, i+.2], [usable.mean()]*2, c='black', lw=1)
                    ax.set(xticks=[1, 2], xticklabels=[str(a), str(b)], xlim=(.6, 2.6), ylim=(-.03, 1.03),
                           ylabel='Per-sample mean wPSI', title=f'{row.plot_label}\n'
                           f'sample diff={row.sample_mean_diff:.3f}; q={row.sample_padj:.2g}')
                    if row.sample_status != 'tested':
                        ax.text(.02, .95, row.sample_status.replace('_', ' '), transform=ax.transAxes,
                                va='top', fontsize=7)
                else:
                    for group, color in zip((a, b), GROUP_COLORS):
                        sub = bin_table.loc[bin_table.gene_id.eq(row.gene_id) & bin_table.group.eq(group)
                                            & bin_table.n_cells.ge(min_cells_per_sample)]
                        for _, trace in sub.groupby('sample', observed=True):
                            trace = trace.sort_values('usage_bin')
                            ax.plot(trace.usage_mean, trace.wpsi_mean, c=color, alpha=.2, lw=.8)
                        avg = sub.groupby('usage_bin', observed=True)[['usage_mean', 'wpsi_mean']].mean()
                        if not avg.empty:
                            ax.plot(avg.usage_mean, avg.wpsi_mean, '-o', c=color, ms=4, label=str(group))
                    ax.set(xlabel=f'{program} usage (shared quantile bins)', ylabel='Mean wPSI', ylim=(-.03, 1.03),
                           title=f'{row.plot_label}\nThin: samples; thick: equal-sample mean')
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=7)
                    if bin_table.empty:
                        ax.text(.05, .5, 'No usable sample/bin data', transform=ax.transAxes)
            for ax in list(axes.flat)[len(selected):]:
                ax.set_visible(False)
            fig.tight_layout()
            save(fig, suffix)

        # G. Annotate existing Layer 1 hits; do not re-enrich selected subsets.
        overlap_rows = []
        symbol_groups = {label: set(genes.loc[genes.report_length_preference.eq(label), 'symbol'].str.upper())
                         for label in COLORS}
        for kind, terms in (enrichment or {}).get(program, {}).items():
            if terms is None or terms.empty:
                continue
            for term in terms.itertuples():
                raw_hits = getattr(term, 'genes', '')
                hits = ({str(g).strip().upper() for g in raw_hits}
                        if isinstance(raw_hits, (list, tuple, set)) else
                        {g.strip().upper() for g in str(raw_hits).split(';') if g.strip()})
                record = dict(gene_set=kind, database=term.database, term=term.term,
                              enrichment_padj=term.padj, hit_symbols=';'.join(sorted(hits)))
                accounted = set()
                for label, group_symbols in symbol_groups.items():
                    matched = hits & group_symbols
                    record[f'n_{label}'] = len(matched)
                    record[f'{label}_genes'] = ';'.join(sorted(matched))
                    accounted.update(matched)
                record['n_unmatched'] = len(hits - accounted)
                record['unmatched_genes'] = ';'.join(sorted(hits - accounted))
                overlap_rows.append(record)
        overlap_columns = ['gene_set', 'database', 'term', 'enrichment_padj', 'hit_symbols']
        for label in COLORS:
            overlap_columns += [f'n_{label}', f'{label}_genes']
        overlap_columns += ['n_unmatched', 'unmatched_genes']
        overlap_table = pd.DataFrame(overlap_rows, columns=overlap_columns)
        if not overlap_table.empty:
            top = (overlap_table.loc[overlap_table.enrichment_padj.lt(alpha)]
                   .sort_values('enrichment_padj').drop_duplicates(['database', 'term']).head(12))
            if not top.empty:
                fig, ax = plt.subplots(figsize=(11, max(4, .45*len(top))))
                offset = np.zeros(len(top))
                for label, color in COLORS.items():
                    values = top[f'n_{label}'].to_numpy()
                    ax.barh(range(len(top)), values, left=offset, color=color, label=label)
                    offset += values
                ax.barh(range(len(top)), top.n_unmatched, left=offset, color='white',
                        edgecolor='black', label='unmatched')
                labels = [textwrap.fill(f'{r.gene_set}: {r.term}', 55) for r in top.itertuples()]
                ax.set(yticks=range(len(top)), yticklabels=labels, xlabel='Number of hit gene symbols',
                       title='Existing enrichment terms annotated by wPSI change\n'
                             'Hit counts only; no new subgroup enrichment tests')
                ax.invert_yaxis()
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
                fig.tight_layout()
                save(fig, 'G_enrichment_length_overlap')

    removed = (cleanup_report_figures(
        out, program=program, remove_formats=tuple(ext for ext in ('png', 'svg') if ext not in formats))
        if cleanup_existing and 'pdf' in formats else [])
    summary = dict(program=program, celltype=celltype, group_a=a, group_b=b,
                   figure_formats=list(formats), removed_figure_files=removed,
                   wpsi_range_audit=range_audit,
                   association=association_summary.loc[program].to_dict(),
                   population_direction=population_direction.loc[program].to_dict(),
                   cell_counts=subset[group_col].value_counts().to_dict(),
                   n_core=len(genes), classification_counts=genes.report_length_preference.value_counts().to_dict(),
                   raw_p_counts=genes.report_length_preference_raw_p.value_counts().to_dict(),
                   bh_counts=genes.report_length_preference_bh.value_counts().to_dict(),
                   significance=significance, alpha=alpha, effect_threshold=effect_threshold,
                   min_cells_per_sample=min_cells_per_sample, n_bins=n_bins,
                   sample_tests='unpaired MWU on eligible sample means; BH across selected program core genes',
                   enrichment_overlap_status='annotated' if not overlap_table.empty else 'no_existing_terms',
                   missing_sample_labels=int(subset[sample_col].isna().sum()) if has_samples else len(subset),
                   requested_genes=requested, unavailable_key_genes=[g for g in requested if g not in set(genes.symbol)],
                   assumptions=['Higher wPSI means distal/longer relative 3UTR preference.',
                                'Upstream missingness must be NaN, not encoded as zero.',
                                'Sample identifiers are not automatically independent patients.',
                                'Cell-level p/q are exploratory; pooled preference does not establish a usage-wPSI slope.'])
    for name, table in [('core_gene_statistics', genes), ('key_gene_statistics', selected),
                        ('sample_wpsi', sample_table), ('usage_statistics', usage_table),
                        ('usage_wpsi_bins', bin_table), ('within_group_correlations', corr_table),
                        ('enrichment_length_overlap', overlap_table)]:
        table.to_csv(out / f'{prefix}_{name}.csv', index=False, encoding='utf-8-sig')
    # Convert NaN/Inf to JSON null, preserving numeric fields.
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is pd.NA:
            return None
        return value
    (out / f'{prefix}_report_summary.json').write_text(json.dumps(clean(summary), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (out / f'{prefix}_figure_notes.md').write_text(textwrap.dedent(f'''\
        # {program} 报告图说明

        比较范围：{celltype}；A={a}，B={b}。ΔwPSI 固定为 B−A。
        wPSI 增大解释为远端/较长 3′UTR 偏好；需核对上游定义及零值编码。
        变化大小不是 3′UTR 长度变化百分比，也不是表达量变化。

        - A：全部细胞上的 {program} 分布与所选细胞群内 usage。箱线图使用全部有效细胞，隐藏离群点；散点为样本均值。
        - B：各核心基因跨细胞的 wPSI 中位数与背景中位数。星号为专属 marker；这是基因间比较，不是疾病差异。
        - C：同一细胞群内各核心基因两组 wPSI 均值与 B−A。蓝=偏短，橙=偏长，灰=不确定，浅灰=未完成检验。
        - D：关键基因经验累积分布，保留零值和一值，不作平滑；虚线为均值。
        - E：每点为一个样本的平均 wPSI；空心点表示有效细胞少于 {min_cells_per_sample}，不参加样本检验。
          黑线为合格样本均值的等权平均；样本检验为非配对 MWU，两组各至少两个合格样本，并在 {program} 内做 BH 校正。
          同一样本 ID 跨两组时不执行非配对检验。sample 可能代表文库，不能自动视作独立患者。
        - F：{program} usage 的共享分位数区间内，分别展示两组的 wPSI。细线=样本，粗线=样本均值等权平均。
          每样本每区间至少 {min_cells_per_sample} 个有效细胞。曲线仅作描述，不保证单调，不作为独立因果证据。
        - G（存在显著富集结果时）：复用 Layer1 term 的命中基因，标注其中偏短、偏长或不确定的数量。
          此图没有对长短基因子集重新富集，不能用来宣称某一子集独立显著富集。

        主图分类使用 {significance}<{alpha} 且 |ΔwPSI|≥{effect_threshold}。
        CSV 同时保留原始 p、{program} 内 BH q、两种长短分类和原 Layer3 concordance，互不覆盖。
        所有细胞均值纳入有效的 0 和 1，NaN 不填零；两组细胞不要求同时具有 usage 才参加 wPSI 检验。
        wpsi_interior_fraction 是 0<wPSI<1 的比例，不代表基因表达检出率。
        范围诊断状态：{range_audit['status']}；浮点容差为 {wpsi_tolerance}，不裁剪、不缩放原始数据。
        非核心基因的细胞级越界单独记录；核心基因或参与绘图/背景比较的缓存中位数明显越界时先保存诊断并停止。
        缺失样本标签的细胞仍参加细胞分析，但不参加样本分析。
        如仅细胞级显著、样本级不显著，不能据此宣称跨患者重复成立。
        关键基因为预先指定列表，不按显著性挑选；缺失于核心集的基因记录在 summary JSON。
        本补充分析不重新查询富集、不覆盖 Layer0、不覆盖既有 Layer3 文件。
        同名 {prefix}_ 文件在重复运行时会更新。
        本次图像格式：{', '.join(formats)}；已清理 {len(removed)} 个同名 PNG/SVG 副本，保留 PDF、统计表和其他分析图。
        '''), encoding='utf-8')
    return dict(genes=genes, key_genes=selected, sample_wpsi=sample_table,
                usage=usage_table, usage_bins=bin_table, correlations=corr_table,
                enrichment_overlap=overlap_table,
                summary=summary, figures=figures, removed_figure_files=removed, output_dir=out)


def _plot_directional_enrichment(results, statuses, *, program, alpha, top_n):
    """Select top terms per group, retaining significant cross-group matches."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    groups = ('shorter', 'longer')
    significant = {}
    keys = []
    for group in groups:
        table = results[group]
        table = table.loc[pd.to_numeric(table.padj, errors='coerce').lt(alpha)].copy()
        table['padj'] = pd.to_numeric(table.padj)
        table = table.sort_values(['padj', 'database', 'term'], kind='stable')
        table = table.drop_duplicates(['database', 'term'])
        significant[group] = table
        for row in table.head(top_n).itertuples():
            key = (row.database, row.term)
            if key not in keys:
                keys.append(key)
    displayed = []
    for group in groups:
        for row in significant[group].to_dict('records'):
            if (row['database'], row['term']) in keys:
                displayed.append(dict(group=group, **row))
    display_table = pd.DataFrame(displayed, columns=['group', *ENRICHMENT_COLUMNS])

    labels = [textwrap.fill(f'{term} [{database}]', 65) for database, term in keys]
    # Wrapped lines get their own space instead of truncating long GO/Reactome names.
    heights = [max(1., .62 * (label.count('\n') + 1)) for label in labels]
    y = np.cumsum(heights) - np.asarray(heights) / 2 if keys else np.array([])
    figure_height = max(4.8, 2.5 + .40 * sum(heights))
    fig, ax = plt.subplots(figsize=(12.5, figure_height))
    fig.subplots_adjust(left=.53, right=.78, bottom=1.15 / figure_height, top=1 - .8 / figure_height)
    ax.set_xlim(-.55, 1.55)
    ax.set_xticks([0, 1], [f'{program} {group}\n'
                         f'ΔwPSI {"<" if group == "shorter" else ">"} 0; n={statuses[group]["n_query_symbols"]}' +
                         (f'\n{statuses[group]["status"]}' if statuses[group]['status'] not in ('ok', 'empty') else '')
                         for group in groups], fontsize=9)
    ax.set_title(f'{program}: enrichment of shorter / longer gene sets\npadj < {alpha:g}', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    if not keys:
        ax.set_yticks([])
        ax.text(.5, .55, 'No significant terms available', ha='center', transform=ax.transAxes)
        ax.text(.5, .32, '\n'.join(f'{g}: {statuses[g]["status"]}' for g in groups),
                ha='center', transform=ax.transAxes, fontsize=9)
    else:
        raw_scores = pd.to_numeric(display_table.combined_score, errors='coerce').to_numpy()
        finite = raw_scores[np.isfinite(raw_scores)]
        vmax = max(float(finite.max()) if finite.size else 1., 1.)
        scores = np.nan_to_num(raw_scores, nan=0., posinf=vmax, neginf=0.).clip(0, vmax)
        # Keep original scores/padj in CSV; cap display area to avoid overlapping dots.
        q = display_table.padj.to_numpy(dtype=float)
        sizes = 24 + 25 * np.minimum(-np.log10(np.maximum(q, np.finfo(float).tiny)), 12)
        scatter = ax.scatter([groups.index(g) for g in display_table.group],
                             [y[keys.index((r.database, r.term))] for r in display_table.itertuples()],
                             s=sizes, c=scores, cmap='Reds', norm=Normalize(0, vmax),
                             edgecolors='#555555', linewidths=.65)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_ylim(y[-1] + heights[-1] / 2, 0)
        legend_q = (.01, .00001)
        handles = [ax.scatter([], [], s=24 + 25 * -np.log10(v), color='#BBBBBB',
                              edgecolors='#555555', label=f'padj={v:g}') for v in legend_q]
        ax.legend(handles=handles, title='Bubble size', bbox_to_anchor=(1.05, 1),
                  loc='upper left', fontsize=9, title_fontsize=9)
        cax = fig.add_axes([.82, .34, .018, .28])
        fig.colorbar(scatter, cax=cax, label='Combined score')
    fig.text(.53, .015,
             'Same PAC-universe background; n = mapped query symbols.\n'
             'Size = -log10(padj), capped at 12; nonfinite scores shown at color limits.\n'
             'A blank is not evidence of a significant difference between groups.', fontsize=8)
    return fig, display_table


def run_directional_enrichment(report_genes, gene_tables, config, *, program='P02',
                               output_dir=None, top_n=10, alpha=.05,
                               session=None, on_error='record'):
    """Query shorter/longer core genes separately and export a PDF bubble plot.

    Call with generate_program_report(...)["genes"] and FULL Layer 0 gene_tables.
    Uses report_length_preference from figure C, not concordance/supports labels.
    Both groups use the same mapped PAC universe, never a genome fallback.
    Enrichr padj is specific to each query/database; the figure does not test
    between-group enrichment differences. No new wPSI tests are performed.
    Existing A-G report figures are preserved; only this function queries a server.
    """
    import matplotlib.pyplot as plt

    if not isinstance(program, str) or not program.strip() or program not in gene_tables:
        raise ValueError('program must identify a program in gene_tables')
    if on_error not in ('raise', 'record'):
        raise ValueError('on_error must be raise or record')
    if not isinstance(top_n, int) or top_n < 1 or not 0 < alpha <= 1:
        raise ValueError('top_n must be positive and alpha must lie in (0,1]')
    required = {'gene_id', 'symbol', 'is_core', 'report_length_preference'}
    if required - set(report_genes):
        raise ValueError(f'Missing report columns: {sorted(required - set(report_genes))}')
    genes = report_genes.copy(deep=True)
    source = gene_tables[program].set_index('gene_id')
    if not genes.gene_id.is_unique or not genes.gene_id.isin(source.index).all():
        raise ValueError('Report gene IDs must be unique and belong to the requested program')
    if not genes.report_length_preference.isin(COLORS).all():
        raise ValueError('Unknown report_length_preference; use the current program report')
    if not genes.symbol.fillna('').eq(genes.gene_id.map(source.symbol).fillna('')).all():
        raise ValueError('Report symbols differ from Layer 0; regenerate the program report')
    genes = genes.loc[genes.is_core.fillna(False)].copy()
    if not genes.gene_id.map(source.is_core).fillna(False).all():
        raise ValueError('Report core membership differs from Layer 0')
    if set(genes.gene_id) != set(source.index[source.is_core]):
        raise ValueError('Pass all core rows from the program report, not a selected key-gene subset')
    universe = pac_universe(gene_tables)
    background = _symbols(universe.symbol)
    databases = config.get('enrichr_databases', DEFAULT_ENRICHR_DATABASES)
    minimum = int(config.get('enrichr_min_genes', 3))
    if not isinstance(databases, (list, tuple)) or not databases or minimum < 1:
        raise ValueError('Enrichment requires a nonempty database list and a positive min_genes')
    groups = ('shorter', 'longer')
    lists = {g: genes.loc[genes.report_length_preference.eq(g)].copy() for g in groups}
    queries = {g: _symbols(t.symbol) for g, t in lists.items()}
    if set(queries['shorter']) & set(queries['longer']):
        raise ValueError('A mapped symbol belongs to both length groups; resolve gene mapping first')
    out = Path(output_dir) if output_dir is not None else Path(config['output_root']) / 'layer3'
    out.mkdir(parents=True, exist_ok=True)
    prefix = quote(program, safe='')
    results, statuses = {}, {}
    with BackgroundEnrichrClient(background, session=session, timeout=config.get('enrichr_timeout', 30)) as client:
        for group in groups:
            table = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
            status = dict(program=program, group=group, n_selected_genes=len(lists[group]),
                          n_query_symbols=len(queries[group]),
                          n_unmapped_query_genes=int(lists[group].symbol.map(lambda s: not _symbols([s])).sum()),
                          n_pac_universe_genes=len(universe), n_background_symbols=len(background),
                          background_scope='pac_universe')
            if not background:
                status.update(status='failed', error='PAC universe has no mapped gene symbols')
                if on_error == 'raise':
                    raise ValueError(status['error'])
            elif len(queries[group]) < minimum:
                status['status'] = 'skipped_insufficient_genes'
            else:
                try:
                    table = client.query(queries[group], list(databases), f'{program}_{group}')
                    status['status'] = 'ok' if len(table) else 'empty'
                except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                    if on_error == 'raise':
                        raise
                    status.update(status='failed', error=f'{type(error).__name__}: {error}')
                    warnings.warn(f'{program} {group} enrichment failed: {error}', stacklevel=2)
            table.attrs.update(status)
            results[group], statuses[group] = table, status
            lists[group]['sent_to_enrichr'] = lists[group].symbol.isin(queries[group])
            lists[group].to_csv(out / f'{prefix}_{group}_gene_list.csv', index=False, encoding='utf-8-sig')
            table.to_csv(out / f'{prefix}_{group}_enrichment.csv', index=False, encoding='utf-8-sig')
    universe.to_csv(out / f'{prefix}_directional_enrichment_background.csv', index=False, encoding='utf-8-sig')
    fig, displayed = _plot_directional_enrichment(results, statuses, program=program, alpha=alpha, top_n=top_n)
    pdf = out / f'{prefix}_G_directional_enrichment_bubble.pdf'
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    displayed.to_csv(out / f'{prefix}_directional_enrichment_displayed_terms.csv', index=False, encoding='utf-8-sig')
    metadata = dict(program=program, groups=statuses, databases=list(databases), alpha=alpha, top_n=top_n,
                    grouping='Core genes; report_length_preference from within-celltype mean wPSI B-A',
                    inference='Exploratory cell-level gene selection; not a between-group enrichment test',
                    padj_scope='Enrichr adjusted p-value, separately per query and database',
                    symbol_mapping='Unmapped ENSG identifiers omitted from queries and background',
                    source_context={c: genes[c].dropna().unique().tolist() for c in
                                    ('within_celltype', 'within_celltype_group_a', 'within_celltype_group_b') if c in genes})
    (out / f'{prefix}_directional_enrichment_summary.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (out / f'{prefix}_directional_enrichment_notes.md').write_text(
        '偏短/偏长组直接沿用图C的 report_length_preference，差值为疾病组B−对照组A。\n'
        '两组分别重新富集，背景均为完整PAC基因全集中可映射的symbol；不是仅核心基因或可靠基因。\n'
        'CSV保留未映射基因及发送状态；显著性为各基因组、各数据库的Enrichr校正值。\n'
        '气泡大小为−log10(padj)，显示上限12；颜色为combined score，非有限值仅在绘图时置于色阶边界。\n'
        '每组取前top_n个显著term的并集；另一组若同样显著也展示。长名称换行且保留数据库。\n'
        '空白仅表示该组未有达到阈值的返回结果；不能据此认定两组富集强度存在显著差异。\n'
        '失败或基因数不足记录在summary，不能解释为没有生物学富集。\n'
        '分组依赖细胞级统计，通路结果属于探索性线索，不能代替独立患者层面的验证。\n'
        '本图与旧G的命中数量图是两种分析；旧图和原Layer1结果保留，不互相覆盖。\n', encoding='utf-8')
    return dict(enrichment=results, gene_lists=lists, status=pd.DataFrame(statuses.values()),
                displayed_terms=displayed, figure=str(pdf), output_dir=out)
