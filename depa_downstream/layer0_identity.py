"""Layer 0: the single source of program gene identities and wPSI QC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore


def build_program_gene_tables(data: dict, config: dict) -> dict[str, pd.DataFrame]:
    """Aggregate PACs by gene (maximum weight/Z); retain every annotated gene.

    Core membership is inclusive (Z >= threshold); specificity requires all
    other programs' gene Z to be strictly below spec_min_z. Reliability is an
    annotation, never a filter on the identity or enrichment gene lists.
    Missing wPSI is represented by nullable booleans, not reliable=False.
    """
    spectra = data['spectra']
    annotation = data['feature_anno']
    if not spectra.index.is_unique or not spectra.columns.is_unique or not annotation.index.is_unique:
        raise ValueError('spectra programs/PACs and feature annotation PACs must be unique')
    programs = list(spectra.index)
    if not programs:
        raise ValueError('spectra must contain at least one program')
    z = data.get('spectra_z')
    if z is None:
        z = pd.DataFrame(zscore(spectra, axis=1, nan_policy='omit'), index=spectra.index, columns=spectra.columns)
    if not z.index.equals(spectra.index) or not z.columns.equals(spectra.columns):
        raise ValueError('spectra and spectra_z must have identical labels and order')
    pacs = spectra.columns.intersection(annotation.index)
    gene_ids = annotation.loc[pacs, 'gene_id'].dropna().astype(str)
    gene_ids = gene_ids[gene_ids.str.strip().ne('')]
    if gene_ids.empty:
        raise ValueError('No annotated genes overlap the spectra PAC universe')
    scores = spectra.loc[:, gene_ids.index].T.groupby(gene_ids).max().T
    gene_z = z.loc[:, gene_ids.index].T.groupby(gene_ids).max().T
    genes = gene_z.columns
    symbol_map = data.get('gene_symbol_map')
    if symbol_map is not None and not symbol_map.index.is_unique:
        raise ValueError('gene_symbol_map index must be unique')

    def symbol(gid):
        if symbol_map is None:
            return gid
        value = symbol_map.get(gid, symbol_map.get(gid.split('.')[0], gid))
        return str(value).strip() if pd.notna(value) and str(value).strip() else gid

    qc = pd.DataFrame(index=genes)
    qc['wpsi_n_valid'] = 0
    qc['wpsi_detection_rate'] = np.nan
    qc['gene_wpsi_median'] = np.nan
    qc['is_wpsi_reliable'] = pd.Series(pd.NA, index=genes, dtype='boolean')
    min_detection = float(config.get('min_detection_rate', 0.1))
    if not 0 <= min_detection <= 1:
        raise ValueError('min_detection_rate must lie in [0, 1]')
    wpsi = data.get('wpsi')
    if wpsi is not None:
        if not wpsi.index.is_unique or not wpsi.columns.is_unique:
            raise ValueError('wPSI gene and cell labels must be unique')
        matrix = wpsi.reindex(genes).replace([np.inf, -np.inf], np.nan)
        valid = matrix.notna().sum(axis=1)
        rate = ((matrix > 0) & (matrix < 1)).sum(axis=1) / valid.replace(0, np.nan)
        qc['wpsi_n_valid'] = valid
        qc['wpsi_detection_rate'] = rate
        qc['gene_wpsi_median'] = matrix.median(axis=1)
        qc['is_wpsi_reliable'] = (rate.ge(min_detection) & valid.gt(0)).astype('boolean')
    threshold = float(config.get('z_threshold', 1.5))
    specificity = float(config.get('spec_min_z', 0.5))
    if not np.isfinite(threshold) or not np.isfinite(specificity):
        raise ValueError('Z thresholds must be finite')
    tables = {}
    for program in programs:
        other = gene_z.drop(index=program).max(axis=0) if len(programs) > 1 else pd.Series(-np.inf, index=genes)
        table = pd.DataFrame({
            'gene_id': genes, 'symbol': [symbol(g) for g in genes],
            'spectra_score': scores.loc[program].to_numpy(),
            'spectra_z': gene_z.loc[program].to_numpy(),
            'other_max_z': other.to_numpy(),
        }).join(qc, on='gene_id')
        table = table.sort_values(['spectra_z', 'gene_id'], ascending=[False, True], kind='stable').reset_index(drop=True)
        table['rank_in_program'] = np.arange(1, len(table) + 1)
        table['is_core'] = table['spectra_z'].ge(threshold)
        table['is_specific'] = table['is_core'] & table['other_max_z'].lt(specificity)
        table.attrs.update(program=program, z_threshold=threshold, spec_min_z=specificity,
                           min_detection_rate=min_detection, pac_universe_n_genes=len(genes))
        tables[program] = table
    return tables


def pac_universe(gene_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the annotated PAC gene universe, independent of core membership."""
    if not gene_tables:
        raise ValueError('gene_tables must not be empty')
    rows = pd.concat([t[['gene_id', 'symbol']] for t in gene_tables.values()], ignore_index=True)
    if rows.groupby('gene_id')['symbol'].nunique().gt(1).any():
        raise ValueError('A gene has inconsistent symbols across program tables')
    return rows.drop_duplicates('gene_id').sort_values('gene_id').reset_index(drop=True)
