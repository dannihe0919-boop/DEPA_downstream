"""Join enrichment hits to final gene identities after Layer 3 has finished."""
from __future__ import annotations

import re

import pandas as pd


def _hit_symbols(value) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().upper() for v in value if str(v).strip()}
    if value is None or pd.isna(value):
        return set()
    return {part.strip().upper() for part in re.split(r'[;,]', str(value)) if part.strip()}


def integrate_results(enrichment: dict, gene_tables: dict) -> dict:
    """Append counts and explicit IDs/symbols of each concordance class per term.

    Counts refer to distinct gene IDs, not hit-list entries. Unmapped symbols,
    ambiguous symbol-to-ID mappings, and untested genes remain explicit.
    Neither input is changed; enrichment is never rerun during integration.
    """
    integrated = {}
    labels = ['concordant', 'discordant', 'inconclusive']
    for program, versions in enrichment.items():
        genes = gene_tables[program]
        if 'concordance' not in genes:
            raise ValueError('integrate_results requires Layer 3b annotations')
        symbol_key = genes['symbol'].astype(str).str.upper()
        integrated[program] = {}
        for kind, table in versions.items():
            rows = []
            for hits in table['genes']:
                symbols = _hit_symbols(hits)
                matched = genes.loc[symbol_key.isin(symbols)].drop_duplicates('gene_id')
                found_symbols = set(matched['symbol'].astype(str).str.upper())
                row = dict(n_hit_symbols=len(symbols), n_mapped_hit_genes=len(matched),
                           n_unmapped_hit_symbols=len(symbols - found_symbols),
                           unmapped_hit_symbols=';'.join(sorted(symbols - found_symbols)),
                           n_ambiguous_hit_symbols=int(matched.groupby(matched['symbol'].str.upper())['gene_id'].nunique().gt(1).sum()),
                           n_unassessed=int((~matched['concordance'].isin(labels)).sum()))
                for label in labels:
                    selected = matched.loc[matched['concordance'] == label]
                    row[f'n_{label}'] = len(selected)
                    row[f'{label}_genes'] = ';'.join(sorted(set(selected['symbol'])))
                    row[f'{label}_gene_ids'] = ';'.join(sorted(set(selected['gene_id'])))
                rows.append(row)
            columns = ['n_hit_symbols', 'n_mapped_hit_genes', 'n_unmapped_hit_symbols', 'unmapped_hit_symbols',
                       'n_ambiguous_hit_symbols', 'n_unassessed']
            columns += [field for label in labels for field in (f'n_{label}', f'{label}_genes', f'{label}_gene_ids')]
            result = pd.concat([table.reset_index(drop=True), pd.DataFrame(rows, columns=columns)], axis=1)
            result.attrs = dict(table.attrs)
            result.attrs['integrated'] = True
            integrated[program][kind] = result
    return integrated
