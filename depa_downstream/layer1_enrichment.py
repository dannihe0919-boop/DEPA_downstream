"""Layer 1: Enrichr/Speedrichr with an explicit PAC-universe background."""
from __future__ import annotations

import warnings

import pandas as pd
import requests

from .layer0_identity import pac_universe

DEFAULT_ENRICHR_DATABASES = [
    'GO_Biological_Process_2023',
    'KEGG_2021_Human',
    'MSigDB_Hallmark_2020',
    'Reactome_2022',
]

SPEEDRICHR_URL = 'https://maayanlab.cloud/speedrichr/api'
ENRICHMENT_COLUMNS = ['database', 'rank', 'term', 'pvalue', 'zscore', 'combined_score', 'genes', 'padj']


def _symbols(values) -> list[str]:
    return sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip()
                   and not str(v).strip().upper().startswith('ENSG')})


class BackgroundEnrichrClient:
    """A context-managed client; one uploaded background reused by all queries.

    Uses the documented Speedrichr addList/addbackground/backgroundenrich
    protocol, also used by GSEApy. Never falls back to a genome background.
    """
    def __init__(self, background: list[str], session=None, timeout: float = 30):
        self.background = background
        self.session = session if session is not None else requests.Session()
        self.owns_session = session is None
        self.timeout = timeout
        self.background_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.owns_session:
            self.session.close()

    def _post(self, endpoint, **kwargs):
        response = self.session.post(f'{SPEEDRICHR_URL}/{endpoint}', timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def query(self, genes: list[str], databases: list[str], description: str) -> pd.DataFrame:
        if self.background_id is None:
            self.background_id = self._post('addbackground', data={'background': '\n'.join(self.background)})['backgroundid']
        user_list = self._post('addList', files={'list': (None, '\n'.join(genes)), 'description': (None, description)})['userListId']
        rows = []
        for database in databases:
            result = self._post('backgroundenrich', data={
                'userListId': user_list, 'backgroundid': self.background_id, 'backgroundType': database,
            })
            if database not in result or not isinstance(result[database], list):
                raise ValueError(f'Malformed Enrichr response for {database}')
            for entry in result[database]:
                if len(entry) < 7:
                    raise ValueError(f'Malformed Enrichr term for {database}')
                hits = entry[5]
                rows.append(dict(database=database, rank=entry[0], term=entry[1], pvalue=entry[2],
                                 zscore=entry[3], combined_score=entry[4],
                                 genes=';'.join(hits) if isinstance(hits, list) else str(hits), padj=entry[6]))
        return pd.DataFrame(rows, columns=ENRICHMENT_COLUMNS).sort_values('padj', kind='stable').reset_index(drop=True)


def run_functional_enrichment(gene_tables: dict, config: dict, *, session=None,
                              on_error: str = 'raise') -> dict:
    """Return {program: {'core': table, 'specific': table}}, using Layer 0 flags.

    Empty/skipped/failed tables have explicit status metadata in .attrs. No
    usage matrix or direction result is read. Unmapped ENSG IDs are reported;
    symbol deduplication applies equally to queries and their background.
    """
    if on_error not in ('raise', 'record'):
        raise ValueError('on_error must be raise or record')
    universe = pac_universe(gene_tables)
    background = _symbols(universe['symbol'])
    if not background:
        message = 'PAC universe has no mapped gene symbols for enrichment'
        if on_error == 'raise':
            raise ValueError(message)
        results = {}
        for program in gene_tables:
            results[program] = {}
            for kind in ('core', 'specific'):
                table = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
                table.attrs.update(status='failed', error=message, program=program, gene_set=kind,
                                   n_background_genes=0, n_pac_universe_genes=len(universe),
                                   n_unmapped_background_genes=len(universe), background_scope='pac_universe')
                results[program][kind] = table
        return results
    unmapped = len(universe) - int(universe['symbol'].map(lambda x: bool(_symbols([x]))).sum())
    if unmapped:
        warnings.warn(f'{unmapped} PAC-universe genes lack usable symbols and cannot be sent to Enrichr', stacklevel=2)
    databases = config.get('enrichr_databases', DEFAULT_ENRICHR_DATABASES)
    if not databases or not isinstance(databases, (list, tuple)):
        raise ValueError('enrichr_databases must be a nonempty list')
    minimum = int(config.get('enrichr_min_genes', 3))
    if minimum < 1:
        raise ValueError('enrichr_min_genes must be positive')
    results = {}
    with BackgroundEnrichrClient(background, session=session, timeout=config.get('enrichr_timeout', 30)) as client:
        for program, genes in gene_tables.items():
            results[program] = {}
            for kind, flag in [('core', 'is_core'), ('specific', 'is_specific')]:
                query = _symbols(genes.loc[genes[flag], 'symbol'])
                table = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
                metadata = dict(program=program, gene_set=kind, n_query_genes=len(query),
                                n_background_genes=len(background), n_pac_universe_genes=len(universe),
                                n_unmapped_background_genes=unmapped, background_scope='pac_universe')
                if len(query) < minimum:
                    metadata['status'] = 'skipped_insufficient_genes'
                else:
                    try:
                        table = client.query(query, list(databases), f'{program}_{kind}')
                        metadata['status'] = 'ok' if len(table) else 'empty'
                    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                        if on_error == 'raise':
                            raise
                        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
                table.attrs.update(metadata)
                results[program][kind] = table
    return results
