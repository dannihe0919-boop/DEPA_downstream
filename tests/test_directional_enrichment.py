"""Separate length-group enrichment: background, failures and honest plot selection."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depa_downstream import build_program_gene_tables
from depa_downstream.layer1_enrichment import ENRICHMENT_COLUMNS
from depa_downstream.program_report import run_directional_enrichment, _plot_directional_enrichment
from test_workflow import fixture, FakeEnrichr


class DirectionalEnrichmentTests(unittest.TestCase):
    def setUp(self):
        data, self.config = fixture()
        self.tables = build_program_gene_tables(data, self.config)
        self.genes = self.tables['P02'].loc[lambda t: t.is_core].copy()
        self.genes['report_length_preference'] = ['shorter'] * 3 + ['longer'] * 3

    def test_two_queries_share_full_background_and_export_pdf_only(self):
        session = FakeEnrichr()
        original = self.genes.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            result = run_directional_enrichment(self.genes, self.tables, self.config,
                                                session=session, output_dir=directory)
            self.assertEqual(result['status'].status.tolist(), ['ok', 'ok'])
            self.assertTrue(Path(result['figure']).read_bytes().startswith(b'%PDF-'))
            self.assertEqual(len(list(Path(directory).glob('*.pdf'))), 1)
            self.assertFalse(list(Path(directory).glob('*.png')))
            self.assertFalse(list(Path(directory).glob('*.svg')))
            saved = json.loads((Path(directory) / 'P02_directional_enrichment_summary.json').read_text())
            self.assertEqual(saved['groups']['shorter']['n_background_symbols'], 20)
            self.assertEqual(len(result['displayed_terms']), 2)
        backgrounds = [kw['data']['background'].split('\n') for url, kw in session.calls
                       if url.endswith('/addbackground')]
        self.assertEqual(len(backgrounds), 1)
        self.assertEqual(len(backgrounds[0]), 20)
        self.assertEqual(list(session.queries.values()),
                         [sorted(self.genes.iloc[:3].symbol), sorted(self.genes.iloc[3:].symbol)])
        self.assertFalse(set(session.queries[1]) & set(session.queries[2]))
        pd.testing.assert_frame_equal(self.genes, original)

    def test_unmapped_identifier_is_audited_and_not_queried(self):
        unmapped = self.genes.iloc[0].gene_id
        for table in self.tables.values():
            table.loc[table.gene_id.eq(unmapped), 'symbol'] = 'ENSG_UNMAPPED'
        self.genes.loc[self.genes.gene_id.eq(unmapped), 'symbol'] = 'ENSG_UNMAPPED'
        self.config['enrichr_min_genes'] = 2
        session = FakeEnrichr()
        with tempfile.TemporaryDirectory() as directory:
            result = run_directional_enrichment(self.genes, self.tables, self.config,
                                                session=session, output_dir=directory)
            status = result['status'].set_index('group').loc['shorter']
            self.assertEqual(status.n_selected_genes, 3)
            self.assertEqual(status.n_query_symbols, 2)
            self.assertEqual(status.n_unmapped_query_genes, 1)
            self.assertEqual(status.n_background_symbols, 19)
            self.assertFalse(result['gene_lists']['shorter'].iloc[0].sent_to_enrichr)
        self.assertNotIn('ENSG_UNMAPPED', session.queries[1])

    def test_insufficient_group_is_not_queried(self):
        self.genes.iloc[:3, self.genes.columns.get_loc('report_length_preference')] = 'inconclusive'
        session = FakeEnrichr()
        with tempfile.TemporaryDirectory() as directory:
            result = run_directional_enrichment(self.genes, self.tables, self.config,
                                                session=session, output_dir=directory)
            self.assertEqual(result['status'].status.tolist(), ['skipped_insufficient_genes', 'ok'])
            self.assertTrue(result['enrichment']['shorter'].empty)
        self.assertEqual(len(session.queries), 1)

    def test_failures_are_recorded_without_genome_fallback(self):
        session = FakeEnrichr(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertWarnsRegex(UserWarning, 'enrichment failed'):
                result = run_directional_enrichment(self.genes, self.tables, self.config,
                                                    session=session, output_dir=directory)
            self.assertEqual(result['status'].status.tolist(), ['failed', 'failed'])
            self.assertTrue(result['displayed_terms'].empty)
            self.assertTrue(Path(result['figure']).exists())
        self.assertTrue(all(url.endswith('/addbackground') for url, _ in session.calls))

    def test_invalid_symbol_mapping_is_rejected_before_network(self):
        self.genes.iloc[0, self.genes.columns.get_loc('symbol')] = 'WRONG'
        session = FakeEnrichr()
        with self.assertRaisesRegex(ValueError, 'symbols differ'):
            run_directional_enrichment(self.genes, self.tables, self.config, session=session)
        self.assertEqual(session.calls, [])

    def test_selected_key_genes_cannot_silently_replace_full_core_set(self):
        session = FakeEnrichr()
        with self.assertRaisesRegex(ValueError, 'all core rows'):
            run_directional_enrichment(self.genes.iloc[:3], self.tables, self.config, session=session)
        self.assertEqual(session.calls, [])

    def test_no_significant_terms_has_explicit_placeholder(self):
        empty = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
        status = {g: dict(n_query_symbols=3, status='empty') for g in ('shorter', 'longer')}
        fig, shown = _plot_directional_enrichment({'shorter': empty, 'longer': empty}, status,
                                                 program='P02', alpha=.05, top_n=10)
        self.assertTrue(shown.empty)
        self.assertTrue(any('No significant terms' in t.get_text() for t in fig.axes[0].texts))
        plt.close(fig)

    def test_top_term_union_includes_other_groups_significant_match(self):
        def table(rows):
            return pd.DataFrame([dict(database='TEST', rank=i, term=term, pvalue=q,
                                      zscore=2, combined_score=score, genes='A;B', padj=q)
                                 for i, (term, q, score) in enumerate(rows)], columns=ENRICHMENT_COLUMNS)
        results = {'shorter': table([('shared', .001, np.inf), ('short', .02, 12)]),
                   'longer': table([('long', .001, 25), ('shared', .02, 20)])}
        status = {g: dict(n_query_symbols=3, status='ok') for g in results}
        fig, shown = _plot_directional_enrichment(results, status, program='P02', alpha=.05, top_n=1)
        self.assertEqual(set(zip(shown.group, shown.term)),
                         {('shorter', 'shared'), ('longer', 'shared'), ('longer', 'long')})
        self.assertTrue(np.isinf(results['shorter'].iloc[0].combined_score))
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
