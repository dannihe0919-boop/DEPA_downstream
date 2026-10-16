"""Statistical regression and pipeline checks, using no patient data or network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu

from depa_downstream import (
    annotate_within_celltype, build_program_gene_tables, compute_contrast_association,
    compute_population_direction, integrate_results, run_functional_enrichment,
    summarize_program_associations,
    plot_enrichment_bubble,
)
from depa_downstream.layer1_enrichment import BackgroundEnrichrClient
from depa_downstream.layer2_association import _build_label
from depa_downstream.pipeline import load_analysis_config, main, run_analysis


def fixture():
    genes = [f'G{i:02}' for i in range(20)]
    pacs = [f'PAC{i:02}' for i in range(20)]
    cells = [f'c{i:03}' for i in range(160)]
    groups = np.repeat(['Healthy', 'Disease'], 80)
    types = np.tile(np.repeat(['Alpha', 'Beta'], 40), 2)
    meta = pd.DataFrame({'group': groups, 'cell_type': types,
                         'sample': [f'{g}_{(i % 80) // 20}' for i, g in enumerate(groups)]}, index=cells)
    usage = pd.DataFrame({
        'P01': np.where(types == 'Alpha', 0.7, 0.1) + (groups == 'Disease') * 0.1,
        'P02': np.where(types == 'Alpha', 0.8, 0.3) - (groups == 'Disease') * 0.1,
    }, index=cells)
    z = np.full((2, 20), -0.5)
    z[0, :6] = 2
    z[0, 0] = 1.5  # Inclusive boundary is intentional.
    z[1, 6:12] = 2
    z = pd.DataFrame(z, index=['P01', 'P02'], columns=pacs)
    rng = np.random.default_rng(10)
    wpsi = pd.DataFrame(rng.uniform(.49, .51, (20, 160)), index=genes, columns=cells)
    wpsi.iloc[:6] = .8
    wpsi.iloc[0] = np.where(groups == 'Disease', .85, .75)
    wpsi.iloc[1] = np.where(groups == 'Disease', .75, .85)
    wpsi.iloc[3, 80:] = np.nan
    wpsi.iloc[4] = 0
    wpsi.iloc[5] = np.nan
    wpsi.iloc[6:12] = .2
    wpsi.iloc[6] = np.where(groups == 'Disease', .25, .15)
    wpsi.iloc[7] = np.where(groups == 'Disease', .15, .25)
    data = dict(spectra=z + 1, spectra_z=z, feature_anno=pd.DataFrame({'gene_id': genes}, index=pacs),
                wpsi=wpsi, usage=usage, meta=meta, programs=['P01', 'P02'], gene_symbol_map=None)
    config = dict(z_threshold=1.5, spec_min_z=.5, min_detection_rate=.1, min_fg_genes=2,
                  min_cells_per_group=5, min_samples_per_group=2,
                  meta_columns={'celltype': 'cell_type', 'sample': 'sample'},
                  enrichr_databases=['TEST_DB'],
                  contrasts=[{'name': 'disease', 'column': 'group', 'type': 'pairwise', 'group_a': 'Healthy', 'group_b': 'Disease', 'category': 'disease'},
                             {'name': 'identity', 'column': 'cell_type', 'type': 'auto_one_vs_rest_all'}])
    return data, config


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeEnrichr:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.queries = {}
        self.counter = 0

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail:
            raise requests.HTTPError('403 network denied')
        if url.endswith('/addbackground'):
            return FakeResponse({'backgroundid': 'bg123'})
        if url.endswith('/addList'):
            self.counter += 1
            self.queries[self.counter] = kwargs['files']['list'][1].split('\n')
            return FakeResponse({'userListId': self.counter})
        if url.endswith('/backgroundenrich'):
            payload = kwargs['data']
            assert payload['backgroundid'] == 'bg123'
            hits = self.queries[payload['userListId']][:3]
            return FakeResponse({payload['backgroundType']: [[1, 'Synthetic pathway', .001, 2, 12, hits, .004]]})
        raise AssertionError(url)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = fixture()

    def test_identity_flags_and_reliability_do_not_remove_core_genes(self):
        tables = build_program_gene_tables(self.data, self.config)
        table = tables['P01'].set_index('gene_id')
        self.assertEqual(len(table), 20)
        self.assertEqual(int(table.is_core.sum()), 6)
        self.assertTrue(table.loc['G00', 'is_core'])
        self.assertTrue(table.loc['G04', 'is_core'])
        self.assertFalse(table.loc['G04', 'is_wpsi_reliable'])
        self.assertFalse(table.loc['G05', 'is_wpsi_reliable'])
        self.assertEqual(sorted(table.rank_in_program), list(range(1, 21)))

    def test_duplicate_pacs_collapse_to_one_gene_with_maximum_weight(self):
        self.data['feature_anno'].loc['PAC01', 'gene_id'] = 'G00'
        table = build_program_gene_tables(self.data, self.config)['P01'].set_index('gene_id')
        self.assertEqual(len(table), 19)
        self.assertEqual(table.loc['G00', 'spectra_z'], 2)
        self.assertEqual(table.loc['G00', 'spectra_score'], 3)

    def test_missing_wpsi_does_not_block_identity_or_enrichment(self):
        del self.data['wpsi']
        tables = build_program_gene_tables(self.data, self.config)
        self.assertTrue(tables['P01'].is_wpsi_reliable.isna().all())
        enriched = run_functional_enrichment(tables, self.config, session=FakeEnrichr())
        self.assertEqual(enriched['P01']['core'].attrs['status'], 'ok')
        self.assertTrue(compute_population_direction(tables, self.config).status.eq('missing_wpsi').all())

    def test_specificity_uses_other_program_gene_z_and_strict_threshold(self):
        self.data['spectra_z'].loc['P02', 'PAC00'] = .5
        table = build_program_gene_tables(self.data, self.config)['P01'].set_index('gene_id')
        self.assertFalse(table.loc['G00', 'is_specific'])
        self.assertTrue(table.loc['G01', 'is_specific'])

    def test_single_program_specificity(self):
        for name in ('spectra', 'spectra_z'):
            self.data[name] = self.data[name].iloc[:1]
        table = build_program_gene_tables(self.data, self.config)['P01']
        self.assertEqual(int(table.is_specific.sum()), int(table.is_core.sum()))

    def test_zscores_can_be_computed_once_from_raw_spectra(self):
        del self.data['spectra_z']
        table = build_program_gene_tables(self.data, self.config)['P01']
        self.assertTrue(np.isfinite(table.spectra_z).all())
        self.assertAlmostEqual(table.spectra_z.mean(), 0)


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = fixture()
        self.tables = build_program_gene_tables(self.data, self.config)

    def test_custom_universe_uploaded_once_for_both_gene_sets_and_programs(self):
        session = FakeEnrichr()
        result = run_functional_enrichment(self.tables, self.config, session=session)
        backgrounds = [kw['data']['background'] for url, kw in session.calls if url.endswith('/addbackground')]
        self.assertEqual(len(backgrounds), 1)
        self.assertEqual(set(backgrounds[0].splitlines()), {f'G{i:02}' for i in range(20)})
        self.assertEqual(len([u for u, _ in session.calls if u.endswith('/backgroundenrich')]), 4)
        self.assertEqual(result['P01']['specific'].iloc[0]['genes'], 'G00;G01;G02')
        self.assertEqual(result['P01']['core'].attrs['n_background_genes'], 20)

    def test_low_detection_core_is_in_enrichment_query(self):
        session = FakeEnrichr()
        run_functional_enrichment(self.tables, self.config, session=session)
        self.assertIn('G04', session.queries[1])
        self.assertIn('G05', session.queries[1])

    def test_no_genome_fallback_and_failures_are_explicit(self):
        with self.assertRaises(requests.HTTPError):
            run_functional_enrichment(self.tables, self.config, session=FakeEnrichr(fail=True))
        result = run_functional_enrichment(self.tables, self.config, session=FakeEnrichr(fail=True), on_error='record')
        self.assertEqual(result['P01']['core'].attrs['status'], 'failed')
        self.assertTrue(result['P01']['core'].empty)

    def test_empty_specific_set_has_skip_status_without_network(self):
        for table in self.tables.values():
            table['is_specific'] = False
        result = run_functional_enrichment(self.tables, self.config, session=FakeEnrichr())
        self.assertEqual(result['P01']['specific'].attrs['status'], 'skipped_insufficient_genes')

    def test_unmapped_ids_are_reported_and_background_deduplicated(self):
        for table in self.tables.values():
            table.loc[table.gene_id.isin(['G00', 'G01']), 'symbol'] = 'SHARED'
            table.loc[table.gene_id.eq('G02'), 'symbol'] = 'ENSG00000000002'
        with self.assertWarns(UserWarning):
            result = run_functional_enrichment(self.tables, self.config, session=FakeEnrichr())
        self.assertEqual(result['P01']['core'].attrs['n_background_genes'], 18)
        self.assertEqual(result['P01']['core'].attrs['n_unmapped_background_genes'], 1)

    def test_malformed_server_response_is_not_successful_empty_result(self):
        client = BackgroundEnrichrClient(['A'], session=FakeEnrichr())
        with patch.object(client, '_post', side_effect=[{'backgroundid': 'b'}, {'userListId': 1}, {}]):
            with self.assertRaisesRegex(ValueError, 'Malformed'):
                client.query(['A'], ['DB'], 'check')


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = fixture()

    def associations(self):
        with patch('depa_downstream.layer2_association._HAS_XGB', False):
            return compute_contrast_association({'usage': self.data['usage'], 'meta': self.data['meta']}, self.config)

    def test_usage_has_no_spectra_or_direction_dependency(self):
        results = self.associations()
        summary = summarize_program_associations(results, self.config)
        self.assertEqual(summary.loc['P01', 'best_celltype'], 'Alpha')
        self.assertEqual(summary.loc['P01', 'best_celltype_n_cells'], 80)
        self.assertEqual(summary.loc['P01', 'best_celltype_auc'], 1)
        self.assertNotIn('direction', results['disease'])

    def test_negative_association_is_considered_by_absolute_auc_and_counts_positive_group(self):
        results = self.associations()
        results['Alpha'].loc['P01', 'auc'] = .65
        results['Beta'].loc['P01', 'auc'] = .05
        results['Beta'].loc['P01', 'n_group_b'] = 17
        summary = summarize_program_associations(results, self.config)
        self.assertEqual(summary.loc['P01', 'best_celltype'], 'Beta')
        self.assertEqual(summary.loc['P01', 'best_celltype_n_cells'], 17)
        self.assertEqual(summary.loc['P01', 'best_celltype_association'], 'depleted')

    def test_unknown_metadata_is_not_rest(self):
        self.data['meta'].loc['c000', 'cell_type'] = None
        labels = _build_label(self.data['meta'], {'type': 'one_vs_rest', 'column': 'cell_type', 'group': 'Alpha'})
        self.assertNotIn('c000', labels.index)

    def test_untested_identity_has_no_manual_fallback(self):
        self.config['min_cells_per_group'] = 1000
        summary = summarize_program_associations(self.associations(), self.config, programs=['P01', 'P02'])
        self.assertTrue(summary.best_celltype.isna().all())
        self.assertTrue(summary.best_celltype_n_cells.eq(0).all())

    def test_ambiguous_disease_requires_explicit_selection(self):
        self.config['contrasts'].append({**self.config['contrasts'][0], 'name': 'another'})
        with self.assertRaisesRegex(ValueError, 'Multiple disease'):
            summarize_program_associations({}, self.config, programs=['P01'])
        self.config['disease_contrast'] = 'another'
        summary = summarize_program_associations({}, self.config, programs=['P01'])
        self.assertEqual(summary.loc['P01', 'disease_contrast'], 'another')


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = fixture()
        self.tables = build_program_gene_tables(self.data, self.config)
        self.population = compute_population_direction(self.tables, self.config)
        self.summary = pd.DataFrame({'best_celltype': ['Alpha', 'Alpha'], 'best_celltype_n_cells': [80, 80]}, index=['P01', 'P02'])

    def annotate(self):
        return annotate_within_celltype(self.tables, self.data, self.config, self.summary, self.population)

    def test_population_matches_independent_mwu_and_uses_layer0_flags(self):
        row = self.population.loc['P01']
        table = self.tables['P01']
        reliable = table.is_wpsi_reliable.fillna(False)
        expected = mannwhitneyu(table.loc[table.is_core & reliable, 'gene_wpsi_median'],
                               table.loc[~table.is_core & reliable, 'gene_wpsi_median']).pvalue
        self.assertAlmostEqual(row.pval, expected)
        self.assertEqual(row.n_fg_genes, 4)
        self.assertEqual(row.expected_direction, 'increase')
        self.assertEqual(self.population.loc['P02', 'expected_direction'], 'decrease')
        modified = {p: t.copy() for p, t in self.tables.items()}
        modified['P01']['is_core'] = False
        self.assertEqual(compute_population_direction(modified, self.config).loc['P01', 'status'], 'insufficient_foreground')

    def test_all_core_genes_annotated_in_place_and_old_identity_preserved(self):
        before = self.tables['P01'][['gene_id', 'spectra_z', 'is_core', 'is_wpsi_reliable']].copy()
        result = self.annotate()
        self.assertIs(result, self.tables)
        pd.testing.assert_frame_equal(before, result['P01'][before.columns])
        table = result['P01'].set_index('gene_id')
        self.assertEqual(table.loc['G00', 'concordance'], 'concordant')
        self.assertEqual(table.loc['G01', 'concordance'], 'discordant')
        self.assertEqual(table.loc['G02', 'concordance'], 'inconclusive')
        self.assertEqual(table.loc['G03', 'within_celltype_status'], 'insufficient_cells')
        self.assertEqual(table.loc['G03', 'concordance'], 'not_tested')
        self.assertEqual(table.loc['G04', 'within_celltype_detection_rate_a'], 0)
        self.assertEqual(table.loc['G04', 'within_celltype_nonzero_frac_a'], 0)

    def test_decreasing_usage_does_not_flip_concordance(self):
        table = self.annotate()['P02'].set_index('gene_id')
        self.assertEqual(table.loc['G06', 'expected_direction'], 'decrease')
        self.assertEqual(table.loc['G06', 'within_celltype_expected_direction'], 'increase')
        self.assertGreater(table.loc['G06', 'within_celltype_mean_diff'], 0)
        self.assertEqual(table.loc['G06', 'concordance'], 'concordant')
        self.assertEqual(table.loc['G07', 'concordance'], 'discordant')

    def test_insufficient_celltype_is_not_silently_replaced(self):
        self.summary.loc['P01', 'best_celltype_n_cells'] = 1
        table = self.annotate()['P01']
        self.assertTrue(table.loc[table.is_core, 'within_celltype_status'].eq('insufficient_cells').all())
        self.assertTrue(table.within_celltype_p.isna().all())

    def test_insufficient_biological_samples_is_explicit(self):
        self.data['meta']['sample'] = self.data['meta']['group']
        table = self.annotate()['P01']
        self.assertTrue(table.loc[table.gene_id.eq('G00'), 'within_celltype_status'].eq('insufficient_samples').all())

    def test_zero_within_usage_change_cannot_be_called_concordant(self):
        self.data['usage']['P01'] = 1.
        table = self.annotate()['P01'].set_index('gene_id')
        self.assertEqual(table.loc['G00', 'concordance'], 'inconclusive')
        self.assertEqual(table.loc['G00', 'within_celltype_status'], 'unknown_expected_direction')

    def test_repeated_annotation_clears_stale_results(self):
        self.annotate()
        self.data['wpsi'] = None
        table = self.annotate()['P01']
        self.assertTrue(table.within_celltype_p.isna().all())
        self.assertTrue(table.concordance.eq('not_tested').all())


class IntegrationAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = fixture()

    def run_pipeline(self, **kwargs):
        with patch('depa_downstream.layer2_association._HAS_XGB', False):
            return run_analysis(self.data, self.config, **kwargs)

    def test_full_pipeline_exports_final_gene_tables_and_term_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_pipeline(enrichment_session=FakeEnrichr(), output_root=directory)
            row = result['integrated_enrichment']['P01']['core'].iloc[0]
            self.assertEqual(row.n_concordant, 1)
            self.assertEqual(row.n_discordant, 1)
            self.assertEqual(row.n_inconclusive, 1)
            self.assertEqual(row.concordant_genes, 'G00')
            exported = pd.read_csv(Path(directory) / 'layer0' / 'P01.csv')
            self.assertIn('concordance', exported)
            manifest = json.loads((Path(directory) / 'analysis_manifest.json').read_text())
            self.assertEqual(manifest['execution_order'], ['layer0', 'layer1', 'layer2', 'layer3a', 'layer3b', 'integrate_results'])
            self.assertEqual(manifest['status'], 'completed')

    def test_enrichment_failure_still_runs_independent_usage_and_direction(self):
        result = self.run_pipeline(enrichment_session=FakeEnrichr(fail=True))
        self.assertEqual(result['manifest']['status'], 'partial')
        self.assertEqual(result['population_direction'].loc['P01', 'status'], 'tested')
        self.assertEqual(result['gene_tables']['P01'].set_index('gene_id').loc['G00', 'concordance'], 'concordant')

    def test_explicit_offline_mode_never_queries_network(self):
        with patch('requests.Session.post', side_effect=AssertionError('network used')):
            result = self.run_pipeline(enrichment_enabled=False)
        self.assertEqual(result['enrichment']['P01']['core'].attrs['status'], 'disabled')

    def test_missing_wpsi_preserves_usage_and_enrichment(self):
        self.data['wpsi'] = None
        result = self.run_pipeline(enrichment_session=FakeEnrichr())
        self.assertEqual(result['enrichment']['P01']['core'].attrs['status'], 'ok')
        self.assertEqual(result['association_summary'].loc['P01', 'best_celltype'], 'Alpha')
        self.assertEqual(result['population_direction'].loc['P01', 'status'], 'missing_wpsi')

    def test_no_mapped_background_does_not_stop_independent_analysis(self):
        self.data['gene_symbol_map'] = pd.Series({f'G{i:02}': f'ENSG{i:011}' for i in range(20)})
        result = self.run_pipeline(enrichment_session=FakeEnrichr())
        self.assertEqual(result['manifest']['status'], 'partial')
        self.assertEqual(result['enrichment']['P01']['core'].attrs['status'], 'failed')
        self.assertEqual(result['population_direction'].loc['P01', 'status'], 'tested')

    def test_integration_deduplicates_hits_and_keeps_unassessed_separate(self):
        result = self.run_pipeline(enrichment_enabled=False)
        terms = pd.DataFrame({'genes': ['G00;G00;G01;G02;G03;UNKNOWN']})
        integrated = integrate_results({'P01': {'core': terms}}, result['gene_tables'])['P01']['core']
        self.assertEqual(integrated.loc[0, 'n_concordant'], 1)
        self.assertEqual(integrated.loc[0, 'n_unassessed'], 1)
        self.assertEqual(integrated.loc[0, 'n_unmapped_hit_symbols'], 1)
        self.assertNotIn('n_concordant', terms)

    def test_integration_cannot_run_before_direction_annotation(self):
        tables = build_program_gene_tables(self.data, self.config)
        with self.assertRaisesRegex(ValueError, 'Layer 3b'):
            integrate_results({'P01': {'core': pd.DataFrame({'genes': []})}}, tables)

    def test_config_resolves_relative_paths_once_and_accepts_absolute_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('root_dir: /old/mac\nh5ad_path: data/counts.h5ad\noutput_root: outputs\n')
            config = load_analysis_config(path, root_dir=directory, path_overrides={'h5ad_path': '/absolute/counts.h5ad'})
            self.assertEqual(config['h5ad_path'], '/absolute/counts.h5ad')
            self.assertEqual(config['output_root'], str(Path(directory) / 'outputs'))

    def test_infinite_enrichr_scores_do_not_hide_hits_in_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        table = pd.DataFrame({'term': ['Infinite score', 'Finite score'], 'padj': [.001, .002],
                              'combined_score': [np.inf, 5.]})
        with tempfile.TemporaryDirectory() as directory, patch('matplotlib.pyplot.close', wraps=plt.close) as close:
            path = plot_enrichment_bubble({'P01': table}, directory)
            figure = close.call_args[0][0]
            points = figure.axes[0].collections[0]
            self.assertFalse(np.ma.getmaskarray(points.get_offsets()).any())
            self.assertTrue(np.isfinite(points.get_array()).all())
            with Image.open(path) as image:
                image.verify()
        self.assertTrue(np.isinf(table.loc[0, 'combined_score']))

    def test_cli_reads_cnmf_and_h5ad_files_and_exports_offline_results(self):
        import anndata as ad
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'cnmf' / 'test'
            run.mkdir(parents=True)
            for kind in ('gene_spectra_score', 'gene_spectra_tpm'):
                self.data['spectra'].to_csv(run / f'test.{kind}.k_2.dt_0_01.txt', sep='\t')
            self.data['usage'].to_csv(run / 'test.usages.k_2.dt_0_01.consensus.txt', sep='\t')
            ad.AnnData(X=np.zeros((160, 20)), obs=self.data['meta'], var=self.data['feature_anno']).write_h5ad(root / 'counts.h5ad')
            self.data['feature_anno'].to_csv(root / 'features.csv')
            config = dict(self.config, cnmf_dir='cnmf', run_name='test', K=2, density_threshold='0_01',
                          h5ad_path='counts.h5ad', feature_anno_path='features.csv', output_root='results')
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            with patch('depa_downstream.layer2_association._HAS_XGB', False):
                status = main(['--config', str(config_path), '--root-dir', str(root), '--skip-enrichment'])
            self.assertEqual(status, 0)
            summary = pd.read_csv(root / 'results' / 'layer2' / 'program_association_summary.csv')
            self.assertEqual(summary.loc[summary.program.eq('P01'), 'best_celltype'].iloc[0], 'Alpha')
            manifest = json.loads((root / 'results' / 'analysis_manifest.json').read_text())
            self.assertEqual(manifest['population_status']['P01'], 'missing_wpsi')


if __name__ == '__main__':
    unittest.main()
