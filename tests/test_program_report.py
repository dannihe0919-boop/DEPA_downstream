"""Program report: direct length signs, zero handling and sample inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from depa_downstream import (build_program_gene_tables, compute_contrast_association,
                             summarize_program_associations, compute_population_direction)
from depa_downstream.program_report import generate_program_report, _preference
from test_workflow import fixture, FakeEnrichr
from depa_downstream import run_functional_enrichment


class ProgramReportTests(unittest.TestCase):
    def test_another_program_uses_its_own_genes_labels_and_output_prefix(self):
        data, config = fixture()
        tables = build_program_gene_tables(data, config)
        associations = compute_contrast_association(data, config)
        summary = summarize_program_associations(associations, config, programs=list(tables))
        directions = compute_population_direction(tables, config)
        with tempfile.TemporaryDirectory() as directory:
            result = generate_program_report(data, config, tables, summary, directions,
                program='P01', output_dir=directory, key_genes=['G00'], formats=('svg',))
            self.assertEqual(result['summary']['program'], 'P01')
            self.assertEqual(set(result['genes'].gene_id), {f'G{i:02}' for i in range(6)})
            self.assertEqual(result['key_genes'].gene_id.tolist(), ['G00'])
            self.assertTrue((Path(directory)/'P01_report_summary.json').is_file())
            self.assertTrue((Path(directory)/'P01_core_gene_statistics.csv').is_file())
            self.assertFalse(list(Path(directory).glob('P02_*')))
            self.assertTrue(all(Path(path).name.startswith('P01_')
                                for paths in result['figures'].values() for path in paths))

    def test_length_labels_use_actual_wpsi_sign_and_selected_significance(self):
        delta = pd.Series([.1, -.1, .001, .1, -.1])
        raw_p = pd.Series([.01, .01, .001, .04, np.nan])
        q = pd.Series([.02, .02, .005, .08, np.nan])
        eligible = pd.Series([True]*5)
        self.assertEqual(_preference(delta, raw_p, eligible, .05, .01).tolist(),
                         ['longer', 'shorter', 'inconclusive', 'longer', 'not_tested'])
        self.assertEqual(_preference(delta, q, eligible, .05, .01).tolist(),
                         ['longer', 'shorter', 'inconclusive', 'inconclusive', 'not_tested'])

    def test_real_export_preserves_zero_one_nan_and_existing_tables(self):
        data, config = fixture()
        data['wpsi'].loc['G08'] = 0
        data['wpsi'].loc['G09', data['meta'].group.eq('Disease')] = np.nan
        data['wpsi'].loc['G10'] = 1
        tables = build_program_gene_tables(data, config)
        before = tables['P02'].copy(deep=True)
        associations = compute_contrast_association(data, config)
        summary = summarize_program_associations(associations, config, programs=list(tables))
        directions = compute_population_direction(tables, config)
        enrichment = run_functional_enrichment(tables, config, session=FakeEnrichr())
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'population_direction.csv'
            sentinel.write_text('existing results')
            report = generate_program_report(data, config, tables, summary, directions,
                output_dir=directory, key_genes=['G06', 'G07', 'G08', 'G09'],
                min_cells_per_sample=5, enrichment=enrichment, formats=('png', 'pdf', 'svg'))
            genes = report['genes'].set_index('gene_id')
            self.assertEqual(genes.loc['G06', 'report_length_preference'], 'longer')
            self.assertEqual(genes.loc['G07', 'report_length_preference'], 'shorter')
            self.assertEqual(genes.loc['G08', 'within_celltype_mean_a'], 0)
            self.assertEqual(genes.loc['G08', 'report_length_preference'], 'inconclusive')
            self.assertEqual(genes.loc['G10', 'within_celltype_mean_b'], 1)
            self.assertTrue(np.isnan(genes.loc['G09', 'within_celltype_mean_b']))
            self.assertEqual(genes.loc['G09', 'report_length_preference'], 'not_tested')
            self.assertEqual(genes.loc['G09', 'sample_status'], 'insufficient_samples')
            self.assertEqual(genes.loc['G06', 'sample_n_a'], 2)
            self.assertEqual(genes.loc['G06', 'sample_n_b'], 2)
            self.assertGreater(genes.loc['G06', 'sample_padj'], .05)
            self.assertTrue(report['sample_wpsi'].loc[lambda t: t.gene_id.eq('G08'),
                                                     'wpsi_zero_fraction'].eq(1).all())
            self.assertEqual(sentinel.read_text(), 'existing results')
            self.assertEqual(len(report['figures']), 7)
            for paths in report['figures'].values():
                self.assertEqual(len(paths), 3)
                self.assertTrue(all(Path(p).stat().st_size > 100 for p in paths))
            self.assertTrue((Path(directory)/'P02_core_gene_statistics.csv').is_file())
            self.assertTrue((Path(directory)/'P02_report_summary.json').is_file())
            self.assertEqual(report['enrichment_overlap'].iloc[0]['n_longer'], 1)
            self.assertEqual(report['enrichment_overlap'].iloc[0]['n_shorter'], 1)
        pd.testing.assert_frame_equal(tables['P02'], before)

    def test_missing_sample_metadata_keeps_cell_analysis_and_skips_sample_tests(self):
        data, config = fixture()
        data['meta'] = data['meta'].drop(columns='sample')
        config['min_samples_per_group'] = 1
        tables = build_program_gene_tables(data, config)
        associations = compute_contrast_association(data, config)
        summary = summarize_program_associations(associations, config, programs=list(tables))
        directions = compute_population_direction(tables, config)
        with tempfile.TemporaryDirectory() as directory:
            result = generate_program_report(data, config, tables, summary, directions,
                output_dir=directory, key_genes=['G06'], formats=('svg',))
        self.assertTrue(result['genes'].sample_status.eq('missing_sample_metadata').all())
        self.assertTrue(result['genes'].sample_p.isna().all())
        self.assertEqual(result['genes'].set_index('gene_id').loc['G06',
                        'report_length_preference'], 'longer')
        self.assertTrue(result['usage_bins'].empty)


if __name__ == '__main__':
    unittest.main()
