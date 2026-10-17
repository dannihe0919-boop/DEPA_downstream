"""Program report: direct length signs, zero handling and sample inference."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from depa_downstream import (build_program_gene_tables, compute_contrast_association,
                             summarize_program_associations, compute_population_direction)
from depa_downstream.program_report import (
    generate_program_report, cleanup_report_figures, REPORT_FIGURES, _preference, _audit_wpsi_range,
)
from test_workflow import fixture, FakeEnrichr
from depa_downstream import run_functional_enrichment


class ProgramReportTests(unittest.TestCase):
    def test_cleanup_removes_only_known_duplicates_with_pdf_counterparts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stem = 'P02_A_identity_usage'
            (out / f'{stem}.pdf').write_bytes(b'%PDF-1.4\nexisting PDF')
            for ext in ('png', 'svg'):
                (out / f'{stem}.{ext}').write_bytes(b'duplicate')
            preserved = [
                'P02_custom_plot.png', 'P04_A_identity_usage.png',
                'layer3_population_direction.png', 'P02_core_gene_statistics.csv',
                'P02_B_pooled_preference.png',  # No PDF: keep the only copy.
                'P02_C_core_gene_length_changes.png',
            ]
            for name in preserved:
                (out / name).write_bytes(b'keep')
            (out / 'P02_C_core_gene_length_changes.pdf').write_bytes(b'not a PDF')
            removed = cleanup_report_figures(out, program='P02')
            self.assertEqual({Path(p).suffix for p in removed}, {'.png', '.svg'})
            self.assertTrue(all((out / name).is_file() for name in preserved))
            self.assertTrue((out / f'{stem}.pdf').is_file())
            self.assertEqual(cleanup_report_figures(out, program='P02'), [])

    def _range_fixture(self):
        table = pd.DataFrame({'gene_id': ['core', 'background'], 'symbol': ['C', 'B'],
                              'is_core': [True, False], 'is_wpsi_reliable': [True, True],
                              'gene_wpsi_median': [.2, .5]})
        matrix = pd.DataFrame([[.1, .3], [.4, .6]], index=table.gene_id, columns=['A', 'B'])
        return table, matrix

    def test_floating_boundary_deviations_are_retained_and_audited(self):
        table, matrix = self._range_fixture()
        matrix.loc['core'] = [-1e-12, 1 + 1e-8]
        original = matrix.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertWarnsRegex(UserWarning, 'within_floating_tolerance'):
                summary = _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)
            examples = pd.read_csv(Path(directory)/'P02_wpsi_range_examples.csv')
            self.assertEqual(summary['n_strict_outside_0_1'], 2)
            self.assertEqual(summary['n_outside_tolerance'], 0)
            self.assertFalse(summary['values_modified'])
            self.assertLess(examples.wpsi.min(), 0)
            self.assertGreater(examples.wpsi.max(), 1)
        pd.testing.assert_frame_equal(matrix, original)

    def test_noncore_cell_outlier_does_not_block_core_report(self):
        table, matrix = self._range_fixture()
        matrix.loc['background', 'A'] = 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertWarnsRegex(UserWarning, 'noncore_outliers_only'):
                summary = _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)
        self.assertEqual(summary['n_noncore_outside_tolerance'], 1)
        self.assertEqual(summary['n_core_outside_tolerance'], 0)

    def test_material_core_outlier_exports_diagnostics_before_stopping(self):
        table, matrix = self._range_fixture()
        matrix.loc['core', 'A'] = -.2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'min=-0.2'):
                _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)
            saved = json.loads((Path(directory)/'P02_wpsi_range_summary.json').read_text())
            self.assertEqual(saved['status'], 'blocked_material_outliers')
            self.assertEqual(saved['n_core_outside_tolerance'], 1)
            self.assertTrue((Path(directory)/'P02_wpsi_range_by_gene.csv').is_file())

    def test_percentage_like_scale_is_not_automatically_rescaled(self):
        table, matrix = self._range_fixture()
        matrix *= 100
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'do not automatically divide by 100'):
                _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)
        self.assertEqual(matrix.loc['core', 'A'], 10)

    def test_invalid_reliable_cached_background_median_is_not_ignored(self):
        table, matrix = self._range_fixture()
        table.loc[1, 'gene_wpsi_median'] = 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, '1 cached medians used by the report'):
                _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)

    def test_unreliable_core_cached_median_is_still_audited_for_the_pooled_plot(self):
        table, matrix = self._range_fixture()
        table.loc[0, 'is_wpsi_reliable'] = False
        table.loc[0, 'gene_wpsi_median'] = -99
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, '1 cached medians used by the report'):
                _audit_wpsi_range(matrix, table, directory, 'P02', 1e-6)

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
            for name in REPORT_FIGURES:
                for extension in ('png', 'svg'):
                    (Path(directory) / f'P02_{name}.{extension}').write_bytes(b'old duplicate')
            report = generate_program_report(data, config, tables, summary, directions,
                output_dir=directory, key_genes=['G06', 'G07', 'G08', 'G09'],
                min_cells_per_sample=5, enrichment=enrichment)
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
                self.assertEqual(len(paths), 1)
                self.assertEqual(Path(paths[0]).suffix, '.pdf')
                self.assertTrue(all(Path(p).stat().st_size > 100 for p in paths))
            self.assertEqual(len(report['removed_figure_files']), 14)
            self.assertFalse(list(Path(directory).glob('P02_*.png')))
            self.assertFalse(list(Path(directory).glob('P02_*.svg')))
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
