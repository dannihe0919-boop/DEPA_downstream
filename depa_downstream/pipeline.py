"""Portable entry points and reproducible exports for the Layer 0–3 workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .integration import integrate_results
from .io_layer import load_cnmf_outputs
from .layer0_identity import build_program_gene_tables
from .layer1_enrichment import ENRICHMENT_COLUMNS, run_functional_enrichment
from .layer2_association import compute_contrast_association, summarize_program_associations
from .layer3_direction import annotate_within_celltype, compute_population_direction

PATH_KEYS = ['cnmf_dir', 'h5ad_path', 'feature_anno_path', 'wpsi_matrix_path', 'psi_matrix_path',
             'depa_feature_source_path', 'gene_symbol_map_path', 'output_root']


def load_analysis_config(path, *, root_dir=None, output_root=None, path_overrides=None) -> dict:
    """Resolve paths once. Explicit overrides also support old absolute Mac paths."""
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    root = Path(root_dir or config.get('root_dir') or path.parent.parent).expanduser().resolve()
    for key, value in (path_overrides or {}).items():
        if key not in PATH_KEYS:
            raise ValueError(f'Unsupported path override: {key}')
        config[key] = value
    if output_root is not None:
        config['output_root'] = str(output_root)
    for key in PATH_KEYS:
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str((root / value).resolve())
    config['root_dir'] = str(root)
    return config


def export_analysis(results: dict, output_root) -> None:
    """Export final Layer 0 tables and every layer, including empty/error statuses."""
    root = Path(output_root)

    def write_table(table, path, *, index=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=index)

    def filename(value):
        # Program IDs and contrast names may come from metadata, not safe paths.
        from urllib.parse import quote
        return quote(str(value), safe='') + '.csv'

    for program, table in results['gene_tables'].items():
        write_table(table, root / 'layer0' / filename(program))
    for field, directory in [('enrichment', 'layer1'), ('integrated_enrichment', 'integrated')]:
        for program, versions in results[field].items():
            for kind, table in versions.items():
                write_table(table, root / directory / kind / filename(program))
    for contrast, table in results['associations'].items():
        if table is not None:
            write_table(table, root / 'layer2' / 'contrasts' / filename(contrast), index=True)
    write_table(results['association_summary'], root / 'layer2' / 'program_association_summary.csv', index=True)
    write_table(results['population_direction'], root / 'layer3' / 'population_direction.csv', index=True)
    (root / 'analysis_manifest.json').write_text(json.dumps(results['manifest'], indent=2, ensure_ascii=False) + '\n')


def run_analysis(data: dict, config: dict, *, enrichment_enabled=True, enrichment_session=None,
                 output_root=None) -> dict:
    """Run 0 → 1 → 2 → 3a → 3b → integration. No manual cell-type choices.

    Record enrichment service failures and complete independent local layers.
    The returned manifest marks failed/empty/skipped/disabled requests, so an
    external failure can never be reported as a successful enrichment run.
    """
    gene_tables = build_program_gene_tables(data, config)
    if enrichment_enabled:
        enrichment = run_functional_enrichment(gene_tables, config, session=enrichment_session, on_error='record')
    else:
        enrichment = {}
        for program in gene_tables:
            enrichment[program] = {}
            for kind in ('core', 'specific'):
                table = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
                table.attrs.update(status='disabled', program=program, gene_set=kind)
                enrichment[program][kind] = table
    # Usage analysis consumes only usage + metadata; direction is computed later.
    associations = compute_contrast_association({'usage': data['usage'], 'meta': data['meta']}, config)
    summary = summarize_program_associations(associations, config, programs=list(gene_tables))
    directions = compute_population_direction(gene_tables, config)
    annotate_within_celltype(gene_tables, data, config, summary, directions)
    result = assemble_analysis_results(gene_tables, enrichment, associations, summary, directions, config)
    if output_root is not None:
        export_analysis(result, output_root)
    return result


def assemble_analysis_results(gene_tables, enrichment, associations, summary, directions, config):
    """Integrate already computed Notebook stages without recomputing any layer."""
    integrated = integrate_results(enrichment, gene_tables)
    statuses = {program: {kind: dict(table.attrs) for kind, table in versions.items()}
                for program, versions in enrichment.items()}
    failed = any(table.attrs['status'] == 'failed' for versions in enrichment.values() for table in versions.values())
    manifest = dict(status='partial' if failed else 'completed', enrichment=statuses,
                    association_status={name: 'skipped' if table is None else 'tested' for name, table in associations.items()},
                    population_status=directions['status'].to_dict(),
                    within_celltype_status={p: t.loc[t['is_core'], 'within_celltype_status'].value_counts().to_dict()
                                            for p, t in gene_tables.items()},
                    gene_table_metadata={p: dict(t.attrs) for p, t in gene_tables.items()},
                    configuration=config, execution_order=['layer0', 'layer1', 'layer2', 'layer3a', 'layer3b', 'integrate_results'])
    result = dict(gene_tables=gene_tables, enrichment=enrichment, associations=associations,
                  association_summary=summary, population_direction=directions,
                  integrated_enrichment=integrated, manifest=manifest)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='DEPA Layer 0–3 analysis and result integration')
    parser.add_argument('--config', required=True, help='YAML configuration')
    parser.add_argument('--root-dir', help='Base directory for relative paths')
    parser.add_argument('--output-root', help='Override the output directory')
    parser.add_argument('--path', action='append', default=[], metavar='KEY=PATH', help='Override a single input path; repeatable')
    parser.add_argument('--skip-enrichment', action='store_true', help='Explicit offline run; enrichment is recorded as disabled')
    args = parser.parse_args(argv)
    overrides = {}
    for entry in args.path:
        key, separator, value = entry.partition('=')
        if not separator or not value:
            parser.error('--path requires KEY=PATH')
        overrides[key] = value
    config = load_analysis_config(args.config, root_dir=args.root_dir, output_root=args.output_root, path_overrides=overrides)
    data = load_cnmf_outputs(config)
    result = run_analysis(data, config, enrichment_enabled=not args.skip_enrichment, output_root=config['output_root'])
    print(json.dumps({'status': result['manifest']['status'], 'output_root': config['output_root']}, ensure_ascii=False))
    return 2 if result['manifest']['status'] == 'partial' else 0


if __name__ == '__main__':
    raise SystemExit(main())
