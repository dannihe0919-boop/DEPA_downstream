"""Public API for the final Layer0–3 workflow and result integration."""
from .config_schema import ConfigError, validate_config
from .io_layer import load_cnmf_outputs, load_gene_symbol_map, load_wpsi
from .layer0_identity import build_program_gene_tables, pac_universe
from .layer1_enrichment import run_functional_enrichment
from .layer2_association import compute_contrast_association, summarize_program_associations
from .layer3_direction import annotate_within_celltype, compute_population_direction
from .integration import integrate_results
from .viz import (
    plot_association_heatmap,
    plot_enrichment_bubble,
    plot_population_direction,
    plot_program_umap_grid,
    plot_within_celltype_concordance,
)

__all__ = [
    'ConfigError', 'validate_config',
    'load_cnmf_outputs', 'load_gene_symbol_map', 'load_wpsi',
    'build_program_gene_tables', 'pac_universe',
    'run_functional_enrichment',
    'compute_contrast_association', 'summarize_program_associations',
    'compute_population_direction', 'annotate_within_celltype',
    'integrate_results',
    'plot_association_heatmap', 'plot_enrichment_bubble', 'plot_population_direction',
    'plot_program_umap_grid', 'plot_within_celltype_concordance',
]
