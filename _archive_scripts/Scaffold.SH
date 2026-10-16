#!/bin/bash
set -e
cd /Volumes/Elements/DEPA_downstream   # 如果你的项目根目录不是这个路径，先改这里

mkdir -p depa_downstream configs notebooks

cat > depa_downstream/config_schema.py << 'PYEOF'
"""
CONFIG 字段校验。

设计原则：
  - 必需字段缺失 -> 立即报错，报清楚缺了哪个字段，不要让后面的函数
    因为 KeyError 报出一个莫名其妙的位置。
  - 可选字段（比如 wpsi_matrix_path / depa_feature_source_path）允许缺失，
    对应的 Layer 函数内部自己判断要不要跳过。
"""

from __future__ import annotations

REQUIRED_FIELDS = [
    "cnmf_dir",       # cNMF 输出所在目录
    "run_name",       # cNMF run 的名字，用于拼文件名
    "K",              # cNMF 的 K 值
    "density_threshold",  # 例如 "0_01"
    "h5ad_path",      # cNMF 输入的 h5ad（含 metadata）
    "feature_anno_path",  # PAC -> gene_id 等注释表
    "meta_columns",   # dict，见下方说明
    "contrasts",      # list[dict]，见下方说明
    "output_root",    # 所有结果输出的根目录
]

# meta_columns 期望的结构（都是 cnmf_meta.csv / h5ad.obs 里的列名）：
#   {
#       "sample": "sample",
#       "celltype": "cell_type",
#       "umap1": "UMAP_1",       # 可选
#       "umap2": "UMAP_2",       # 可选
#   }
REQUIRED_META_COLUMNS = ["sample", "celltype"]

# contrasts 里每一条期望的结构，两种 type：
#   {"name": "disease", "column": "group", "type": "pairwise",
#    "group_a": "Healthy", "group_b": "Severe_COVID"}
#   {"name": "platelet_id", "column": "cell_type", "type": "one_vs_rest",
#    "group": "Platelet"}
VALID_CONTRAST_TYPES = ["pairwise", "one_vs_rest"]

# 可选字段一览（不校验存在性，只在这里留文档，方便回头查）：
#   wpsi_matrix_path          — Layer 1 用，没有则 Layer 1 跳过并给出提示
#   psi_matrix_path           — 同上
#   depa_feature_source_path  — Layer 3 的 infer_program_modules 用，
#                                没有则退回手动 MODULE_DEF
#   module_def                — dict，手动指定的功能模块划分（可选，
#                                优先级低于 depa_feature_source_path 自动推断）
#   z_threshold / top_n_pacs / min_cells_per_group / ... — 各层分析阈值，
#                                缺失时每个函数使用自己的默认值


class ConfigError(ValueError):
    pass


def validate_config(config: dict) -> None:
    """校验 CONFIG 是否包含所有必需字段，字段格式是否合理。

    Parameters
    ----------
    config : dict
        从 yaml / 手写 dict 加载的完整配置。

    Raises
    ------
    ConfigError
        缺少必需字段，或字段格式不对时抛出，信息里会指明具体缺了什么。
    """
    missing = [f for f in REQUIRED_FIELDS if f not in config]
    if missing:
        raise ConfigError(f"CONFIG 缺少必需字段: {missing}")

    meta_cols = config["meta_columns"]
    missing_meta = [c for c in REQUIRED_META_COLUMNS if c not in meta_cols]
    if missing_meta:
        raise ConfigError(f"CONFIG['meta_columns'] 缺少必需子字段: {missing_meta}")

    if not isinstance(config["contrasts"], list) or len(config["contrasts"]) == 0:
        raise ConfigError("CONFIG['contrasts'] 必须是非空 list")

    for i, ctr in enumerate(config["contrasts"]):
        if "name" not in ctr or "column" not in ctr or "type" not in ctr:
            raise ConfigError(f"CONFIG['contrasts'][{i}] 缺少 name/column/type 之一: {ctr}")
        if ctr["type"] not in VALID_CONTRAST_TYPES:
            raise ConfigError(
                f"CONFIG['contrasts'][{i}]['type']={ctr['type']!r} 不合法，"
                f"必须是 {VALID_CONTRAST_TYPES} 之一"
            )
        if ctr["type"] == "pairwise" and not ("group_a" in ctr and "group_b" in ctr):
            raise ConfigError(f"CONFIG['contrasts'][{i}] type=pairwise 需要 group_a/group_b: {ctr}")
        if ctr["type"] == "one_vs_rest" and "group" not in ctr:
            raise ConfigError(f"CONFIG['contrasts'][{i}] type=one_vs_rest 需要 group: {ctr}")
PYEOF

cat > depa_downstream/io_layer.py << 'PYEOF'
"""
读取 cNMF 输出、metadata、feature 注释，统一封装成一个 data 字典。

所有下游 Layer 函数都只接收这个 data 字典（+ config），不直接碰文件路径，
这样 Layer 函数和"数据从哪读"彻底解耦。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config_schema import validate_config


def _build_cnmf_paths(config: dict) -> dict:
    """根据 CONFIG 拼出 cNMF 三件套的具体文件路径。"""
    base = config["cnmf_dir"]
    name = config["run_name"]
    K = config["K"]
    dt = config["density_threshold"]
    return {
        "spectra_score": f"{base}/{name}/{name}.gene_spectra_score.k_{K}.dt_{dt}.txt",
        "spectra_tpm": f"{base}/{name}/{name}.gene_spectra_tpm.k_{K}.dt_{dt}.txt",
        "usage": f"{base}/{name}/{name}.usages.k_{K}.dt_{dt}.consensus.txt",
    }


def load_cnmf_outputs(config: dict) -> dict:
    """读取 cNMF spectra/usage + metadata + feature 注释，返回统一的 data 字典。

    Parameters
    ----------
    config : dict
        完整 CONFIG（会先跑 validate_config 校验）。

    Returns
    -------
    dict，包含以下 key：
        "spectra"       : DataFrame, programs x PACs (原始 spectra_score)
        "spectra_z"     : DataFrame, programs x PACs (对 spectra 做全局 Z-score)
        "usage"         : DataFrame, cells x programs
        "meta"          : DataFrame, cells x metadata 列（index=cell barcode）
        "feature_anno"  : DataFrame, index=PAC id，至少含 gene_id 列
        "programs"      : list[str]，program 名称（如 ["P01", ..., "P06"]）
        "wpsi"          : DataFrame or None，gene x cell（若 config 里给了
                           wpsi_matrix_path 且文件存在才读取，否则为 None）
        "psi"           : DataFrame or None，同上
        "depa_feature_source" : DataFrame or None，若 config 里给了
                           depa_feature_source_path 才读取，用于 Layer3
                           的 program -> context 自动反查
    """
    validate_config(config)
    paths = _build_cnmf_paths(config)

    print("检查 cNMF 输入文件...")
    for key, p in paths.items():
        status = "OK" if os.path.exists(p) else "缺失"
        size = f"{os.path.getsize(p) / 1e6:.1f} MB" if os.path.exists(p) else "-"
        print(f"  [{status}] {key}: {p} ({size})")

    # TODO: 实现读取逻辑（可直接照搬原 notebook Cell1 里对应部分）
    #   spectra = pd.read_csv(paths["spectra_score"], sep="\t", index_col=0)
    #   spectra_z = (spectra - spectra.mean()) / spectra.std()  # 或用 scipy.stats.zscore
    #   usage = pd.read_csv(paths["usage"], sep="\t", index_col=0)
    #   meta = ... 从 h5ad_path 或 cnmf_meta.csv 读（config 里指定用哪个）
    #   feature_anno = pd.read_csv(config["feature_anno_path"], index_col=0)
    #   programs = list(spectra.index)
    raise NotImplementedError("待填充：读取 spectra/usage/meta/feature_anno 的具体逻辑")


def load_wpsi_psi(config: dict) -> dict:
    """按需读取 wPSI / PSI 矩阵（.rds，用 pyreadr）。

    没有配置对应路径，或文件不存在时返回 {"wpsi": None, "psi": None}，
    并打印提示——调用方（Layer1）据此判断是否跳过 wPSI 相关分析，
    而不是直接报错中断整个流程。
    """
    # TODO: 实现，参考原 notebook Cell 用 pyreadr.read_r(...)[None]
    raise NotImplementedError


def load_depa_feature_source(config: dict) -> pd.DataFrame | None:
    """按需读取 DEPA_feature_source.csv（Layer3 的 program->context 反查用）。

    这是可选输入：不是所有数据集在构建 cNMF 输入时都经过
    scAPAflow::build_depa_feature_object 这一步（比如最早的 ALL 白血病数据
    就没有），所以缺失时返回 None，不报错。
    """
    # TODO: 实现
    raise NotImplementedError
PYEOF

cat > depa_downstream/layer1_direction.py << 'PYEOF'
"""
Layer 1 — 程序特征基础：基于 wPSI 的 APA 方向验证（MWU版）。

对应原 notebook 的 Cell 2。核心逻辑不依赖数据集语义（不管 program 是按疾病
分组还是按细胞身份/状态分组跑出来的，方向检验的算法是一样的），
所以这层几乎不需要因为数据集不同而改函数本身，只是入参（wpsi/spectra_z）不同。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def compute_program_direction(
    data: dict,
    config: dict,
    min_fg_genes: int = 5,
) -> pd.DataFrame | None:
    """对每个 program 做 wPSI 前景 vs 背景的 MWU 检验，判定 Proximal/Distal 方向。

    Parameters
    ----------
    data : dict
        io_layer.load_cnmf_outputs() 的返回值，需要用到
        data["spectra_z"] / data["feature_anno"] / data["programs"] / data["wpsi"]。
    config : dict
        用到 config 里的 "z_threshold"（默认 1.5，前景PAC筛选阈值）。
    min_fg_genes : int
        前景基因数少于这个值的 program 跳过，不做检验（样本太小检验没意义）。

    Returns
    -------
    DataFrame，index=program，列包括：
        n_fg_genes, delta_median, mwu_stat, pval, padj, direction
        (direction 取值 "Proximal" / "Distal")
    若 data["wpsi"] 为 None（该数据集没有配置 wPSI 矩阵），打印提示并返回 None，
    调用方（Layer2/Layer3 的可视化部分）需要对 None 做判断，不能默认它一定存在。
    """
    if data.get("wpsi") is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过 APA 方向检验（compute_program_direction 返回 None）")
        return None

    z_threshold = config.get("z_threshold", 1.5)
    # TODO: 实现，照搬原 notebook Cell2 的核心逻辑：
    #   1. wpsi 按基因取跨细胞中位数
    #   2. 对每个 program 取 spectra_z 高于 z_threshold 的 PAC -> 映射到基因 -> 前景基因集
    #   3. MWU 检验前景 vs 全局背景
    #   4. FDR 校正，按 delta_median 符号判定 Proximal/Distal
    raise NotImplementedError


def plot_program_direction_summary(program_direction: pd.DataFrame, config: dict, out_dir: str) -> None:
    """画方向检验的汇总图（原 Fig D：MWU统计条形图）。

    传入的是 compute_program_direction() 的结果，不重新计算。
    """
    raise NotImplementedError
PYEOF

cat > depa_downstream/layer2_association.py << 'PYEOF'
"""
Layer 2 — 疾病/条件差异分析：program usage 与任意 contrast 的关联度。

对应原 notebook 的 Cell 3，但原版写死了"疾病 AUC"一个二分类。
这里改成对 config["contrasts"] 里的每一条独立跑一遍，函数本身不关心
某条 contrast 在语义上是"疾病 vs 健康"还是"细胞身份 vs 其余"——
全部当成"某一列的某种分组方式"来处理，保证跨数据集复用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _build_label(meta: pd.DataFrame, contrast: dict) -> pd.Series:
    """根据一条 contrast 配置，从 meta 里构建 0/1 标签（Series，index=cell）。

    pairwise   : 只保留 group_a/group_b 两组细胞，其余细胞不参与（label为NaN会被剔除）
    one_vs_rest: group 组记为1，其余全部记为0
    """
    col = contrast["column"]
    if contrast["type"] == "pairwise":
        mask = meta[col].isin([contrast["group_a"], contrast["group_b"]])
        label = (meta.loc[mask, col] == contrast["group_b"]).astype(int)
        return label
    elif contrast["type"] == "one_vs_rest":
        label = (meta[col] == contrast["group"]).astype(int)
        return label
    else:
        raise ValueError(f"未知 contrast type: {contrast['type']}")


def compute_contrast_association(
    data: dict,
    config: dict,
    program_direction: pd.DataFrame | None = None,
) -> dict:
    """对 config["contrasts"] 里每一条，计算各 program usage 与该分组的关联度。

    Parameters
    ----------
    data : dict
        需要 data["usage"] (cells x programs) 和 data["meta"]。
    config : dict
        用到 config["contrasts"]，以及可选的 min_cells_per_group。
    program_direction : DataFrame, optional
        Layer1 的输出，用于后续画图时标注方向（可以为 None，不影响本函数计算）。

    Returns
    -------
    dict，key 是每条 contrast 的 "name"，value 是一个 DataFrame：
        index=program，列：auc, lr_coef（逻辑回归系数，标准化后）,
        dt_importance（决策树特征重要性）, xgb_importance（若可用）,
        n_group_a, n_group_b（各组细胞数，用于判断这条对比是否可靠）
    """
    usage = data["usage"]
    meta = data["meta"]
    min_cells = config.get("min_cells_per_group", 20)

    results = {}
    for contrast in config["contrasts"]:
        name = contrast["name"]
        # TODO: 实现，照搬原 notebook Cell3 的分类器集成逻辑，
        # 但标签来自 _build_label(meta, contrast) 而不是写死的 ALL/Normal
        # 记得检查每组细胞数 >= min_cells，不够的话跳过这条 contrast 并打印提示
        raise NotImplementedError

    return results


def plot_association_heatmap(assoc_results: dict, config: dict, out_dir: str) -> None:
    """画所有 contrast x program 的关联度汇总热图。"""
    raise NotImplementedError


def plot_program_umap_grid(data: dict, config: dict, out_dir: str) -> None:
    """全 program 的 UMAP 网格图（每个 program 一张小图，按 usage 染色）。

    需要 config["meta_columns"] 里配置了 umap1/umap2，没配置则跳过并提示。
    """
    if "umap1" not in config["meta_columns"] or "umap2" not in config["meta_columns"]:
        print("  [Layer2] 未配置 UMAP 坐标列，跳过 UMAP 网格图")
        return
    raise NotImplementedError
PYEOF

cat > depa_downstream/layer3_markers.py << 'PYEOF'
"""
Layer 3 — Marker 基因富集分析：程序间基因重叠 / 专属marker筛选 / 功能富集 /
program -> 功能模块 自动推断。

对应原 notebook 的 Cell 4-7。其中 infer_program_modules() 是这次新增的
（原来 MODULE_DEF 是手写死表），用 DEPA_feature_source.csv 反查每个 program
的 PAC 主要来自哪个 context，自动给出模块标签初稿。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import requests


def compute_program_gene_sets(data: dict, config: dict) -> dict:
    """为每个 program 构建核心基因集（Z > z_threshold，同基因取最高Z值）。

    Returns
    -------
    dict: {"gene_sets": {program: set(gene_id)}, "gene_z": {program: {gene_id: z}}}
    """
    raise NotImplementedError


def compute_jaccard_matrix(gene_sets: dict, programs: list) -> pd.DataFrame:
    """计算 program 两两之间基因集的 Jaccard 相似度矩阵。"""
    raise NotImplementedError


def select_program_markers(
    gene_sets: dict,
    config: dict,
    top_n: int = 20,
    spec_min_z: float = 0.5,
) -> dict:
    """为每个 program 筛选"专属" marker 基因（其他 program 里该基因 Z 值都很低）。

    Returns
    -------
    dict: {program: DataFrame(columns=[gene_id, symbol, z_focal, ...])}
    """
    raise NotImplementedError


def run_enrichment(
    gene_symbols_by_program: dict,
    config: dict,
    databases: list[str] | None = None,
) -> dict:
    """调用 Enrichr API，对每个 program 的基因列表做功能富集。

    Parameters
    ----------
    gene_symbols_by_program : dict
        {program: [symbol, ...]}，可以传全部显著基因，也可以传 marker 基因，
        调用两次分别处理"输入A/输入B"，不在函数内部做区分。
    databases : list[str], optional
        默认用 config.get("enrichr_databases", [...默认四个库...])

    Returns
    -------
    dict: {program: DataFrame(columns=[term, pval, padj, ...])}
    """
    raise NotImplementedError


def infer_program_modules(
    data: dict,
    config: dict,
    gene_sets: dict,
    top_n_context: int = 3,
) -> pd.DataFrame | None:
    """反查每个 program 的 top PAC 主要来自 DEPA_feature_source.csv 里的哪些 context，
    自动生成"program -> 主导 context / 模块"标签初稿。

    只有当 data["depa_feature_source"] 不为 None 时才能跑（这是可选输入，
    不是所有数据集在构建 cNMF 输入时都经过 scAPAflow::build_depa_feature_object）。
    没有的话返回 None，调用方应该退回使用 config.get("module_def")（手动指定的表）。

    Returns
    -------
    DataFrame，index=program，列：
        top_contexts（该 program 的 PAC 主要来自的 context，按占比降序，取前
        top_n_context 个），dominant_context（占比最高的一个），
        dominant_fraction（占比数值，供人工判断这个自动标签靠不靠谱——
        比如占比只有 20% 说明这个 program 是多个 context 混合，不该简单打单一标签）
    """
    if data.get("depa_feature_source") is None:
        print("  [Layer3] 未提供 DEPA_feature_source.csv，跳过模块自动推断，"
              "请在 config['module_def'] 里手动指定")
        return None
    raise NotImplementedError


def resolve_module_labels(
    data: dict,
    config: dict,
    gene_sets: dict,
) -> pd.DataFrame:
    """统一入口：优先用 infer_program_modules() 自动推断，失败/没有数据则退回
    config['module_def']（手动查表 program -> module 名），两者都没有则每个
    program 各自成一个模块（不合并，保证下游画图不报错）。
    """
    auto = infer_program_modules(data, config, gene_sets)
    if auto is not None:
        return auto
    manual = config.get("module_def")
    if manual is not None:
        # TODO: 把手动 dict {module_name: [program, ...]} 转成 index=program 的表
        raise NotImplementedError
    print("  [Layer3] 未提供自动推断数据也未提供 module_def，每个 program 独立成模块")
    programs = data["programs"]
    return pd.DataFrame({"module": programs}, index=programs)
PYEOF

cat > depa_downstream/viz.py << 'PYEOF'
"""
纯画图函数。原则：只接收已经算好的结果（DataFrame/dict），不在这里做任何
统计计算——计算逻辑都应该在 layer1/2/3 里，方便单独测试和复用。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_jaccard_heatmap(jaccard_mat: pd.DataFrame, out_path: str) -> None:
    raise NotImplementedError


def plot_enrichment_bubble(enrichment_by_program: dict, out_path: str, top_n: int = 15) -> None:
    raise NotImplementedError


def plot_marker_wpsi_violin(marker_wpsi_records: list[dict], out_path: str) -> None:
    raise NotImplementedError
PYEOF

cat > depa_downstream/__init__.py << 'PYEOF'
from .config_schema import validate_config, ConfigError
from .io_layer import load_cnmf_outputs, load_wpsi_psi, load_depa_feature_source
from .layer1_direction import compute_program_direction, plot_program_direction_summary
from .layer2_association import compute_contrast_association, plot_association_heatmap, plot_program_umap_grid
from .layer3_markers import (
    compute_program_gene_sets,
    compute_jaccard_matrix,
    select_program_markers,
    run_enrichment,
    infer_program_modules,
    resolve_module_labels,
)

__all__ = [
    "validate_config", "ConfigError",
    "load_cnmf_outputs", "load_wpsi_psi", "load_depa_feature_source",
    "compute_program_direction", "plot_program_direction_summary",
    "compute_contrast_association", "plot_association_heatmap", "plot_program_umap_grid",
    "compute_program_gene_sets", "compute_jaccard_matrix", "select_program_markers",
    "run_enrichment", "infer_program_modules", "resolve_module_labels",
]
PYEOF

cat > configs/covid19.yaml << 'YAMLEOF'
# COVID-19 数据集配置
# 对应目录：/Volumes/Elements/data/covid_19_data/

cnmf_dir: "/Volumes/Elements/data/covid_19_data/cnmf_output_identity_state"
run_name: "COVID_DEAPA_programs"
K: 6
density_threshold: "0_01"

h5ad_path: "/Volumes/Elements/data/covid_19_data/cnmf_output_identity_state/COVID_DEAPA_1231PAC_counts.h5ad"
feature_anno_path: "/Volumes/Elements/data/covid_19_data/cnmf_input_identity_state/cnmf_feature_anno.csv"

# 可选：没有就不填，对应 Layer1 会自动跳过
wpsi_matrix_path: "/Volumes/Elements/data/covid_19_data/PSI_results/movAPA_wPSI_matrix.rds"
psi_matrix_path: "/Volumes/Elements/data/covid_19_data/PSI_results/movAPA_PSI_matrix.rds"

# 可选：Layer3 自动推断模块用，来自 scAPAflow::build_depa_feature_object 的输出
depa_feature_source_path: "/Volumes/Elements/data/covid_19_data/cnmf_input_identity_state/DEPA_feature_source.csv"

meta_columns:
  sample: "sample"
  celltype: "cell_type"
  umap1: "UMAP_1"
  umap2: "UMAP_2"

# 允许任意多条、任意类型的对比，不再限定"疾病vs正常"
contrasts:
  - name: "disease"
    column: "group"
    type: "pairwise"
    group_a: "Healthy"
    group_b: "Severe_COVID"
  # 如果之后想看某个身份/状态本身与 program usage 的关联，可以继续加：
  # - name: "platelet_identity"
  #   column: "cell_type"
  #   type: "one_vs_rest"
  #   group: "Platelet"

# 可选：如果不想等 infer_program_modules 自动推断，也可以手动指定粗分组
# module_def:
#   Myeloid: [P01, P02]
#   Lymphoid: [P03, P04]

z_threshold: 1.5
top_n_pacs: 100
min_cells_per_group: 20
min_samples_per_group: 2
enrichr_databases:
  - "GO_Biological_Process_2023"
  - "KEGG_2021_Human"
  - "MSigDB_Hallmark_2020"
  - "Reactome_2022"
enrichr_cutoff_padj: 0.05
diff_padj_cutoff: 0.05

output_root: "/Volumes/Elements/data/covid_19_data/9.results/"
YAMLEOF

echo "骨架文件已生成，目录结构："
find depa_downstream configs -type f