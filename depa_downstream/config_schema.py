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
VALID_CONTRAST_TYPES = ["pairwise", "one_vs_rest", "auto_one_vs_rest_all"]

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
