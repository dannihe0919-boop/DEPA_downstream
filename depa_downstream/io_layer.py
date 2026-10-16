"""
读取 cNMF 输出、metadata、feature 注释，统一封装成一个 data 字典。

所有下游 Layer 函数都只接收这个 data 字典（+ config），不直接碰文件路径，
这样 Layer 函数和"数据从哪读"彻底解耦。
"""

from __future__ import annotations

import os

import pandas as pd
from scipy.stats import zscore

from .config_schema import validate_config


def _build_cnmf_paths(config: dict) -> dict:
    """根据 CONFIG 拼出 cNMF 权重与 usage 输出的具体文件路径。"""
    base = config["cnmf_dir"]
    name = config["run_name"]
    K = config["K"]
    dt = config["density_threshold"]
    return {
        "spectra_score": f"{base}/{name}/{name}.gene_spectra_score.k_{K}.dt_{dt}.txt",
        "usage": f"{base}/{name}/{name}.usages.k_{K}.dt_{dt}.consensus.txt",
    }


def _check_files_exist(paths: dict) -> None:
    print("检查 cNMF 输入文件...")
    missing = []
    for key, p in paths.items():
        exists = os.path.exists(p)
        size = f"{os.path.getsize(p) / 1e6:.1f} MB" if exists else "-"
        status = "OK" if exists else "缺失"
        print(f"  [{status}] {key}: {p} ({size})")
        if not exists:
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"以下 cNMF 输出文件不存在，请检查 CONFIG 里的 "
                                 f"cnmf_dir/run_name/K/density_threshold 是否正确: {missing}")


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
        "spectra_z"     : DataFrame, programs x PACs (对每个 program 行做 Z-score，
                           即在该 program 内部比较 PAC 之间的相对权重)
        "usage"         : DataFrame, cells x programs
        "meta"          : DataFrame, cells x metadata 列（index=cell barcode），
                           从 h5ad 的 .obs 读取
        "feature_anno"  : DataFrame, index=PAC id，至少含 gene_id 列
        "programs"      : list[str]，program 名称（如 ["P01", ..., "P06"]）
        "wpsi"          : DataFrame or None，gene x cell
    """
    validate_config(config)
    paths = _build_cnmf_paths(config)
    _check_files_exist(paths)

    # ── 1. spectra / usage ──────────────────────────────────────
    print("\n读取 cNMF spectra 与 usage...")
    spectra = pd.read_csv(paths["spectra_score"], sep="\t", index_col=0)
    usage = pd.read_csv(paths["usage"], sep="\t", index_col=0)

    # program 命名统一化：不同数据集的 cNMF 输出表头格式不一致
    # （有的是 "P01".."P18" 字符串，有的是纯数字 1..6），统一转成
    # 零填充字符串（如 K=6 -> "P01".."P06"，K=120 -> "P001".."P120"），
    # 保证下游所有函数（尤其是手动指定的 contrasts 里用到的
    # program 标签）不用关心这层格式差异。
    K = config["K"]
    pad_width = max(2, len(str(K)))
    def _normalize_program_label(x):
        s = str(x).strip()
        # 已经是 "P01" 这种格式就直接保留数字部分重新格式化，
        # 否则原样当作数字处理
        digits = "".join(ch for ch in s if ch.isdigit()) or s
        try:
            return f"P{int(digits):0{pad_width}d}"
        except ValueError:
            return s  # 实在不是数字格式的，原样保留，不强行转换报错

    spectra.index = [_normalize_program_label(p) for p in spectra.index]
    usage.columns = [_normalize_program_label(p) for p in usage.columns]

    programs = list(spectra.index)
    print(f"  spectra: {spectra.shape}  (programs x PACs)")
    print(f"  usage:   {usage.shape}  (cells x programs)")
    print(f"  programs (已统一命名): {programs}")

    # 每个 program（行）内部做 Z-score，用于后续"该 program 里哪些 PAC 权重显著偏高"的判断
    print("\n计算 spectra 逐 program 的 Z-score...")
    z_vals = zscore(spectra.values, axis=1, nan_policy="omit")
    spectra_z = pd.DataFrame(z_vals, index=spectra.index, columns=spectra.columns)
    print(f"  spectra_z: {spectra_z.shape}")

    # ── 2. metadata（从 h5ad 的 .obs 读）──────────────────────────
    print("\n读取 metadata (h5ad .obs)...")
    import anndata as ad
    adata = ad.read_h5ad(config["h5ad_path"], backed="r")
    try:
        meta = adata.obs.copy()
    finally:
        adata.file.close()
    print(f"  meta: {meta.shape}  (cells x columns)")

    meta_cols = config["meta_columns"]
    missing_cols = [c for c in meta_cols.values() if c not in meta.columns]
    if missing_cols:
        raise KeyError(
            f"config['meta_columns'] 里指定的列在 h5ad.obs 中找不到: {missing_cols}\n"
            f"  h5ad.obs 实际列名: {list(meta.columns)}"
        )
    for role, col in meta_cols.items():
        print(f"  {role:10s} -> 列 '{col}'，取值样例: {meta[col].dropna().unique()[:5].tolist()}")

    # contrasts 里用到的列也顺带检查一遍，缺失就提前报错，不要等 Layer2 才炸
    for ctr in config["contrasts"]:
        if ctr["column"] not in meta.columns:
            raise KeyError(f"contrast {ctr['name']!r} 用到的列 {ctr['column']!r} 不在 meta 里，"
                            f"实际列名: {list(meta.columns)}")

    common_cells = usage.index.intersection(meta.index)
    print(f"\n  usage 与 meta 共有细胞数: {len(common_cells)} "
          f"(usage独有 {len(usage.index.difference(meta.index))}, "
          f"meta独有 {len(meta.index.difference(usage.index))})")
    if len(common_cells) == 0:
        raise ValueError("usage 和 meta 的细胞 barcode 没有任何交集，检查两者index格式是否一致"
                          "（比如是否都带/都不带 '-1' 后缀）")

    # ── 3. feature 注释 ──────────────────────────────────────────
    print("\n读取 feature 注释...")
    feature_anno = pd.read_csv(config["feature_anno_path"], index_col=0)

    # gene id 列名在不同数据集里不一致（scAPAflow 输出用的是 'gene'，
    # 早期 ALL 数据用的是 'gene_id'），两个都找一下，统一改名成 'gene_id'
    # 方便后续 Layer0–3 里不用管这层差异。
    GENE_ID_COL_CANDIDATES = ["gene_id", "gene"]
    gene_col = next((c for c in GENE_ID_COL_CANDIDATES if c in feature_anno.columns), None)
    if gene_col is None:
        raise KeyError(f"feature_anno_path 指向的表里找不到基因ID列（尝试过 "
                        f"{GENE_ID_COL_CANDIDATES}），实际列名: {list(feature_anno.columns)}")
    if gene_col != "gene_id":
        print(f"  基因ID列名为 '{gene_col}'，统一重命名为 'gene_id'")
        feature_anno = feature_anno.rename(columns={gene_col: "gene_id"})

    # 校验 index 是否真的是 PAC id（应该和 spectra 的列名大量重合）；
    # 如果重合度很低，尝试用 'pac_id' 列重新设为 index 再试一次，
    # 而不是直接假设 index_col=0 读对了。
    overlap = len(spectra.columns.intersection(feature_anno.index))
    overlap_frac = overlap / len(spectra.columns)
    print(f"  当前 index 与 spectra PAC 的重合度: {overlap}/{len(spectra.columns)} ({overlap_frac:.0%})")
    if overlap_frac < 0.5 and "pac_id" in feature_anno.columns:
        print("  重合度过低，尝试改用 'pac_id' 列作为 index...")
        feature_anno = feature_anno.set_index("pac_id")
        overlap = len(spectra.columns.intersection(feature_anno.index))
        overlap_frac = overlap / len(spectra.columns)
        print(f"  改用 pac_id 后重合度: {overlap}/{len(spectra.columns)} ({overlap_frac:.0%})")
    if overlap_frac < 0.5:
        raise ValueError(
            f"feature_anno 的 index 与 spectra 的 PAC 列名重合度过低 ({overlap_frac:.0%})，"
            f"可能 feature_anno_path 指向的文件不对，或者需要指定别的列作为 index。\n"
            f"  spectra PAC 样例: {list(spectra.columns[:3])}\n"
            f"  feature_anno index 样例: {list(feature_anno.index[:3])}"
        )

    print(f"  feature_anno: {feature_anno.shape}")

    missing_pacs = spectra.columns.difference(feature_anno.index)
    if len(missing_pacs) > 0:
        print(f"  ⚠ spectra 中有 {len(missing_pacs)} 个 PAC 在 feature_anno 里找不到注释"
              f"（后续分析会自动跳过这些PAC）")

    # Optional wPSI is used by Layer0 QC and Layer3; symbols by Layer0/1.
    wpsi = load_wpsi(config)
    if wpsi is not None:
        wpsi = _align_wpsi_cells(wpsi, meta.index)
    gene_symbol_map = load_gene_symbol_map(config)


    print("\n✓ load_cnmf_outputs 完成")

    return {
        "spectra": spectra,
        "spectra_z": spectra_z,
        "usage": usage,
        "meta": meta,
        "feature_anno": feature_anno,
        "programs": programs,
        "wpsi": wpsi,
        "gene_symbol_map": gene_symbol_map,
    }


def load_wpsi(config: dict) -> pd.DataFrame | None:
    """Read an optional gene×cell wPSI RDS; report unavailable inputs explicitly."""
    path = config.get('wpsi_matrix_path')
    if not path:
        print('  [io_layer] 未配置 wpsi_matrix_path，跳过 wPSI；Layer3 将标注缺少输入')
        return None
    if not os.path.isfile(path):
        print(f'  [io_layer] wPSI 文件不存在: {path}，跳过 wPSI')
        return None
    import pyreadr
    try:
        matrix = pyreadr.read_r(path)[None]
    except (pyreadr.PyreadrError, pyreadr.LibrdataError, KeyError, ValueError) as error:
        print(f'  [io_layer] 无法解析 wPSI RDS: {type(error).__name__}: {error}')
        print('  请确认对象为带基因行名和细胞列名的 data.frame；可在 R 中另存 gzip RDS。')
        return None
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError('wPSI RDS must contain a gene-by-cell data.frame')
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError('wPSI gene and cell labels must be unique')
    print(f'  读取 wPSI: {matrix.shape} (genes × cells)')
    return matrix


def _align_wpsi_cells(wpsi: pd.DataFrame, meta_index: pd.Index) -> pd.DataFrame:
    """Keep matching cells, with an unambiguous 10x lane-suffix fallback."""
    direct = wpsi.columns.intersection(meta_index)
    if len(direct) >= 0.5 * wpsi.shape[1]:
        aligned = wpsi.loc[:, direct]
    else:
        stripped = meta_index.str.replace(r'-\d+$', '', regex=True)
        mapping = pd.Series(meta_index, index=stripped)
        if not mapping.index.is_unique:
            raise ValueError('Stripping metadata barcode suffixes creates ambiguous cell IDs')
        matched = wpsi.columns.intersection(mapping.index)
        aligned = wpsi.loc[:, matched].copy()
        aligned.columns = mapping.loc[matched].to_numpy()
    print(f'  wPSI 与 metadata 对齐: {aligned.shape[1]}/{wpsi.shape[1]} 个细胞')
    if aligned.shape[1] == 0:
        raise ValueError('wPSI and metadata have no matching cell barcodes')
    return aligned


def load_gene_symbol_map(config: dict) -> pd.Series | None:
    """按需读取 gene_id -> symbol 映射表（Layer1 富集分析用，Enrichr等
    功能富集工具通常需要基因symbol而不是ENSG这类ID）。

    期望的表格式：两列，第一列gene_id，第二列symbol（列名不强制，取前两列）。
    没配置路径或文件不存在时返回 None，调用方（Layer0）遇到 None 时直接
    用 gene_id 本身当作 symbol 使用，不阻塞流程——不是所有数据集的
    gene_id 都需要额外转换（有些movAPA注释里 gene_id 本身就是symbol）。
    """
    path = config.get("gene_symbol_map_path")
    if not path:
        print("  [io_layer] CONFIG 未配置 gene_symbol_map_path，Layer0 将保留 gene_id；Layer1 会报告无法映射的 ENSG")
        return None
    if not os.path.exists(path):
        print(f"  [io_layer] gene_symbol_map_path 指向的文件不存在: {path}，跳过")
        return None
    df = pd.read_csv(path)
    mapping = df.set_index(df.columns[0])[df.columns[1]]
    n_before = len(mapping)
    mapping = mapping[~mapping.index.duplicated(keep="first")]
    if len(mapping) < n_before:
        print(f"  ⚠ gene symbol 映射表中有 {n_before - len(mapping)} 个重复gene_id，"
              f"保留首次出现的映射")
    print(f"  读取 gene symbol 映射: {len(mapping)} 条")
    return mapping
