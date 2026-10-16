cat > depa_downstream/io_layer.py << 'PYEOF_MARKER_io_layer_py'
"""
读取 cNMF 输出、metadata、feature 注释，统一封装成一个 data 字典。

所有下游 Layer 函数都只接收这个 data 字典（+ config），不直接碰文件路径，
这样 Layer 函数和"数据从哪读"彻底解耦。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import zscore

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
        "spectra_tpm"   : DataFrame, programs x PACs (原始 spectra_tpm)
        "spectra_z"     : DataFrame, programs x PACs (对每个 program 行做 Z-score，
                           即在该 program 内部比较 PAC 之间的相对权重)
        "usage"         : DataFrame, cells x programs
        "meta"          : DataFrame, cells x metadata 列（index=cell barcode），
                           从 h5ad 的 .obs 读取
        "feature_anno"  : DataFrame, index=PAC id，至少含 gene_id 列
        "programs"      : list[str]，program 名称（如 ["P01", ..., "P06"]）
        "wpsi"          : DataFrame or None，gene x cell
        "psi"           : DataFrame or None，gene x cell
        "depa_feature_source" : DataFrame or None
    """
    validate_config(config)
    paths = _build_cnmf_paths(config)
    _check_files_exist(paths)

    # ── 1. spectra / usage ──────────────────────────────────────
    print("\n读取 cNMF spectra 与 usage...")
    spectra = pd.read_csv(paths["spectra_score"], sep="\t", index_col=0)
    spectra_tpm = pd.read_csv(paths["spectra_tpm"], sep="\t", index_col=0)
    usage = pd.read_csv(paths["usage"], sep="\t", index_col=0)

    # program 命名统一化：不同数据集的 cNMF 输出表头格式不一致
    # （有的是 "P01".."P18" 字符串，有的是纯数字 1..6），统一转成
    # 零填充字符串（如 K=6 -> "P01".."P06"，K=120 -> "P001".."P120"），
    # 保证下游所有函数（尤其是手动指定的 module_def/contrasts 里用到的
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
    spectra_tpm.index = [_normalize_program_label(p) for p in spectra_tpm.index]
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
    adata = ad.read_h5ad(config["h5ad_path"])
    meta = adata.obs.copy()
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
    # 方便后续 Layer1/3 里不用管这层差异。
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

    # ── 4. 可选输入：wPSI / PSI / DEPA_feature_source / gene symbol映射 ─
    wpsi_psi = load_wpsi_psi(config)
    depa_source = load_depa_feature_source(config)
    gene_symbol_map = load_gene_symbol_map(config)

    print("\n✓ load_cnmf_outputs 完成")

    return {
        "spectra": spectra,
        "spectra_tpm": spectra_tpm,
        "spectra_z": spectra_z,
        "usage": usage,
        "meta": meta,
        "feature_anno": feature_anno,
        "programs": programs,
        "wpsi": wpsi_psi["wpsi"],
        "psi": wpsi_psi["psi"],
        "depa_feature_source": depa_source,
        "gene_symbol_map": gene_symbol_map,
    }


def load_wpsi_psi(config: dict) -> dict:
    """按需读取 wPSI / PSI 矩阵（.rds，用 pyreadr）。

    没有配置对应路径、文件不存在、或者读取失败（.rds 格式 pyreadr 解析不了，
    比如较新 R 版本默认用了 pyreadr 不支持的压缩方式）时对应 key 都设为 None，
    并打印清楚原因——这是可选输入，读取失败不应该让整个 load_cnmf_outputs()
    崩溃，调用方（Layer1）会根据 None 自动跳过 wPSI 相关分析。
    """
    result = {"wpsi": None, "psi": None}
    for key, cfg_key in [("wpsi", "wpsi_matrix_path"), ("psi", "psi_matrix_path")]:
        path = config.get(cfg_key)
        if not path:
            print(f"  [io_layer] CONFIG 未配置 {cfg_key}，跳过读取 {key}")
            continue
        if not os.path.exists(path):
            print(f"  [io_layer] {cfg_key} 指向的文件不存在: {path}，跳过读取 {key}")
            continue
        import pyreadr
        print(f"  读取 {key} 矩阵: {path}")
        try:
            df = pyreadr.read_r(path)[None]
        except Exception as e:
            print(f"  ⚠ 读取 {key} 失败（pyreadr无法解析该.rds文件），跳过，"
                  f"不影响其余数据: {type(e).__name__}: {e}")
            print(f"    常见原因：R端 saveRDS() 用了pyreadr不支持的压缩格式(如xz)，"
                  f"或对象不是普通data.frame。\n"
                  f"    解决办法二选一：\n"
                  f"      1) 在R里重存一份: saveRDS(obj, '{path}', compress='gzip')\n"
                  f"      2) 改用 rpy2 读取（需要本机有R环境）：\n"
                  f"         pip install rpy2\n"
                  f"         import rpy2.robjects as robjects\n"
                  f"         from rpy2.robjects import pandas2ri; pandas2ri.activate()\n"
                  f"         df = robjects.r['readRDS']('{path}')")
            continue
        print(f"    {key} shape: {df.shape}")
        result[key] = df
    return result


def load_gene_symbol_map(config: dict) -> pd.Series | None:
    """按需读取 gene_id -> symbol 映射表（Layer3 富集分析用，Enrichr等
    功能富集工具通常需要基因symbol而不是ENSG这类ID）。

    期望的表格式：两列，第一列gene_id，第二列symbol（列名不强制，取前两列）。
    没配置路径或文件不存在时返回 None，调用方（Layer3）遇到 None 时直接
    用 gene_id 本身当作 symbol 使用，不阻塞流程——不是所有数据集的
    gene_id 都需要额外转换（有些movAPA注释里 gene_id 本身就是symbol）。
    """
    path = config.get("gene_symbol_map_path")
    if not path:
        print("  [io_layer] CONFIG 未配置 gene_symbol_map_path，Layer3将直接用gene_id当symbol使用")
        return None
    if not os.path.exists(path):
        print(f"  [io_layer] gene_symbol_map_path 指向的文件不存在: {path}，跳过")
        return None
    df = pd.read_csv(path)
    mapping = df.set_index(df.columns[0])[df.columns[1]]
    print(f"  读取 gene symbol 映射: {len(mapping)} 条")
    return mapping


def load_depa_feature_source(config: dict) -> pd.DataFrame | None:
    """按需读取 DEPA_feature_source.csv（Layer3 的 program->context 反查用）。

    这是可选输入：不是所有数据集在构建 cNMF 输入时都经过
    scAPAflow::build_depa_feature_object 这一步（比如最早的 ALL 白血病数据
    就没有），所以缺失时返回 None，不报错。
    """
    path = config.get("depa_feature_source_path")
    if not path:
        print("  [io_layer] CONFIG 未配置 depa_feature_source_path，跳过（Layer3将退回手动module_def）")
        return None
    if not os.path.exists(path):
        print(f"  [io_layer] depa_feature_source_path 指向的文件不存在: {path}，跳过")
        return None
    df = pd.read_csv(path)
    print(f"  读取 DEPA_feature_source: {df.shape}")
    return df
PYEOF_MARKER_io_layer_py

cat > depa_downstream/layer3_markers.py << 'PYEOF_MARKER_layer3_markers_py'
"""
Layer 3 — Marker 基因富集分析：程序间基因重叠 / 专属marker筛选 / 功能富集 /
program -> 功能模块 自动推断。

对应原 notebook 的 Cell 4-7。其中 infer_program_modules() 是新增的
（原来 MODULE_DEF 是手写死表），用 DEPA_feature_source.csv 反查每个 program
的 top PAC 主要来自哪个 context，自动给出模块标签初稿。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

ENRICHR_ADD_LIST_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
DEFAULT_ENRICHR_DATABASES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "MSigDB_Hallmark_2020",
    "Reactome_2022",
]


# ────────────────────────────────────────────────────────────────
# 共享底层：program x gene 的连续Z矩阵（把PAC级Z-score聚合到基因级）
# ────────────────────────────────────────────────────────────────

def _build_gene_program_zmatrix(data: dict, config: dict) -> pd.DataFrame:
    """把 spectra_z（programs x PACs）聚合成 programs x genes。

    同一个基因如果有多个PAC，取该基因在这个program下所有PAC里的最大Z值
    （即"只要有一个PAC在这个program里权重很高，这个基因就算进这个program"）。
    """
    spectra_z = data["spectra_z"]
    feature_anno = data["feature_anno"]
    pac_to_gene = feature_anno["gene_id"]

    common_pacs = spectra_z.columns.intersection(pac_to_gene.index)
    sub = spectra_z[common_pacs].T
    sub["gene_id"] = pac_to_gene.loc[common_pacs].values
    zmat = sub.groupby("gene_id").max().T
    return zmat


# ────────────────────────────────────────────────────────────────
# Step 1: program间基因重叠分析
# ────────────────────────────────────────────────────────────────

def compute_program_gene_sets(data: dict, config: dict) -> dict:
    """为每个 program 构建核心基因集（Z > z_threshold）。

    Parameters
    ----------
    data : dict
        io_layer.load_cnmf_outputs() 的返回值。
    config : dict
        用到 "z_threshold"（默认1.5，和Layer1共用同一个默认值，
        含义是"该基因在该program里至少有一个PAC的Z-score超过这个阈值"）。

    Returns
    -------
    dict:
        "gene_sets": {program: set(gene_id)}  -- 核心基因集
        "zmat": DataFrame (programs x genes)  -- 完整的连续Z值矩阵，
                 供 select_program_markers() 判断"专属性"时使用，
                 不需要重新计算。
    """
    zmat = _build_gene_program_zmatrix(data, config)
    z_threshold = config.get("z_threshold", 1.5)
    programs = data["programs"]

    gene_sets = {}
    for p in programs:
        genes = set(zmat.columns[zmat.loc[p] > z_threshold])
        gene_sets[p] = genes
        print(f"  [{p}] 核心基因集大小: {len(genes)}")

    return {"gene_sets": gene_sets, "zmat": zmat}


def compute_jaccard_matrix(gene_sets: dict, programs: list) -> pd.DataFrame:
    """计算 program 两两之间基因集的 Jaccard 相似度矩阵。

    Parameters
    ----------
    gene_sets : dict
        compute_program_gene_sets() 返回值里的 "gene_sets"。
    programs : list
        program 顺序。

    Returns
    -------
    DataFrame，index/columns 都是 programs，对角线为1。
    """
    n = len(programs)
    mat = pd.DataFrame(np.eye(n), index=programs, columns=programs)
    for i, p1 in enumerate(programs):
        for j, p2 in enumerate(programs):
            if j <= i:
                continue
            a, b = gene_sets[p1], gene_sets[p2]
            union = len(a | b)
            jac = len(a & b) / union if union > 0 else 0.0
            mat.loc[p1, p2] = jac
            mat.loc[p2, p1] = jac
    return mat


# ────────────────────────────────────────────────────────────────
# Step 2: 跨program专属marker基因筛选
# ────────────────────────────────────────────────────────────────

def select_program_markers(
    program_gene_sets: dict,
    data: dict,
    config: dict,
    top_n: int = 20,
    spec_min_z: float = 0.5,
) -> dict:
    """为每个 program 筛选"专属" marker 基因。

    专属的定义：该基因在本program的Z值 > z_threshold（已经在gene_sets里），
    且在其余所有program里的Z值都 < spec_min_z（也就是别的program对这个
    基因几乎没有权重）。满足条件的基因按本program的Z值从高到低排序，
    取前 top_n 个。

    Parameters
    ----------
    program_gene_sets : dict
        compute_program_gene_sets() 的返回值（需要用到其中的 "zmat"）。
    data : dict
        用到 data["programs"]，以及可选的 data["gene_symbol_map"]
        （没有就直接用gene_id当symbol显示）。
    config : dict
        用到 "z_threshold"（默认1.5，仅用于日志展示，实际候选集合已经
        由 compute_program_gene_sets 用这个阈值筛过）。
    top_n : int
        每个program最多保留多少个marker。
    spec_min_z : float
        "专属性"判定阈值：其他所有program里该基因的最大Z值必须低于这个值。

    Returns
    -------
    dict: {program: DataFrame(columns=[gene_id, symbol, focal_z, other_max_z])}
        某个program如果一个专属marker都筛不出来，对应DataFrame为空表（不报错）。
    """
    zmat = program_gene_sets["zmat"]
    gene_sets = program_gene_sets["gene_sets"]
    programs = data["programs"]
    symbol_map = data.get("gene_symbol_map")

    def _symbol(gene_id):
        if symbol_map is not None and gene_id in symbol_map.index:
            return symbol_map.loc[gene_id]
        return gene_id

    result = {}
    for p in programs:
        candidates = gene_sets[p]
        if len(candidates) == 0:
            print(f"  [{p}] 核心基因集为空，跳过marker筛选")
            result[p] = pd.DataFrame(columns=["gene_id", "symbol", "focal_z", "other_max_z"])
            continue

        other_programs = [q for q in programs if q != p]
        focal_z = zmat.loc[p, list(candidates)]
        other_max_z = zmat.loc[other_programs, list(candidates)].max(axis=0)

        exclusive_mask = other_max_z < spec_min_z
        exclusive_genes = focal_z.index[exclusive_mask]

        marker_df = pd.DataFrame({
            "gene_id": exclusive_genes,
            "focal_z": focal_z.loc[exclusive_genes].values,
            "other_max_z": other_max_z.loc[exclusive_genes].values,
        }).sort_values("focal_z", ascending=False).head(top_n).reset_index(drop=True)
        marker_df["symbol"] = marker_df["gene_id"].apply(_symbol)
        marker_df = marker_df[["gene_id", "symbol", "focal_z", "other_max_z"]]

        print(f"  [{p}] 候选基因{len(candidates)}个 -> 专属marker{len(marker_df)}个 "
              f"(spec_min_z={spec_min_z})")
        result[p] = marker_df

    return result


# ────────────────────────────────────────────────────────────────
# Step 3: 功能富集（Enrichr）
# ────────────────────────────────────────────────────────────────

def _query_enrichr(gene_symbols: list, databases: list, description: str = "gene_list") -> pd.DataFrame:
    """调用 Enrichr API，对一份基因列表在多个数据库里做富集，合并返回。"""
    resp = requests.post(
        ENRICHR_ADD_LIST_URL,
        files={"list": (None, "\n".join(gene_symbols)), "description": (None, description)},
        timeout=30,
    )
    resp.raise_for_status()
    user_list_id = resp.json()["userListId"]

    rows = []
    for db in databases:
        r = requests.get(ENRICHR_ENRICH_URL, params={"userListId": user_list_id, "backgroundType": db}, timeout=30)
        r.raise_for_status()
        payload = r.json().get(db, [])
        for entry in payload:
            # Enrichr 返回格式: [rank, term, pvalue, zscore, combined_score,
            #                    overlapping_genes, adjusted_pvalue, old_pvalue, old_adjusted_pvalue]
            rows.append({
                "database": db,
                "rank": entry[0],
                "term": entry[1],
                "pvalue": entry[2],
                "zscore": entry[3],
                "combined_score": entry[4],
                "genes": ";".join(entry[5]) if len(entry) > 5 else "",
                "padj": entry[6] if len(entry) > 6 else np.nan,
            })
        time.sleep(0.2)  # 避免请求过快被限流

    return pd.DataFrame(rows)


def run_enrichment(
    gene_lists: dict,
    config: dict,
    databases: list | None = None,
    min_genes: int = 3,
) -> dict:
    """对若干份命名的基因列表分别做 Enrichr 功能富集。

    不区分传入的基因列表是"某个program的全部显著基因"还是"专属marker基因"
    ——由调用方（notebook）决定要传什么进来、用什么样的key命名，本函数
    只负责对每份列表跑一遍富集。

    Parameters
    ----------
    gene_lists : dict
        {label: [symbol, ...]}，label自定，比如 "P03_all" / "P03_markers"。
    config : dict
        用到 config.get("enrichr_databases", DEFAULT_ENRICHR_DATABASES)。
    databases : list, optional
        显式传入时覆盖 config 里的设置。
    min_genes : int
        基因数少于这个值的列表直接跳过（Enrichr对太短的列表结果没有意义）。

    Returns
    -------
    dict: {label: DataFrame or None}
        DataFrame 列: database, rank, term, pvalue, zscore, combined_score,
        genes, padj，已按padj排序。
        某个label请求失败（网络问题等）或基因数不够，值为 None，不中断其他label。
    """
    dbs = databases or config.get("enrichr_databases", DEFAULT_ENRICHR_DATABASES)
    results = {}
    for label, genes in gene_lists.items():
        genes = [g for g in genes if isinstance(g, str) and g.strip()]
        if len(genes) < min_genes:
            print(f"  [{label}] 基因数 {len(genes)} < min_genes={min_genes}，跳过")
            results[label] = None
            continue
        try:
            df = _query_enrichr(genes, dbs, description=label)
        except Exception as e:
            print(f"  ⚠ [{label}] Enrichr查询失败: {e}")
            results[label] = None
            continue
        if len(df) == 0:
            print(f"  [{label}] 富集结果为空")
            results[label] = df
            continue
        df = df.sort_values("padj").reset_index(drop=True)
        top_term = df.iloc[0]["term"]
        print(f"  [{label}] {len(genes)}个基因 -> {len(df)}条富集term，最强: {top_term} (padj={df.iloc[0]['padj']:.2e})")
        results[label] = df
    return results


# ────────────────────────────────────────────────────────────────
# program -> 功能模块 自动推断 / 手动兜底
# ────────────────────────────────────────────────────────────────

def infer_program_modules(
    data: dict,
    config: dict,
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
        top_n_context 个，格式为 [(context, fraction), ...]），
        dominant_context（占比最高的一个），
        dominant_fraction（占比数值，供人工判断这个自动标签靠不靠谱——
        比如占比只有 20% 说明这个 program 是多个 context 混合，
        不该简单打单一标签）
    """
    source = data.get("depa_feature_source")
    if source is None:
        print("  [Layer3] 未提供 DEPA_feature_source.csv，跳过模块自动推断，"
              "请在 config['module_def'] 里手动指定")
        return None

    spectra_z = data["spectra_z"]
    programs = data["programs"]
    z_threshold = config.get("z_threshold", 1.5)

    if "pac_id" not in source.columns or "context_id" not in source.columns:
        raise KeyError(f"depa_feature_source 缺少 'pac_id' 或 'context_id' 列，"
                        f"实际列名: {list(source.columns)}")
    source_idx = source.set_index("pac_id")

    records = []
    for p in programs:
        z_row = spectra_z.loc[p]
        top_pacs = z_row[z_row > z_threshold].index
        matched = source_idx.loc[source_idx.index.intersection(top_pacs)]

        if len(matched) == 0:
            print(f"  [{p}] 没有任何top PAC能在DEPA_feature_source里找到匹配，跳过")
            records.append({"program": p, "dominant_context": None,
                             "dominant_fraction": np.nan, "top_contexts": []})
            continue

        fracs = matched["context_id"].value_counts(normalize=True)
        top_contexts = list(zip(fracs.head(top_n_context).index, fracs.head(top_n_context).round(3).values))
        print(f"  [{p}] top_contexts: {top_contexts}")
        records.append({
            "program": p,
            "dominant_context": fracs.index[0],
            "dominant_fraction": round(float(fracs.iloc[0]), 3),
            "top_contexts": top_contexts,
        })

    return pd.DataFrame(records).set_index("program")


def resolve_module_labels(data: dict, config: dict) -> pd.DataFrame:
    """统一入口：优先用 infer_program_modules() 自动推断，没有数据则退回
    config['module_def']（手动查表 program -> module 名），两者都没有则每个
    program 各自成一个模块（不合并，保证下游画图不报错）。

    Returns
    -------
    DataFrame，index=program，至少含 "module" 列（用于画图分组着色），
    若走的是自动推断路径，还会带上 infer_program_modules() 的其余列。
    """
    auto = infer_program_modules(data, config)
    if auto is not None:
        auto = auto.copy()
        auto["module"] = auto["dominant_context"].fillna("unassigned")
        return auto

    manual = config.get("module_def")
    programs = data["programs"]
    if manual is not None:
        prog_to_module = {}
        for module_name, prog_list in manual.items():
            for p in prog_list:
                prog_to_module[p] = module_name
        modules = [prog_to_module.get(p, "unassigned") for p in programs]
        print("  [Layer3] 使用手动 module_def 分配模块")
        return pd.DataFrame({"module": modules}, index=programs)

    print("  [Layer3] 未提供自动推断数据也未提供 module_def，每个 program 独立成模块")
    return pd.DataFrame({"module": programs}, index=programs)
PYEOF_MARKER_layer3_markers_py

cat > depa_downstream/viz.py << 'PYEOF_MARKER_viz_py'
"""
纯画图函数。原则：只接收已经算好的结果（DataFrame/dict），不在这里做任何
统计计算——计算逻辑都应该在 layer1/2/3 里，方便单独测试和复用。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def plot_jaccard_heatmap(jaccard_mat: pd.DataFrame, out_dir: str, module_labels: pd.DataFrame | None = None) -> str:
    """画 program 间基因集 Jaccard 相似度热图，可选按模块分组排序。

    Parameters
    ----------
    jaccard_mat : DataFrame
        layer3_markers.compute_jaccard_matrix() 的返回值。
    out_dir : str
        保存目录。
    module_labels : DataFrame, optional
        layer3_markers.resolve_module_labels() 的返回值（需要含"module"列），
        提供时会按模块把program重新排序聚在一起，方便看模块内/模块间的重叠模式。
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    programs = list(jaccard_mat.index)
    if module_labels is not None and "module" in module_labels.columns:
        order = module_labels.loc[programs].sort_values("module").index.tolist()
    else:
        order = sorted(programs)
    mat = jaccard_mat.loc[order, order]

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(order)), max(4, 0.6 * len(order))))
    off_diag = mat.values[~np.eye(len(order), dtype=bool)]
    vmax = max(0.05, off_diag.max() if len(off_diag) else 0.05)
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val < mat.values.max() * 0.6 else "black", fontsize=8)
    ax.set_title("Program gene-set Jaccard similarity")
    fig.colorbar(im, ax=ax, label="Jaccard index", fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "layer3_jaccard_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_enrichment_bubble(enrichment_by_label: dict, out_dir: str, top_n: int = 10, padj_threshold: float = 0.05) -> str:
    """富集结果气泡图：每个label取padj最小的前top_n条term，跨label并排展示。

    气泡大小 = -log10(padj)，气泡颜色 = combined_score，横轴是label，
    纵轴是term（按label分组，组内按显著性排序）。

    Parameters
    ----------
    enrichment_by_label : dict
        layer3_markers.run_enrichment() 的返回值（{label: DataFrame or None}）。
    out_dir : str
        保存目录。
    top_n : int
        每个label最多展示多少条term。
    padj_threshold : float
        只展示padj低于此阈值的term；某个label如果一条都不满足，会被跳过
        （不在图上占位），并打印提示。
    """
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in enrichment_by_label.items() if v is not None and len(v) > 0}
    if not valid:
        print("  [Layer3] 没有任何富集结果，跳过画图")
        return ""

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for label, df in valid.items():
        sig = df[df["padj"] < padj_threshold].sort_values("padj").head(top_n)
        if len(sig) == 0:
            print(f"  [{label}] 没有padj<{padj_threshold}的term，跳过展示")
            continue
        for _, r in sig.iterrows():
            rows.append({"label": label, "term": r["term"], "padj": r["padj"],
                         "combined_score": r["combined_score"]})
    if not rows:
        print(f"  [Layer3] 所有label都没有padj<{padj_threshold}的term，跳过画图")
        return ""

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values(["label", "padj"])
    plot_df["term_short"] = plot_df["term"].str.slice(0, 45)
    term_order = list(dict.fromkeys(plot_df["term_short"].tolist()))
    labels = list(dict.fromkeys(plot_df["label"]))

    plot_df["neglog10_padj"] = -np.log10(plot_df["padj"].clip(lower=1e-300))
    y_pos = {t: i for i, t in enumerate(term_order)}
    x_pos = {l: i for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), max(3, 0.35 * len(term_order))))
    sc = ax.scatter(
        [x_pos[l] for l in plot_df["label"]],
        [y_pos[t] for t in plot_df["term_short"]],
        s=plot_df["neglog10_padj"] * 30,
        c=plot_df["combined_score"],
        cmap="Reds",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=8)
    ax.set_title(f"Top enriched terms (padj < {padj_threshold})")
    fig.colorbar(sc, ax=ax, label="combined score", fraction=0.04, pad=0.08)

    for size_val, label_txt in [(2, "padj=1e-2"), (5, "padj=1e-5")]:
        ax.scatter([], [], s=size_val * 30, c="gray", edgecolors="black", label=label_txt)
    ax.legend(loc="upper left", bbox_to_anchor=(1.35, 1), fontsize=8, title="bubble size")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "layer3_enrichment_bubble.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path
PYEOF_MARKER_viz_py

cat > depa_downstream/__init__.py << 'PYEOF_MARKER___init___py'
from .config_schema import validate_config, ConfigError
from .io_layer import load_cnmf_outputs, load_wpsi_psi, load_depa_feature_source, load_gene_symbol_map
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
from .viz import plot_jaccard_heatmap, plot_enrichment_bubble

__all__ = [
    "validate_config", "ConfigError",
    "load_cnmf_outputs", "load_wpsi_psi", "load_depa_feature_source", "load_gene_symbol_map",
    "compute_program_direction", "plot_program_direction_summary",
    "compute_contrast_association", "plot_association_heatmap", "plot_program_umap_grid",
    "compute_program_gene_sets", "compute_jaccard_matrix", "select_program_markers",
    "run_enrichment", "infer_program_modules", "resolve_module_labels",
    "plot_jaccard_heatmap", "plot_enrichment_bubble",
]
PYEOF_MARKER___init___py

echo "4个文件已更新: io_layer.py / layer3_markers.py / viz.py / __init__.py"