"""
Layer 3 — Marker 基因富集分析：程序间基因重叠 / 专属marker筛选 / 功能富集 /
program -> 功能模块 自动推断。
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
        if symbol_map is None:
            return gene_id
        if gene_id in symbol_map.index:
            return symbol_map.loc[gene_id]
        # ENSG带版本号(如 ENSG00000158481.13)时，映射表可能只有不带版本号的
        # 基础ID(ENSG00000158481)，去掉版本号后缀再试一次
        bare_id = gene_id.split(".")[0]
        if bare_id in symbol_map.index:
            return symbol_map.loc[bare_id]
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
        n_before = len(genes)
        genes = [g for g in genes if not g.upper().startswith("ENSG")]
        n_dropped = n_before - len(genes)
        if n_dropped > 0:
            if n_dropped == n_before:
                print(f"  [{label}] {n_before} 个基因全部仍是ENSG格式（一个都没成功映射到symbol），"
                      f"很可能是未配置gene_symbol_map_path或映射表覆盖不全，而不是这个program真的没有marker")
            else:
                print(f"  [{label}] 过滤掉 {n_dropped} 个仍是ENSG格式(未成功映射到symbol)的基因，"
                      f"剩余 {len(genes)} 个送去Enrichr查询")
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

    if "pac_id" not in source.columns:
        raise KeyError(f"depa_feature_source 缺少 'pac_id' 列，实际列名: {list(source.columns)}")

    # 优先找单值的 context_id 列；没有的话退回 contexts（复数）列——
    # 说明一个PAC可能同时来自多个显著context（对应n_selected_contexts>1），
    # 这种情况下把该行"展开"成多条(每个context各算一次)，而不是只取其中一个。
    if "context_id" in source.columns:
        context_col = "context_id"
        exploded = source[["pac_id", context_col]].rename(columns={context_col: "context"})
    elif "contexts" in source.columns:
        print("  [Layer3] depa_feature_source 没有单值的'context_id'列，使用'contexts'列"
              "（可能一个PAC对应多个context，将展开逐个计数）")
        sample_vals = source["contexts"].dropna().head(3).tolist()
        print(f"  contexts列原始样例: {sample_vals}")
        # 尝试用常见分隔符切分（逗号/分号/竖线），多数情况下能覆盖
        split_series = source["contexts"].astype(str).str.split(r"[;,|]\s*")
        exploded = source[["pac_id"]].copy()
        exploded["context"] = split_series
        exploded = exploded.explode("context")
        exploded["context"] = exploded["context"].str.strip()
    else:
        raise KeyError(f"depa_feature_source 既没有'context_id'也没有'contexts'列，"
                        f"实际列名: {list(source.columns)}")

    source_idx = exploded.set_index("pac_id")

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

        fracs = matched["context"].value_counts(normalize=True)
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


# ────────────────────────────────────────────────────────────────
# 把enrichment结果下钻到具体驱动基因，并和Layer1的leading genes对上
# ────────────────────────────────────────────────────────────────

def extract_enrichment_driver_genes(
    enrichment: dict,
    leading_genes: dict,
    data: dict,
    config: dict,
    top_n_terms: int = 2,
    label_suffix: str = "_markers",
) -> pd.DataFrame:
    """把run_enrichment()的结果下钻到"具体是哪些基因撑起了这个通路"，并和
    compute_leading_genes()的结果对上——判断这些驱动基因是否也是该program
    在Layer1里被判定为方向一致的leading gene，附上delta_from_bg作为独立
    证据（"这个通路不仅富集显著，驱动它的基因确实也在APA层面偏离了背景"）。

    Parameters
    ----------
    enrichment : dict
        run_enrichment() 的返回值，key形如 f"{program}{label_suffix}"
        （默认对应notebook里跑的"P01_markers"这批专属marker，不是"_all"
        那批——专属marker更适合做驱动基因溯源，"_all"里含大量非专属基因，
        容易让驱动基因列表虚高）。
    leading_genes : dict
        layer1_direction.compute_leading_genes() 的返回值。
    top_n_terms : int
        每个program只看padj最小的前几条term。
    label_suffix : str
        enrichment字典key的后缀，跟你跑enrichment时gene_lists的命名对应。

    Returns
    -------
    DataFrame，每行是一个(program, term, gene)组合，列：
        program, database, term, padj, gene_symbol, gene_id, in_leading_genes,
        delta_from_bg, gene_wpsi, spectra_z_max
        后四列在 in_leading_genes=False 时为NaN（该基因功能上富集显著，
        但不在Layer1的leading gene列表里，没有独立的APA偏离证据）。
        某个program没有enrichment结果或结果为空时直接跳过（不产生空行）。
    """
    programs = data["programs"]
    rows = []
    for p in programs:
        label = f"{p}{label_suffix}"
        df = enrichment.get(label)
        if df is None or len(df) == 0:
            print(f"  [{p}] enrichment结果为空或不存在(label={label!r})，跳过驱动基因提取")
            continue

        lg = leading_genes.get(p)
        lg_by_symbol = lg.set_index("symbol") if lg is not None and len(lg) > 0 else None

        top_terms = df.sort_values("padj").head(top_n_terms)
        n_in_before = len(rows)
        for _, term_row in top_terms.iterrows():
            gene_symbols = [g.strip() for g in str(term_row["genes"]).split(";") if g.strip()]
            for sym in gene_symbols:
                if lg_by_symbol is not None and sym in lg_by_symbol.index:
                    lg_row = lg_by_symbol.loc[sym]
                    if isinstance(lg_row, pd.DataFrame):  # 理论上不该重复，防御一下
                        lg_row = lg_row.iloc[0]
                    rows.append({
                        "program": p, "database": term_row["database"], "term": term_row["term"],
                        "padj": term_row["padj"], "gene_symbol": sym,
                        "gene_id": lg_row["gene_id"], "in_leading_genes": True,
                        "delta_from_bg": lg_row["delta_from_bg"],
                        "gene_wpsi": lg_row["gene_wpsi"],
                        "spectra_z_max": lg_row["spectra_z_max"],
                    })
                else:
                    rows.append({
                        "program": p, "database": term_row["database"], "term": term_row["term"],
                        "padj": term_row["padj"], "gene_symbol": sym,
                        "gene_id": np.nan, "in_leading_genes": False,
                        "delta_from_bg": np.nan, "gene_wpsi": np.nan, "spectra_z_max": np.nan,
                    })

        added = rows[n_in_before:]
        n_in = sum(r["in_leading_genes"] for r in added)
        print(f"  [{p}] top{top_n_terms}条term共{len(added)}个驱动基因(去重前)，"
              f"其中{n_in}个同时是leading gene")

    if not rows:
        print("  没有任何program产出驱动基因，返回空表")
        return pd.DataFrame(columns=["program", "database", "term", "padj", "gene_symbol",
                                      "gene_id", "in_leading_genes", "delta_from_bg",
                                      "gene_wpsi", "spectra_z_max"])

    return pd.DataFrame(rows)