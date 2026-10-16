"""
Layer 1 — 程序特征基础：基于 wPSI 的 APA 方向验证（MWU版）+ 基因/细胞层面下探。

对应原 notebook 的 Cell 2，并在此基础上扩展了"program是被哪些基因驱动的
(leading genes)"和"这些基因的差异是否独立于细胞组成之外(组内对照)"两层
下探分析。核心逻辑不依赖数据集语义（不管 program 是按疾病分组还是按细胞
身份/状态分组跑出来的，方向检验的算法是一样的），所以这层几乎不需要因为
数据集不同而改函数本身，只是入参（wpsi/spectra_z）不同。

方向惯例（重要，如果和你的数据不一致需要翻转）：
    wPSI 值越大 -> 越偏向 Distal（远端PA位点使用比例越高）
    wPSI 值越小 -> 越偏向 Proximal

背景基因池的两种模式（config["background_scope"]，默认 "pac_universe"）：
    "pac_universe" — 背景 = 本次cNMF分析涉及的全部基因（即 feature_anno 里
                      所有PAC对应的基因），减去当前program的前景基因。
                      这样背景本身就是"和APA/细胞身份状态相关"的基因集合，
                      前景vs背景比较的是"这个program相对于同类基因有没有
                      方向偏向"，区分力更强，是默认推荐模式。
    "genome_wide"  —  背景 = wPSI矩阵里的全部基因（movAPA的全基因组输出）减
                      去前景。背景池巨大且跨program几乎不变（约等于全局中位
                      数），统计功效通常很低，仅作为对比参照保留。

wPSI在单细胞层面高度零膨胀（大多数细胞的wPSI恰好卡在0或1边界，因为该
细胞该基因只观测到极少reads），跨细胞取中位数会被这类噪声钉死在边界值，
所以本模块统一用 _compute_detection_filtered_gene_wpsi() 先过滤掉低检出率
基因，再计算中位数——compute_program_direction() 和 compute_leading_genes()
都调用这一个共享函数，保证两处用的是同一套过滤后的基因池，口径统一。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, spearmanr, binomtest

# ────────────────────────────────────────────────────────────────
# 共享底层：检出率过滤后的基因级wPSI
# ────────────────────────────────────────────────────────────────

def _compute_detection_filtered_gene_wpsi(wpsi: pd.DataFrame, min_detection_rate: float = 0.1) -> pd.Series:
    """跨细胞聚合出基因级wPSI（中位数），并排除检出率过低的基因。

    背景：单细胞层面wPSI在低表达基因上高度零膨胀（大多数细胞wPSI恰好卡在
    0或1边界，因为该细胞该基因只被观测到极少reads），这类基因的跨细胞
    中位数会被噪声钉死在边界值，不能反映真实的APA使用比例，不该参与
    "前景/背景基因池"或"leading gene"的判断。

    detection_rate定义：wPSI落在开区间(0,1)内(即该细胞里真正观测到了
    proximal和distal信号共存)的细胞数 / 该基因非NaN细胞总数。

    compute_program_direction() 和 compute_leading_genes() 都调用这个函数，
    保证两处用的是同一套过滤后的基因池，口径统一（否则方向检验用了未过滤
    的基因、leading genes却用了过滤后的，两边对不上）。

    Parameters
    ----------
    min_detection_rate : float
        检出率低于此阈值的基因会被排除。默认0.1（即至少10%的细胞要同时
        观测到proximal和distal信号），可通过config["min_detection_rate"]调整。

    Returns
    -------
    Series，index=gene_id（已排除低检出基因），值为跨细胞中位数wPSI。
    """
    non_edge = ((wpsi > 0) & (wpsi < 1)).sum(axis=1)
    n_valid = wpsi.notna().sum(axis=1)
    detection_rate = non_edge / n_valid.replace(0, np.nan)
    keep = detection_rate[detection_rate >= min_detection_rate].index
    n_dropped = wpsi.shape[0] - len(keep)
    print(f"  基因检出率过滤(min_detection_rate={min_detection_rate}): "
          f"排除 {n_dropped}/{wpsi.shape[0]} 个基因(wPSI在多数细胞中卡在0/1边界，中位数不可靠)")
    gene_wpsi = wpsi.loc[keep].median(axis=1, skipna=True).dropna()
    print(f"  过滤后有效基因数: {len(gene_wpsi)}")
    return gene_wpsi


# ────────────────────────────────────────────────────────────────
# Program 层面：方向检验 + leading genes
# ────────────────────────────────────────────────────────────────

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
        用到 config 里的：
        "z_threshold"（默认 1.5，前景PAC筛选阈值）
        "background_scope"（默认 "pac_universe"，见模块docstring）
        "min_detection_rate"（默认 0.1，见 _compute_detection_filtered_gene_wpsi）
    min_fg_genes : int
        前景基因数少于这个值的 program 跳过，不做检验（样本太小检验没意义）。

    Returns
    -------
    DataFrame，index=program，列包括：
        n_fg_genes, n_bg_genes, fg_median, bg_median, delta_median,
        mwu_stat, pval, padj, direction ("Proximal" / "Distal" / "NA")
    若 data["wpsi"] 为 None（该数据集没有配置 wPSI 矩阵），打印提示并返回 None，
    调用方需要对 None 做判断，不能默认它一定存在。

    注意：这是一个基于全体细胞池化中位数的粗筛统计量，对局限在特定细胞
    亚群内的信号不敏感（信号会被占多数的背景细胞"稀释"到接近0）。它的
    作用是从全部program里快速圈定候选方向，不能单独作为最终结论——真正
    站得住的证据需要靠 compute_leading_genes() + summarize_leading_gene_
    within_celltype() 做基因层面、细胞类型内部的精细化验证。
    """
    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过 APA 方向检验（compute_program_direction 返回 None）")
        return None

    spectra_z = data["spectra_z"]
    feature_anno = data["feature_anno"]
    programs = data["programs"]
    z_threshold = config.get("z_threshold", 1.5)
    background_scope = config.get("background_scope", "pac_universe")
    if background_scope not in ("pac_universe", "genome_wide"):
        raise ValueError(f"config['background_scope'] 必须是 'pac_universe' 或 'genome_wide'，"
                          f"收到: {background_scope!r}")

    # ── 1. 基因级 wPSI（跨细胞取中位数，并过滤低检出率基因）────────
    print(f"计算基因级 wPSI（{wpsi.shape[0]} 个基因 x {wpsi.shape[1]} 个细胞）...")
    min_detection_rate = config.get("min_detection_rate", 0.1)
    gene_wpsi = _compute_detection_filtered_gene_wpsi(wpsi, min_detection_rate)

    # PAC -> gene_id 映射
    pac_to_gene = feature_anno["gene_id"]

    # background_scope="pac_universe" 时的候选背景基因池：
    # 本次cNMF分析实际涉及的PAC（spectra_z的列，而不是feature_anno的全部行——
    # feature_anno常常是筛选前的更大注释表）对应的基因（去重），再和 gene_wpsi 取交集
    analysis_pacs = spectra_z.columns.intersection(pac_to_gene.index)
    pac_universe_genes = pd.Index(pac_to_gene.loc[analysis_pacs].dropna().unique()).intersection(gene_wpsi.index)
    print(f"  cNMF分析涉及的基因全集（pac_universe，基于{len(analysis_pacs)}个PAC）大小: {len(pac_universe_genes)}")
    print(f"  background_scope = {background_scope!r}")

    records = []
    for prog in programs:
        z_row = spectra_z.loc[prog]
        top_pacs = z_row[z_row > z_threshold].index
        top_pacs = top_pacs.intersection(pac_to_gene.index)
        fg_genes = pac_to_gene.loc[top_pacs].dropna().unique()
        fg_genes = pd.Index(fg_genes).intersection(gene_wpsi.index)

        if len(fg_genes) < min_fg_genes:
            print(f"  [{prog}] 前景基因数 {len(fg_genes)} < min_fg_genes={min_fg_genes}，跳过")
            records.append({
                "program": prog, "n_fg_genes": len(fg_genes), "n_bg_genes": np.nan,
                "fg_median": np.nan, "bg_median": np.nan, "delta_median": np.nan,
                "mwu_stat": np.nan, "pval": np.nan, "direction": "NA",
            })
            continue

        if background_scope == "pac_universe":
            bg_genes = pac_universe_genes.difference(fg_genes)
        else:  # genome_wide
            bg_genes = gene_wpsi.index.difference(fg_genes)

        fg_vals = gene_wpsi.loc[fg_genes].values
        bg_vals = gene_wpsi.loc[bg_genes].values

        stat, pval = mannwhitneyu(fg_vals, bg_vals, alternative="two-sided")
        fg_median = float(np.median(fg_vals))
        bg_median = float(np.median(bg_vals))
        delta = fg_median - bg_median
        if delta > 0:
            direction = "Distal"
        elif delta < 0:
            direction = "Proximal"
        else:
            direction = "Unchanged"

        print(f"  [{prog}] 前景基因={len(fg_genes)}, 背景基因={len(bg_genes)}, "
              f"delta_median={delta:+.4f}, pval={pval:.2e} -> {direction}")

        records.append({
            "program": prog, "n_fg_genes": len(fg_genes), "n_bg_genes": len(bg_genes),
            "fg_median": fg_median, "bg_median": bg_median, "delta_median": delta,
            "mwu_stat": float(stat), "pval": float(pval), "direction": direction,
        })

    result = pd.DataFrame(records).set_index("program")

    # FDR 校正（只对做了检验的行）
    valid = result["pval"].notna()
    if valid.sum() > 0:
        _, padj, _, _ = multipletests(result.loc[valid, "pval"], method="fdr_bh")
        result.loc[valid, "padj"] = padj
    else:
        result["padj"] = np.nan

    print(f"\n共 {valid.sum()}/{len(programs)} 个 program 完成方向检验")
    return result


def compute_leading_genes(
    data: dict,
    config: dict,
    program_direction: pd.DataFrame,
    top_n: int = 15,
) -> dict:
    """对每个已完成方向检验的 program，从前景基因里挑出"leading genes"——
    即该基因的 gene-level wPSI 离背景中位数最远（且偏离方向与program整体
    方向一致）的那批基因，用来回答"这个program的方向判定主要是谁在驱动"。

    只依赖 compute_program_direction() 内部同样的计算方式（前景基因 =
    spectra_z > z_threshold 的PAC对应基因，背景按 background_scope 定义），
    但这里不是compute_program_direction()的重复实现——直接复用它的
    输出（program_direction）里已经确定的方向和背景中位数，只重新拿一次
    前景基因列表来算每个基因的偏离量，避免两处逻辑分叉。

    注意：这里的排序统计量 delta_from_bg 基于中位数，当多个基因的
    gene_wpsi 恰好都精确等于同一个值（常见于检出率刚过阈值的边界基因）
    时会出现并列、无法区分谁更重要。若需要更细致地区分这类并列基因，
    应在下游用 summarize_leading_gene_within_celltype() 的效应量
    （非零比例、均值）重新评估，而不是只依赖这里的排序。

    Parameters
    ----------
    program_direction : DataFrame
        compute_program_direction() 的返回值（需要用到 bg_median 列）。
    top_n : int
        每个program最多保留多少个leading gene。

    Returns
    -------
    dict: {program: DataFrame(columns=[gene_id, symbol, gene_wpsi,
           delta_from_bg, spectra_z_max])}
        按 |delta_from_bg| 降序排列。direction为"NA"的program（前景基因
        不足）值为空表。gene_wpsi 用的还是跨细胞取中位数后的基因级wPSI，
        和compute_program_direction()里算法一致，不是重新按其他方式算的。
    """
    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过 leading genes 计算")
        return {}

    spectra_z = data["spectra_z"]
    feature_anno = data["feature_anno"]
    programs = data["programs"]
    z_threshold = config.get("z_threshold", 1.5)
    symbol_map = data.get("gene_symbol_map")

    def _symbol(gene_id):
        if symbol_map is None:
            return gene_id
        if gene_id in symbol_map.index:
            return symbol_map.loc[gene_id]
        bare_id = gene_id.split(".")[0]
        if bare_id in symbol_map.index:
            return symbol_map.loc[bare_id]
        return gene_id

    min_detection_rate = config.get("min_detection_rate", 0.1)
    gene_wpsi = _compute_detection_filtered_gene_wpsi(wpsi, min_detection_rate)
    pac_to_gene = feature_anno["gene_id"]

    result = {}
    for prog in programs:
        row = program_direction.loc[prog]
        if row["direction"] == "NA" or pd.isna(row["bg_median"]):
            print(f"  [{prog}] 方向检验未完成，跳过 leading genes")
            result[prog] = pd.DataFrame(columns=["gene_id", "symbol", "gene_wpsi", "delta_from_bg", "spectra_z_max"])
            continue

        z_row = spectra_z.loc[prog]
        top_pacs = z_row[z_row > z_threshold].index
        top_pacs = top_pacs.intersection(pac_to_gene.index)
        fg_genes = pac_to_gene.loc[top_pacs]
        fg_genes = fg_genes[fg_genes.isin(gene_wpsi.index)]

        # 每个基因可能对应多个前景PAC，取该基因在这些PAC里最大的spectra_z
        # 作为它在这个program里的"权重强度"，方便和delta一起判断谁最重要
        gene_to_max_z = {}
        for pac, gid in fg_genes.items():
            z_val = z_row[pac]
            if gid not in gene_to_max_z or z_val > gene_to_max_z[gid]:
                gene_to_max_z[gid] = z_val

        bg_median = row["bg_median"]
        recs = []
        for gid in fg_genes.unique():
            gw = gene_wpsi.loc[gid]
            recs.append({
                "gene_id": gid,
                "symbol": _symbol(gid),
                "gene_wpsi": float(gw),
                "delta_from_bg": float(gw - bg_median),
                "spectra_z_max": float(gene_to_max_z[gid]),
            })
        df = pd.DataFrame(recs)
        # 方向一致：Distal program(delta_median>0)保留delta_from_bg>0的基因，
        # Proximal program反之——防止把和整体方向相反的"噪音基因"排进leading gene
        if row["direction"] == "Distal":
            df = df[df["delta_from_bg"] > 0]
        elif row["direction"] == "Proximal":
            df = df[df["delta_from_bg"] < 0]
        df = df.reindex(df["delta_from_bg"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)

        print(f"  [{prog}] direction={row['direction']}, leading genes: "
              f"{df['symbol'].tolist()[:5]}{'...' if len(df) > 5 else ''}")
        result[prog] = df

    return result


def plot_program_direction_summary(
    program_direction: pd.DataFrame,
    config: dict,
    out_dir: str,
    padj_threshold: float = 0.05,
    pval_threshold: float = 0.05,
    filename: str = "layer1_program_direction_summary.png",
) -> str:
    """画方向检验的汇总条形图（delta_median，按方向着色，标注显著性）。

    传入的是 compute_program_direction() 的结果，不重新计算。
    样本量小、检验普遍不显著时（比如COVID这批数据），仍然按 delta_median
    大小排序展示，用星号区分"padj显著"/"仅raw pval显著"/"不显著"三档，
    不因为没有显著结果就不出图——描述性的方向倾向本身也是有信息量的。

    Parameters
    ----------
    program_direction : DataFrame
        compute_program_direction() 的返回值。
    out_dir : str
        图片保存目录（自动创建）。
    padj_threshold, pval_threshold : float
        显著性星号标注阈值。

    Returns
    -------
    str: 保存的图片路径。若 program_direction 为 None，打印提示并返回空字符串。
    """
    if program_direction is None:
        print("  [Layer1] program_direction 为 None，跳过画图")
        return ""

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    df = program_direction.copy()
    df = df[df["direction"] != "NA"].sort_values("delta_median")
    if len(df) == 0:
        print("  [Layer1] 没有任何 program 完成检验，跳过画图")
        return ""

    def _sig_marker(row):
        if pd.notna(row["padj"]) and row["padj"] < padj_threshold:
            return "**"
        if pd.notna(row["pval"]) and row["pval"] < pval_threshold:
            return "*"
        return ""

    colors = ["#4C72B0" if d == "Proximal" else "#DD8452" for d in df["direction"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df))))
    bars = ax.barh(df.index, df["delta_median"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ median wPSI (foreground − background)")
    ax.set_title("Program APA direction (wPSI-based MWU test)\n"
                  "blue = Proximal, orange = Distal   (** padj<%.2f, * pval<%.2f)"
                  % (padj_threshold, pval_threshold))

    max_abs = df["delta_median"].abs().max()
    pad = max_abs * 0.35 + 1e-6
    ax.set_xlim(df["delta_median"].min() - pad, df["delta_median"].max() + pad)

    for bar, (_, row) in zip(bars, df.iterrows()):
        marker = _sig_marker(row)
        label = f"n={int(row['n_fg_genes'])}{'  ' + marker if marker else ''}"
        x = bar.get_width()
        offset = pad * 0.15 if x >= 0 else -pad * 0.15
        ha = "left" if x >= 0 else "right"
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2, label,
                 va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────
# 细胞类型内部对照：排除"疾病关联是否被细胞组成混淆"这一可能性
# ────────────────────────────────────────────────────────────────

def compare_within_celltype(
    data: dict,
    config: dict,
    celltype: str,
    disease_column: str,
    group_a: str,
    group_b: str,
    gene_id: str | None = None,
    program: str | None = None,
) -> dict:
    """在指定的单一细胞类型（cluster）内部，比较两个疾病分组的差异——
    用于回答"某个基因/program的组间差异，是不是只是细胞组成比例不同带来的
    假象，还是在同一细胞状态下依然存在的独立信号"。

    gene_id 和 program 至少传一个，两个都传则两层都算。
    gene_id 比较的是该基因的 wPSI（单细胞级，来自 data["wpsi"]）；
    program 比较的是该 program 的 usage（来自 data["usage"]）。

    注意：这里用的是中位数比较，适合快速筛查（尤其program usage这类
    非零膨胀的连续指标）。但对稀疏的单基因wPSI，中位数容易被钉死在0，
    看不出真实差异——需要更细致的基因层面判断时，改用
    summarize_leading_gene_within_celltype()（基于非零比例和均值的
    效应量分类），不要只依赖这里的median结果下结论。

    Parameters
    ----------
    celltype : str
        限定比较范围的细胞类型标签（data["meta"][celltype_col] 里的取值）。
    disease_column : str
        用来分两组的meta列名，如 config["contrasts"][0]["column"]。
    group_a, group_b : str
        该列下要比较的两个取值。

    Returns
    -------
    dict，可能含 "wpsi" 和/或 "usage" 两个key，每个value是：
        {"n_a", "n_b", "median_a", "median_b", "pval"}
        某一层因为数据缺失或细胞数不足被跳过时，对应key不存在（不是None，
        直接不出现在返回dict里，调用方用 .get() 判断）。
    """
    meta = data["meta"]
    celltype_col = config["meta_columns"]["celltype"]

    mask = meta[celltype_col] == celltype
    sub_meta = meta.loc[mask]
    groups = sub_meta[disease_column]

    print(f"=== 组内比较: celltype='{celltype}', {disease_column}: "
          f"{group_a} vs {group_b} ===")
    print(f"  该celltype总细胞数: {len(sub_meta)} "
          f"({group_a}: {(groups == group_a).sum()}, {group_b}: {(groups == group_b).sum()})")

    result = {}

    if gene_id is not None:
        wpsi = data.get("wpsi")
        if wpsi is None:
            print("  [wPSI] 未提供wPSI矩阵，跳过基因层面比较")
        elif gene_id not in wpsi.index:
            print(f"  [wPSI] {gene_id} 不在wPSI矩阵里，跳过")
        else:
            common_cells = wpsi.columns.intersection(sub_meta.index)
            vals = wpsi.loc[gene_id, common_cells]
            g = groups.loc[common_cells]
            a_vals = vals[g == group_a].dropna()
            b_vals = vals[g == group_b].dropna()
            if len(a_vals) < 20 or len(b_vals) < 20:
                print(f"  [wPSI] {group_a}细胞数={len(a_vals)}, {group_b}细胞数={len(b_vals)}，"
                      f"某一组不足20个细胞，跳过检验")
            else:
                stat, pval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                print(f"  [wPSI] {group_a} median={a_vals.median():.4f} (n={len(a_vals)})  "
                      f"{group_b} median={b_vals.median():.4f} (n={len(b_vals)})  pval={pval:.3e}")
                result["wpsi"] = {
                    "n_a": len(a_vals), "n_b": len(b_vals),
                    "median_a": float(a_vals.median()), "median_b": float(b_vals.median()),
                    "pval": float(pval),
                }

    if program is not None:
        usage = data["usage"]
        common_cells = usage.index.intersection(sub_meta.index)
        vals = usage.loc[common_cells, program]
        g = groups.loc[common_cells]
        a_vals = vals[g == group_a].dropna()
        b_vals = vals[g == group_b].dropna()
        if len(a_vals) < 20 or len(b_vals) < 20:
            print(f"  [usage] {group_a}细胞数={len(a_vals)}, {group_b}细胞数={len(b_vals)}，"
                  f"某一组不足20个细胞，跳过检验")
        else:
            stat, pval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            print(f"  [usage:{program}] {group_a} median={a_vals.median():.4f} (n={len(a_vals)})  "
                  f"{group_b} median={b_vals.median():.4f} (n={len(b_vals)})  pval={pval:.3e}")
            result["usage"] = {
                "n_a": len(a_vals), "n_b": len(b_vals),
                "median_a": float(a_vals.median()), "median_b": float(b_vals.median()),
                "pval": float(pval),
            }

    if not result:
        print("  两层都被跳过，没有可用结果")

    return result


def _compare_gene_within_celltype_effect_size(
    gene_id: str,
    data: dict,
    config: dict,
    celltype: str,
    disease_column: str,
    group_a: str,
    group_b: str,
    min_cells: int = 20,
) -> dict | None:
    """在单一细胞类型内部比较单个基因的wPSI，返回效应量指标（非零比例、均值）
    而不是中位数——中位数在稀疏的单细胞wPSI数据里经常被钉死在0，看不出
    真实差异（参见 _compute_detection_filtered_gene_wpsi 的检出率过滤问题），
    效应量更可靠。

    与 compare_within_celltype 的区别：那个函数返回median用于快速筛查，
    这个函数专门服务于 summarize_leading_gene_within_celltype 的批量效应量
    评估，两者不合并是因为返回字段和使用场景不同。

    Returns
    -------
    dict 或 None（细胞数不足/基因不在wPSI矩阵里时）。
    """
    wpsi = data.get("wpsi")
    if wpsi is None or gene_id not in wpsi.index:
        return None

    meta = data["meta"]
    celltype_col = config["meta_columns"]["celltype"]
    mask = meta[celltype_col] == celltype
    sub_meta = meta.loc[mask]
    common_cells = wpsi.columns.intersection(sub_meta.index)

    vals = wpsi.loc[gene_id, common_cells]
    groups = sub_meta.loc[common_cells, disease_column]
    a_vals = vals[groups == group_a].dropna()
    b_vals = vals[groups == group_b].dropna()

    if len(a_vals) < min_cells or len(b_vals) < min_cells:
        return None

    _, pval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    return {
        "nonzero_frac_a": float((a_vals > 0).mean()),
        "nonzero_frac_b": float((b_vals > 0).mean()),
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "pval": float(pval),
    }


def summarize_leading_gene_within_celltype(
    program: str,
    leading_genes: dict,
    data: dict,
    config: dict,
    celltype: str,
    disease_column: str,
    group_a: str,
    group_b: str,
    effect_threshold: float = 0.01,
    pval_threshold: float = 0.05,
    min_cells: int = 20,
) -> pd.DataFrame:
    """对某个program的全部leading gene，逐个做细胞类型内部对照，
    并按效应量+显著性分类为"支持program整体方向/方向相反/无实质差异"三类。

    这是把leading gene组内对照的循环+分类逻辑封装成的可复用流程，跑其他
    program（如P04/P07）时只需换 program/celltype 参数，不用重写循环。

    Parameters
    ----------
    program : str
        要分析的program名字（如"P02"），仅用于打印和结果标注，
        实际取的是 leading_genes[program] 这批基因。
    leading_genes : dict
        compute_leading_genes() 的返回值。
    celltype : str
        限定比较范围的细胞类型标签。
    disease_column : str
        用来分两组的meta列名。
    group_a, group_b : str
        该列下要比较的两个取值。分类的"supports/opposes"是相对
        mean_b > mean_a 而言（即该基因在group_b里wPSI更高时归为
        "supports"）。调用方需确保group_a/group_b的顺序和program usage
        比较时一致（哪一组对应program usage更高的那一组），否则
        "supports/opposes"的语义会和program层面的结论对不上。
    effect_threshold : float
        |mean_diff| 低于此值视为"无实质差异"，即使pval显著也不算数
        （大样本下微小差异也可能显著，这个阈值用来过滤这类噪音）。
    pval_threshold : float
        显著性阈值。
    min_cells : int
        某一组细胞数低于此值时跳过该基因。

    Returns
    -------
    DataFrame，index=symbol，含 nonzero_frac_a/b, mean_a/b, pval,
    mean_diff, abs_mean_diff, category（"supports"/"opposes"/"no_meaningful_diff"）。
    按 abs_mean_diff 降序排列。某基因因细胞数不足被跳过时不出现在结果里
    （不是保留空行），并打印提示。
    """
    genes_df = leading_genes.get(program)
    if genes_df is None or len(genes_df) == 0:
        print(f"  [{program}] 没有leading genes，跳过")
        return pd.DataFrame()

    rows = []
    for _, row in genes_df.iterrows():
        res = _compare_gene_within_celltype_effect_size(
            row["gene_id"], data, config, celltype, disease_column,
            group_a, group_b, min_cells=min_cells,
        )
        if res is None:
            print(f"  [{program}] {row['symbol']} 细胞数不足或不在wPSI矩阵里，跳过")
            continue
        rows.append({"symbol": row["symbol"], "gene_id": row["gene_id"], **res})

    if not rows:
        print(f"  [{program}] 没有任何基因有足够细胞数完成比较")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["mean_diff"] = df["mean_b"] - df["mean_a"]
    df["abs_mean_diff"] = df["mean_diff"].abs()

    def _classify(r):
        if r["pval"] >= pval_threshold or r["abs_mean_diff"] < effect_threshold:
            return "no_meaningful_diff"
        return "supports" if r["mean_diff"] > 0 else "opposes"

    df["category"] = df.apply(_classify, axis=1)
    df = df.sort_values("abs_mean_diff", ascending=False).set_index("symbol")

    counts = df["category"].value_counts()
    print(f"  [{program}] leading gene组内对照分类: "
          f"supports={counts.get('supports', 0)}, "
          f"opposes={counts.get('opposes', 0)}, "
          f"no_meaningful_diff={counts.get('no_meaningful_diff', 0)}")

    return df

# ────────────────────────────────────────────────────────────────
# usage分箱趋势：验证leading genes的wPSI是否随usage单调变化
# ────────────────────────────────────────────────────────────────

def compute_usage_binned_wpsi(
    data: dict,
    config: dict,
    leading_genes: dict,
    program: str,
    n_bins: int = 10,
    top_n_genes: int = 5,
):
    """把细胞按program usage切成n_bins个分位数bin，计算该program leading genes
    的"每细胞平均wPSI"在各bin里的均值和95% CI，用于验证"usage越高，
    compute_program_direction() 判定的方向是否在细胞尺度上也单调地体现出来"。

    只用leading_genes（而不是全部前景基因），是因为leading genes已经是
    方向一致、偏离量最大的那批基因，用它们做趋势验证最能反映program的核心
    方向信号，不会被噪音基因稀释。

    Parameters
    ----------
    leading_genes : dict
        compute_leading_genes() 的返回值。
    program : str
        要验证哪个program。
    n_bins : int
        usage分位数分箱数。
    top_n_genes : int
        额外单独画出分bin轨迹的代表基因数（按高/低usage两端wPSI差值排序）。

    Returns
    -------
    (summary, top_genes_binned, spearman_r, spearman_p) 或 None（wPSI矩阵缺失/
    该program没有leading genes时）。
        summary: DataFrame [usage_median, wpsi_mean, wpsi_ci_low, wpsi_ci_high, n_cells]
        top_genes_binned: DataFrame index=bin, columns=代表基因symbol
        spearman_r/p: usage与每细胞平均wPSI的Spearman相关（单调性定量指标）
    """
    wpsi = data.get("wpsi")
    if wpsi is None:
        print(f"  [{program}] 未提供 wPSI 矩阵，跳过 usage 分箱趋势")
        return None

    genes_df = leading_genes.get(program)
    if genes_df is None or len(genes_df) == 0:
        print(f"  [{program}] 没有leading genes，跳过 usage 分箱趋势")
        return None

    usage = data["usage"][program]
    gene_ids = pd.Index(genes_df["gene_id"]).intersection(wpsi.index)
    if len(gene_ids) == 0:
        print(f"  [{program}] leading genes 都不在 wPSI 矩阵里，跳过")
        return None
    print(f"  [{program}] {len(gene_ids)}/{len(genes_df)} 个leading genes用于usage分箱趋势")

    sub_wpsi = wpsi.loc[gene_ids]
    per_cell_mean = sub_wpsi.mean(axis=0, skipna=True)

    common_cells = usage.index.intersection(per_cell_mean.index)
    usage = usage.loc[common_cells]
    per_cell_mean = per_cell_mean.loc[common_cells]

    bins = pd.qcut(usage, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"usage": usage, "wpsi_mean": per_cell_mean, "bin": bins}).dropna()

    def _ci(x):
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return pd.Series({"mean": m, "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se, "n": len(x)})

    grouped = df.groupby("bin", observed=True)["wpsi_mean"].apply(_ci).unstack()
    usage_median = df.groupby("bin", observed=True)["usage"].median()
    summary = pd.DataFrame({
        "usage_median": usage_median,
        "wpsi_mean": grouped["mean"],
        "wpsi_ci_low": grouped["ci_low"],
        "wpsi_ci_high": grouped["ci_high"],
        "n_cells": grouped["n"],
    }).reset_index(drop=True)

    # 代表基因：高/低usage两端wPSI差值最大的top_n_genes个
    high_cells = usage[usage >= usage.quantile(0.8)].index.intersection(sub_wpsi.columns)
    low_cells = usage[usage <= usage.quantile(0.2)].index.intersection(sub_wpsi.columns)
    effect = (sub_wpsi[high_cells].mean(axis=1) - sub_wpsi[low_cells].mean(axis=1)).abs()
    top_gene_ids = effect.sort_values(ascending=False).head(top_n_genes).index
    id_to_symbol = genes_df.set_index("gene_id")["symbol"]

    top_records = {}
    for gid in top_gene_ids:
        label = id_to_symbol.get(gid, gid)
        gene_binned = sub_wpsi.loc[gid].reindex(common_cells)
        top_records[label] = (
            pd.DataFrame({"bin": bins, "val": gene_binned}).dropna()
            .groupby("bin", observed=True)["val"].mean()
        )
    top_genes_binned = pd.DataFrame(top_records)

    spearman_r, spearman_p = spearmanr(df["usage"], df["wpsi_mean"])
    print(f"  [{program}] usage vs 每细胞平均wPSI 的Spearman相关: r={spearman_r:.3f}, p={spearman_p:.2e}")

    return summary, top_genes_binned, float(spearman_r), float(spearman_p)


# ────────────────────────────────────────────────────────────────
# 模块内部基因方向一致性：不看pooled显著性，看符号是否一致
# ────────────────────────────────────────────────────────────────

def compute_foreground_gene_directions(
    data: dict,
    config: dict,
    program_direction: pd.DataFrame,
    min_fg_genes: int = 5,
):
    """对每个program的全部前景基因（不截断top_n、不按方向过滤）计算delta_from_bg，
    并做符号一致性检验：不管幅度大小，这批基因里有多少个偏向program整体判定的
    方向。用二项检验(binomtest)而不是MWU，因为这里问的是"方向是否一致"而不是
    "pooled中位数是否有显著偏移"——两者是不同的问题，前者对基因间幅度异质性
    不敏感，样本量小(几十个基因)时往往比MWU更有力。

    跟 compute_leading_genes() 的关系：那个函数会按 |delta_from_bg| 排序、
    只保留top_n、且只保留方向一致的基因（用于"谁是驱动者"）；这个函数保留
    全部前景基因、不管方向（用于"整体上偏不偏一边"），两者服务的问题不同，
    不是重复实现。返回的gene_tables列结构和compute_leading_genes()对齐
    （symbol/delta_from_bg等），可以直接传给 viz.plot_leading_genes_bar()。

    Parameters
    ----------
    program_direction : DataFrame
        compute_program_direction() 的返回值（需要 bg_median / delta_median 列）。
    min_fg_genes : int
        前景基因数（delta_from_bg非零的）少于这个值时跳过该program的检验。

    Returns
    -------
    (gene_tables, consistency_summary)
        gene_tables : dict {program: DataFrame(columns=[gene_id, symbol, gene_wpsi,
                      delta_from_bg, spectra_z_max])}，全部前景基因，未过滤未截断，
                      按|delta_from_bg|降序排列。
        consistency_summary : DataFrame，index=program，列：
                      n_fg_genes, n_pos, n_neg, n_zero, n_consistent, consistency_frac,
                      binom_pval（该program方向判断为"NA"或前景基因不足时，
                      后几列为NaN）。
    """
    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过方向一致性检验")
        return {}, pd.DataFrame()

    spectra_z = data["spectra_z"]
    feature_anno = data["feature_anno"]
    programs = data["programs"]
    z_threshold = config.get("z_threshold", 1.5)
    symbol_map = data.get("gene_symbol_map")

    def _symbol(gene_id):
        if symbol_map is None:
            return gene_id
        if gene_id in symbol_map.index:
            return symbol_map.loc[gene_id]
        bare_id = gene_id.split(".")[0]
        if bare_id in symbol_map.index:
            return symbol_map.loc[bare_id]
        return gene_id

    min_detection_rate = config.get("min_detection_rate", 0.1)
    gene_wpsi = _compute_detection_filtered_gene_wpsi(wpsi, min_detection_rate)
    pac_to_gene = feature_anno["gene_id"]

    gene_tables = {}
    summary_rows = []
    for prog in programs:
        row = program_direction.loc[prog]
        if pd.isna(row["bg_median"]):
            print(f"  [{prog}] 背景中位数缺失（前景基因数不足），跳过")
            gene_tables[prog] = pd.DataFrame(columns=["gene_id", "symbol", "gene_wpsi", "delta_from_bg", "spectra_z_max"])
            summary_rows.append({"program": prog, "n_fg_genes": 0, "n_pos": np.nan, "n_neg": np.nan,
                                  "n_zero": np.nan, "n_consistent": np.nan, "consistency_frac": np.nan, "binom_pval": np.nan})
            continue

        z_row = spectra_z.loc[prog]
        top_pacs = z_row[z_row > z_threshold].index
        top_pacs = top_pacs.intersection(pac_to_gene.index)
        fg_genes = pac_to_gene.loc[top_pacs]
        fg_genes = fg_genes[fg_genes.isin(gene_wpsi.index)]

        gene_to_max_z = {}
        for pac, gid in fg_genes.items():
            z_val = z_row[pac]
            if gid not in gene_to_max_z or z_val > gene_to_max_z[gid]:
                gene_to_max_z[gid] = z_val

        bg_median = row["bg_median"]
        recs = []
        for gid in fg_genes.unique():
            gw = gene_wpsi.loc[gid]
            recs.append({
                "gene_id": gid, "symbol": _symbol(gid),
                "gene_wpsi": float(gw), "delta_from_bg": float(gw - bg_median),
                "spectra_z_max": float(gene_to_max_z[gid]),
            })
        df = pd.DataFrame(recs)
        df = df.reindex(df["delta_from_bg"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        gene_tables[prog] = df

        n_pos = int((df["delta_from_bg"] > 0).sum())
        n_neg = int((df["delta_from_bg"] < 0).sum())
        n_zero = len(df) - n_pos - n_neg
        n_total = n_pos + n_neg

        if n_total < min_fg_genes:
            print(f"  [{prog}] 非零前景基因数 {n_total} < min_fg_genes={min_fg_genes}，跳过一致性检验")
            summary_rows.append({"program": prog, "n_fg_genes": len(df), "n_pos": n_pos, "n_neg": n_neg,
                                  "n_zero": n_zero, "n_consistent": np.nan, "consistency_frac": np.nan, "binom_pval": np.nan})
            continue

        n_consistent = n_pos if row["delta_median"] >= 0 else n_neg
        binom_res = binomtest(n_consistent, n_total, p=0.5, alternative="greater")
        consistency_frac = n_consistent / n_total

        print(f"  [{prog}] 前景基因{len(df)}个(非零{n_total}个): 与program方向({row['direction']})"
              f"一致 {n_consistent}/{n_total} ({consistency_frac:.1%}), 符号检验 p={binom_res.pvalue:.3f}")

        summary_rows.append({
            "program": prog, "n_fg_genes": len(df), "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
            "n_consistent": n_consistent, "consistency_frac": consistency_frac, "binom_pval": float(binom_res.pvalue),
        })

    consistency_summary = pd.DataFrame(summary_rows).set_index("program")
    return gene_tables, consistency_summary

# ────────────────────────────────────────────────────────────────
# 限定单一细胞类型内部重做方向检验：排查pooled结果是否被其他细胞类型混淆
# ────────────────────────────────────────────────────────────────

def subset_data_to_celltype(data: dict, config: dict, celltype: str, min_cells: int = 50) -> dict | None:
    """构造一份只包含指定细胞类型的data浅拷贝，用于把方向检验/usage分箱趋势
    限定在单一细胞类型内部重跑——排查pooled结果(全体细胞一起算)是否被"该
    program本质是某细胞类型的开关型指标、pooled背景其实是其余细胞类型大
    杂烩"这种情况混淆（典型场景见P06：usage对Plasma身份AUC=0.98，但usage-
    binned趋势图显示usage越高wPSI越低，与pooled方向判断相反）。

    只浅拷贝并替换 wpsi/usage/meta 这几个跟细胞对齐的字段（按该celltype筛选），
    spectra_z/feature_anno/programs/gene_symbol_map等跟基因或program绑定、
    不随细胞变化的字段原样保留，不重新计算。返回值可以直接传给
    compute_program_direction / compute_leading_genes /
    compute_foreground_gene_directions / compute_usage_binned_wpsi等现有函数，
    这些函数内部不需要任何改动。

    Parameters
    ----------
    celltype : str
        data["meta"][config["meta_columns"]["celltype"]] 里的取值。
    min_cells : int
        细胞数低于这个阈值直接跳过（太少的话背景基因池会不稳）。

    Returns
    -------
    dict，结构跟原data相同；细胞数不足min_cells时打印提示并返回None。
    """
    meta = data["meta"]
    celltype_col = config["meta_columns"]["celltype"]
    cells = meta.index[meta[celltype_col] == celltype]
    n_cells = len(cells)
    if n_cells < min_cells:
        print(f"  [subset] {celltype} 细胞数={n_cells} < min_cells={min_cells}，跳过")
        return None
    print(f"  [subset] 限定 celltype={celltype!r}，共 {n_cells} 个细胞")

    sub_data = dict(data)  # 浅拷贝，不改动原data

    wpsi = data.get("wpsi")
    if wpsi is not None:
        common_cells = wpsi.columns.intersection(cells)
        sub_data["wpsi"] = wpsi[common_cells]

    usage = data.get("usage")
    if usage is not None:
        common_cells = usage.index.intersection(cells)
        sub_data["usage"] = usage.loc[common_cells]

    sub_data["meta"] = meta.loc[meta.index.intersection(cells)]
    return sub_data
