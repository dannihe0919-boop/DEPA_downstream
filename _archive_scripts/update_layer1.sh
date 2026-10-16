cat > depa_downstream/layer1_direction.py << 'PYEOF_MARKER'
"""
Layer 1 — 程序特征基础：基于 wPSI 的 APA 方向验证（MWU版）。

对应原 notebook 的 Cell 2。核心逻辑不依赖数据集语义（不管 program 是按疾病
分组还是按细胞身份/状态分组跑出来的，方向检验的算法是一样的），
所以这层几乎不需要因为数据集不同而改函数本身，只是入参（wpsi/spectra_z）不同。

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
"""

from __future__ import annotations

import os

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
        用到 config 里的：
        "z_threshold"（默认 1.5，前景PAC筛选阈值）
        "background_scope"（默认 "pac_universe"，见模块docstring）
    min_fg_genes : int
        前景基因数少于这个值的 program 跳过，不做检验（样本太小检验没意义）。

    Returns
    -------
    DataFrame，index=program，列包括：
        n_fg_genes, n_bg_genes, fg_median, bg_median, delta_median,
        mwu_stat, pval, padj, direction ("Proximal" / "Distal" / "NA")
    若 data["wpsi"] 为 None（该数据集没有配置 wPSI 矩阵），打印提示并返回 None，
    调用方需要对 None 做判断，不能默认它一定存在。
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

    # ── 1. 基因级 wPSI（跨细胞取中位数）──────────────────────────
    print(f"计算基因级 wPSI（{wpsi.shape[0]} 个基因 x {wpsi.shape[1]} 个细胞 -> 取跨细胞中位数）...")
    gene_wpsi = wpsi.median(axis=1, skipna=True)
    gene_wpsi = gene_wpsi.dropna()
    print(f"  有效基因数（非全NaN）: {len(gene_wpsi)}")

    # PAC -> gene_id 映射
    pac_to_gene = feature_anno["gene_id"]

    # background_scope="pac_universe" 时的候选背景基因池：
    # 本次分析涉及的所有PAC对应的基因（去重），再和 gene_wpsi 取交集
    pac_universe_genes = pd.Index(pac_to_gene.dropna().unique()).intersection(gene_wpsi.index)
    print(f"  cNMF分析涉及的基因全集（pac_universe）大小: {len(pac_universe_genes)}")
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
        direction = "Distal" if delta > 0 else "Proximal"

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


def plot_program_direction_summary(
    program_direction: pd.DataFrame,
    config: dict,
    out_dir: str,
    padj_threshold: float = 0.05,
    pval_threshold: float = 0.05,
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
    out_path = os.path.join(out_dir, "layer1_program_direction_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path
PYEOF_MARKER

echo "layer1_direction.py 已更新（新增plot_program_direction_summary画图函数）"