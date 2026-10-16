"""
纯画图函数。原则：只接收已经算好的结果（DataFrame/dict），不在这里做任何
统计计算——计算逻辑都应该在 layer1/2/3 里，方便单独测试和复用。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

def plot_gene_wpsi_by_group(
    gene_id: str,
    data: dict,
    config: dict,
    groupby_col: str,
    out_dir: str,
    gene_symbol: str | None = None,
) -> str:
    """单个基因的wPSI在某个meta分组下的箱线图（叠加散点抖动），用single-cell级别
    的wPSI值（不是取中位数后的），直接回答"这个基因的APA变化在各组里是什么样"。

    Parameters
    ----------
    gene_id : str
        要画的基因（wPSI矩阵的index）。
    groupby_col : str
        data["meta"]里的分组列名，如 config["meta_columns"]["celltype"]
        或 contrasts里用到的列（如"group"）。
    gene_symbol : str, optional
        图标题显示用的symbol，不传则直接显示gene_id。
    """
    import matplotlib.pyplot as plt

    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过单基因分组图")
        return ""
    if gene_id not in wpsi.index:
        print(f"  [Layer1] {gene_id} 不在 wPSI 矩阵里，跳过")
        return ""

    meta = data["meta"]
    common_cells = wpsi.columns.intersection(meta.index)
    vals = wpsi.loc[gene_id, common_cells]
    groups = meta.loc[common_cells, groupby_col]

    plot_df = pd.DataFrame({"wpsi": vals.values, "group": groups.values}).dropna()
    if len(plot_df) == 0:
        print(f"  [Layer1] {gene_id} 在该分组列下没有有效数据，跳过")
        return ""

    os.makedirs(out_dir, exist_ok=True)
    order = sorted(plot_df["group"].unique())
    data_by_group = [plot_df.loc[plot_df["group"] == g, "wpsi"].values for g in order]

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(order)), 4))
    bp = ax.boxplot(data_by_group, labels=order, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.5)
    # 叠加抖动散点，看真实分布不只是箱线摘要
    rng = np.random.default_rng(0)
    for i, vals_i in enumerate(data_by_group, start=1):
        jitter = rng.normal(i, 0.06, size=len(vals_i))
        ax.scatter(jitter, vals_i, s=4, alpha=0.3, color="black")

    label = gene_symbol or gene_id
    ax.set_ylabel("wPSI (per cell)")
    ax.set_title(f"{label} — wPSI by {groupby_col}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"layer1_gene_wpsi_{label}_{groupby_col}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path

def plot_leading_genes_bar(leading_genes: dict, program: str, out_dir: str, direction_label: str = "") -> str:
    """画单个program的leading genes条形图：横轴delta_from_bg，纵轴基因symbol，
    按|delta|降序排列。用来直观回答"这个program的方向判定主要是谁在驱动"。

    Parameters
    ----------
    leading_genes : dict
        compute_leading_genes() 的返回值。
    program : str
        要画哪个program。
    direction_label : str
        可选，program整体方向（"Proximal"/"Distal"），显示在标题里。
    """
    import matplotlib.pyplot as plt

    df = leading_genes.get(program)
    if df is None or len(df) == 0:
        print(f"  [Layer1] {program} 没有leading genes，跳过画图")
        return ""

    os.makedirs(out_dir, exist_ok=True)
    df = df.sort_values("delta_from_bg")

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(df))))
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in df["delta_from_bg"]]
    ax.barh(df["symbol"], df["delta_from_bg"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("gene wPSI − program background median")
    title = f"{program} leading genes"
    if direction_label:
        title += f" ({direction_label})"
    ax.set_title(title)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"layer1_leading_genes_{program}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path

def plot_gene_wpsi_by_group(
    gene_id: str,
    data: dict,
    config: dict,
    groupby_col: str,
    out_dir: str,
    gene_symbol: str | None = None,
) -> str:
    """单个基因的wPSI在某个meta分组下的箱线图（叠加散点抖动），用single-cell级别
    的wPSI值（不是取中位数后的），直接回答"这个基因的APA变化在各组里是什么样"。

    Parameters
    ----------
    gene_id : str
        要画的基因（wPSI矩阵的index）。
    groupby_col : str
        data["meta"]里的分组列名，如 config["meta_columns"]["celltype"]
        或 contrasts里用到的列（如"group"）。
    gene_symbol : str, optional
        图标题显示用的symbol，不传则直接显示gene_id。
    """
    import matplotlib.pyplot as plt

    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer1] 未提供 wPSI 矩阵，跳过单基因分组图")
        return ""
    if gene_id not in wpsi.index:
        print(f"  [Layer1] {gene_id} 不在 wPSI 矩阵里，跳过")
        return ""

    meta = data["meta"]
    common_cells = wpsi.columns.intersection(meta.index)
    vals = wpsi.loc[gene_id, common_cells]
    groups = meta.loc[common_cells, groupby_col]

    plot_df = pd.DataFrame({"wpsi": vals.values, "group": groups.values}).dropna()
    if len(plot_df) == 0:
        print(f"  [Layer1] {gene_id} 在该分组列下没有有效数据，跳过")
        return ""

    os.makedirs(out_dir, exist_ok=True)
    order = sorted(plot_df["group"].unique())
    data_by_group = [plot_df.loc[plot_df["group"] == g, "wpsi"].values for g in order]

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(order)), 4))
    bp = ax.boxplot(data_by_group, labels=order, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.5)
    # 叠加抖动散点，看真实分布不只是箱线摘要
    rng = np.random.default_rng(0)
    for i, vals_i in enumerate(data_by_group, start=1):
        jitter = rng.normal(i, 0.06, size=len(vals_i))
        ax.scatter(jitter, vals_i, s=4, alpha=0.3, color="black")

    label = gene_symbol or gene_id
    ax.set_ylabel("wPSI (per cell)")
    ax.set_title(f"{label} — wPSI by {groupby_col}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"layer1_gene_wpsi_{label}_{groupby_col}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


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
                     color="white" if val < vmax * 0.6 else "black", fontsize=8)
    ax.set_title("Program gene-set Jaccard similarity")
    fig.colorbar(im, ax=ax, label="Jaccard index", fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "layer3_jaccard_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_enrichment_bubble(
        enrichment_by_label: dict,
        out_dir: str,
        top_n: int = 10,
        padj_threshold: float = 0.05,
        filename: str = "layer3_enrichment_bubble.png",
) -> str:
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
    filename : str
        输出图片文件名（不含目录）。默认 "layer3_enrichment_bubble.png"。
        同一批分析里如果要对不同基因列表（比如"专属marker" vs "全部核心基因"）
        各画一张图，传不同的 filename 避免互相覆盖。
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

    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), max(3, 0.35 * len(term_order))))
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
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=8)
    ax.set_title(f"Top enriched terms (padj < {padj_threshold})")
    fig.colorbar(sc, ax=ax, label="combined score", fraction=0.04, pad=0.08)

    for size_val, label_txt in [(2, "padj=1e-2"), (5, "padj=1e-5")]:
        ax.scatter([], [], s=size_val * 30, c="gray", edgecolors="black", label=label_txt)
    ax.legend(loc="upper left", bbox_to_anchor=(1.35, 1), fontsize=8, title="bubble size")

    # 底部按最长label的字符数估算需要的边距，避免斜排文字被裁切或和图框重叠；
    # 右侧给图例留固定空间（legend 用 bbox_to_anchor 画在轴外）。
    max_label_len = max((len(l) for l in labels), default=10)
    bottom_margin = min(0.45, 0.12 + 0.012 * max_label_len)

    # 左边距同理按最长term标签的字符数动态估算，而不是用固定值——之前固定
    # left=0.32 在label列数少、图整体较窄时（比如只有2列supports/opposes对比）
    # 撑不住term_short最长45字符的文字，导致左侧term名称被裁切。这里换算成
    # "按字符数估算所需英寸宽度 / 图的总宽度"得到left比例，图越窄，需要的left
    # 比例就越大，保证term文字始终完整显示。
    fig_width_inch = fig.get_figwidth()
    max_term_len = max((len(t) for t in term_order), default=10)
    # 经验值：8pt字体下，一个字符大约需要0.075英寸宽度，再加一点余量
    required_left_inch = 0.075 * max_term_len + 0.15
    left_margin = min(0.55, max(0.28, required_left_inch / fig_width_inch))

    fig.subplots_adjust(bottom=bottom_margin, right=0.78, left=left_margin, top=0.92)

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path



def plot_usage_binned_trend(
    summary: pd.DataFrame,
    top_genes_binned: pd.DataFrame,
    program: str,
    spearman_r: float,
    spearman_p: float,
    program_direction: pd.DataFrame,
    out_dir: str,
    filename: str | None = None,   # 新增
) -> str:
    """usage分箱趋势图：主线=leading genes整体的每细胞平均wPSI(95% CI阴影)，
    细虚线=效应量最大的代表基因，标题对照compute_program_direction()判定的
    方向与这里算出的Spearman相关是否一致。

    Parameters
    ----------
    summary, top_genes_binned, spearman_r, spearman_p :
        compute_usage_binned_wpsi() 的返回值，原样传入。
    program_direction : DataFrame
        compute_program_direction() 的返回值，用来在标题里标注方向判断。
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, summary["wpsi_mean"], color="black", linewidth=2.5,
            label=f"{program} leading genes mean wPSI", zorder=3)
    ax.fill_between(x, summary["wpsi_ci_low"], summary["wpsi_ci_high"],
                     color="gray", alpha=0.25, zorder=2)

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, gene in enumerate(top_genes_binned.columns):
        ax.plot(x, top_genes_binned[gene].values, color=palette[i % len(palette)],
                linewidth=1, alpha=0.7, linestyle="--", label=gene, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary["usage_median"]], rotation=45, ha="right")
    ax.set_xlabel(f"{program} usage (bin median)")
    ax.set_ylabel("wPSI")

    direction_label = program_direction.loc[program, "direction"] if program in program_direction.index else ""
    ax.set_title(f"{program} (direction: {direction_label}) | "
                 f"usage-wPSI Spearman r={spearman_r:.2f}, p={spearman_p:.1e}")

    ax.legend(fontsize=8, loc="best", frameon=False)
    fig.tight_layout()

    if filename is None:
        filename = f"layer1_usage_binned_trend_{program}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_gene_celltype_heatmap(
    wpsi_matrix: pd.DataFrame,
    gene_program_map: pd.Series,
    gene_symbol_map: pd.Series,
    out_dir: str,
    row_zscore: bool = True,
    filename: str = "layer2_gene_celltype_heatmap.png",
) -> str:
    """gene x celltype(或其它分组) 的wPSI热图，行按program分组排列并用色条
    标注归属。默认对每个基因跨分组做z-score（更容易看出"相对哪些分组wPSI
    偏高/偏低"），传row_zscore=False展示原始wPSI均值。用纯matplotlib实现
    （和plot_jaccard_heatmap一致的风格，不引入seaborn依赖）。

    Parameters
    ----------
    wpsi_matrix, gene_program_map, gene_symbol_map :
        compute_gene_celltype_wpsi() 的返回值，原样传入。
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    programs = sorted(gene_program_map.unique())
    cmap_prog = plt.get_cmap("tab10")
    program_color = {p: cmap_prog(i % 10) for i, p in enumerate(programs)}

    order = gene_program_map.sort_values().index
    mat = wpsi_matrix.loc[order].copy()

    if row_zscore:
        mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
        cbar_label, vmin, vmax = "wPSI (row z-score)", -2, 2
    else:
        cbar_label, vmin, vmax = "mean wPSI", None, None

    row_labels = [gene_symbol_map.get(g, g) for g in mat.index]
    n_rows, n_cols = mat.shape

    fig, (ax_color, ax_main) = plt.subplots(
        1, 2, figsize=(max(6, 0.7 * n_cols) + 1.5, max(4, 0.28 * n_rows)),
        gridspec_kw={"width_ratios": [0.4, n_cols]},
    )
    ax_color.imshow([[program_color[gene_program_map.loc[g]]] for g in mat.index], aspect="auto")
    ax_color.set_xticks([]); ax_color.set_yticks([])

    im = ax_main.imshow(mat.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax_main.set_xticks(range(n_cols)); ax_main.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax_main.set_yticks(range(n_rows)); ax_main.set_yticklabels(row_labels, fontsize=8)
    ax_main.set_title("Leading gene wPSI by group (rows colored by program)")
    fig.colorbar(im, ax=ax_main, label=cbar_label, fraction=0.03, pad=0.02)

    handles = [Patch(facecolor=c, label=p) for p, c in program_color.items()]
    ax_main.legend(handles=handles, title="program", bbox_to_anchor=(1.3, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_within_celltype_comparison(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> str:
    """通用的两组对比箱线图（叠加抖动散点），供组内比较（wPSI或usage）复用，
    不重复写画图逻辑。

    Parameters
    ----------
    values_a, values_b : array-like
        两组的原始数值（如某基因wPSI，或某program usage）。
    out_path : str
        完整输出文件路径（含文件名），由调用方按wpsi/usage拼不同的文件名。
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bp = ax.boxplot([values_a, values_b], labels=[label_a, label_b],
                     showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(0)
    for i, vals in enumerate([values_a, values_b], start=1):
        jitter = rng.normal(i, 0.06, size=len(vals))
        ax.scatter(jitter, vals, s=4, alpha=0.3, color="black")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_within_celltype_effect_summary(
    effect_df: pd.DataFrame,
    program: str,
    celltype: str,
    out_dir: str,
) -> str:
    """画某个program的leading gene在细胞类型内部对照的效应量条形图，
    按supports/opposes/no_meaningful_diff三类着色，直观呈现
    "program内部基因是否同质地支持整体方向"这个问题。

    Parameters
    ----------
    effect_df : DataFrame
        summarize_leading_gene_within_celltype() 的返回值
        （index=symbol，需要 mean_diff 和 category 列）。
    """
    import matplotlib.pyplot as plt

    if effect_df is None or len(effect_df) == 0:
        print("  [Layer1] effect_df 为空，跳过画图")
        return ""

    os.makedirs(out_dir, exist_ok=True)
    df = effect_df.sort_values("mean_diff")

    color_map = {"supports": "#DD8452", "opposes": "#4C72B0", "no_meaningful_diff": "#B0B0B0"}
    colors = [color_map[c] for c in df["category"]]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(df))))
    ax.barh(df.index, df["mean_diff"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"mean wPSI diff (within {celltype})")
    ax.set_title(f"{program} leading genes — within-celltype effect\n"
                 f"orange=supports, blue=opposes, gray=no meaningful diff")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"layer1_within_celltype_effect_{program}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_driver_gene_heatmap(
    driver_genes: pd.DataFrame,
    wpsi_matrix: pd.DataFrame,
    gene_symbol_map: pd.Series,
    out_dir: str,
    program: str,
    row_zscore: bool = True,
    filename: str | None = None,
) -> str:
    """针对单个program，把其Top富集term的驱动基因（且同时是leading gene的
    那些）单独画一张聚焦热图，行按term分组着色、图例标term名，直接呈现
    "功能通路 -> 具体基因 -> 各细胞类型wPSI"这条完整链路。

    Parameters
    ----------
    driver_genes : DataFrame
        extract_enrichment_driver_genes() 的返回值（未过滤，函数内部按
        program筛选，且只保留in_leading_genes=True的行——不在leading gene
        里的基因没有gene_id，没法在wpsi_matrix里定位，会被跳过并打印提示）。
    wpsi_matrix, gene_symbol_map :
        layer2_association.compute_gene_celltype_wpsi() 返回的三元组里的
        第一个和第三个（这里不需要gene_program_map，本图行分组用term而不是
        program，因为这里只画单个program）。
    program : str
        要画哪个program。
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    sub = driver_genes[(driver_genes["program"] == program) & (driver_genes["in_leading_genes"])]
    if len(sub) == 0:
        print(f"  [{program}] 没有可用的驱动基因(都不在leading genes里，或该program没有enrichment结果)，跳过画图")
        return ""

    # 同一个基因可能出现在多个term里，保留第一次出现的term作为它的分组标签
    sub = sub.drop_duplicates(subset="gene_id", keep="first")
    gene_ids = [g for g in sub["gene_id"] if g in wpsi_matrix.index]
    n_missing = len(sub) - len(gene_ids)
    if n_missing:
        print(f"  [{program}] {n_missing}个驱动基因不在wpsi_matrix里"
              f"（可能compute_gene_celltype_wpsi用了不同的min_cells过滤），已跳过")
    if not gene_ids:
        print(f"  [{program}] 没有驱动基因能在wpsi_matrix里找到，跳过画图")
        return ""

    sub = sub.set_index("gene_id").loc[gene_ids]
    mat = wpsi_matrix.loc[gene_ids].copy()

    if row_zscore:
        mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
        cbar_label, vmin, vmax = "wPSI (row z-score)", -2, 2
    else:
        cbar_label, vmin, vmax = "mean wPSI", None, None

    terms = sub["term"].tolist()
    unique_terms = list(dict.fromkeys(terms))
    cmap_term = plt.get_cmap("Set2")
    term_color = {t: cmap_term(i % 8) for i, t in enumerate(unique_terms)}

    row_labels = [gene_symbol_map.get(g, g) for g in mat.index]
    n_rows, n_cols = mat.shape

    fig, (ax_color, ax_main) = plt.subplots(
        1, 2, figsize=(max(6, 0.7 * n_cols) + 2.5, max(3, 0.32 * n_rows)),
        gridspec_kw={"width_ratios": [0.4, n_cols]},
    )
    ax_color.imshow([[term_color[t]] for t in terms], aspect="auto")
    ax_color.set_xticks([]); ax_color.set_yticks([])

    im = ax_main.imshow(mat.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax_main.set_xticks(range(n_cols)); ax_main.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax_main.set_yticks(range(n_rows)); ax_main.set_yticklabels(row_labels, fontsize=8)
    ax_main.set_title(f"{program} — enrichment driver genes (rows colored by term)")
    fig.colorbar(im, ax=ax_main, label=cbar_label, fraction=0.03, pad=0.02)

    term_labels_short = {t: (t[:40] + "…" if len(t) > 40 else t) for t in unique_terms}
    handles = [Patch(facecolor=term_color[t], label=term_labels_short[t]) for t in unique_terms]
    ax_main.legend(handles=handles, title="term", bbox_to_anchor=(1.3, 1), loc="upper left", fontsize=7)

    fig.tight_layout()
    if filename is None:
        filename = f"layer3_driver_gene_heatmap_{program}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path