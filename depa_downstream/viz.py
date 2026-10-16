"""Plots for final Layer 0–3 results; no statistical tests or gene screening."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

CATEGORY_ORDER = ["disease", "identity", "state"]
CATEGORY_COLORS = {"disease": "#C44E52", "identity": "#4C72B0", "state": "#55A868"}


def plot_enrichment_bubble(
        enrichment_by_label: dict,
        out_dir: str,
        top_n: int = 10,
        padj_threshold: float = 0.05,
        filename: str = "layer1_enrichment_bubble.png",
) -> str:
    """富集结果气泡图：每个label取padj最小的前top_n条term，跨label并排展示。

    气泡大小 = -log10(padj)，气泡颜色 = combined_score，横轴是label，
    纵轴是term（按label分组，组内按显著性排序）。

    Parameters
    ----------
    enrichment_by_label : dict
        按 core 或 specific 取出的 Layer1 结果（{program: DataFrame}）。
    out_dir : str
        保存目录。
    top_n : int
        每个label最多展示多少条term。
    padj_threshold : float
        只展示padj低于此阈值的term；某个label如果一条都不满足，会被跳过
        （不在图上占位），并打印提示。
    filename : str
        输出图片文件名（不含目录）。默认 "layer1_enrichment_bubble.png"。
        同一批分析里如果要对不同基因列表（比如"专属marker" vs "全部核心基因"）
        各画一张图，传不同的 filename 避免互相覆盖。
    """
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in enrichment_by_label.items() if v is not None and len(v) > 0}
    if not valid:
        print("  [Layer1] 没有任何富集结果，跳过画图")
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
        print(f"  [Layer1] 所有label都没有padj<{padj_threshold}的term，跳过画图")
        return ""

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values(["label", "padj"])
    plot_df["term_short"] = plot_df["term"].str.slice(0, 45)
    term_order = list(dict.fromkeys(plot_df["term_short"].tolist()))
    labels = list(dict.fromkeys(plot_df["label"]))

    plot_df["neglog10_padj"] = -np.log10(plot_df["padj"].clip(lower=1e-300))
    # Custom small backgrounds can legitimately yield infinite Enrichr
    # combined scores. Matplotlib otherwise masks those hits entirely.
    scores = pd.to_numeric(plot_df['combined_score'], errors='coerce')
    finite = scores[np.isfinite(scores)]
    upper = max(1.0, float(finite.max())) if len(finite) else 1.0
    lower = min(0.0, float(finite.min())) if len(finite) else 0.0
    display_scores = scores.replace({np.inf: upper, -np.inf: lower}).fillna(lower)
    y_pos = {t: i for i, t in enumerate(term_order)}
    x_pos = {l: i for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), max(3, 0.35 * len(term_order))))
    sc = ax.scatter(
        [x_pos[l] for l in plot_df["label"]],
        [y_pos[t] for t in plot_df["term_short"]],
        s=plot_df["neglog10_padj"] * 30,
        c=display_scores,
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
    if not np.isfinite(scores).all():
        ax.set_title(ax.get_title() + '\n(non-finite scores shown at color limits)')
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



def plot_association_heatmap(assoc_results: dict, config: dict, out_dir: str) -> str:
    """画所有 contrast x program 的 AUC 汇总热图，按 category 分组展示。

    列先按 category（disease -> identity -> state -> 其他，未知类别追加在
    最后）分组，组内按名字排序；组间画分隔线；顶部加一条颜色条标注每列
    属于哪个 category，并附图例。
    """
    valid = {k: v for k, v in assoc_results.items() if v is not None}
    if not valid:
        print("  [Layer2] 没有任何 contrast 有有效结果，跳过画图")
        return ""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)
    programs = sorted(next(iter(valid.values())).index.tolist())

    # 按 category 分组排序列
    cat_of = {name: df.attrs.get("category", "other") for name, df in valid.items()}
    all_cats = list(dict.fromkeys(CATEGORY_ORDER + sorted(set(cat_of.values()) - set(CATEGORY_ORDER))))
    ordered_names = sorted(valid.keys(), key=lambda n: (all_cats.index(cat_of[n]), n))

    mat = pd.DataFrame({name: valid[name]["auc"] for name in ordered_names}).reindex(programs)
    col_categories = [cat_of[name] for name in ordered_names]

    # 给每个出现过的 category 分配颜色（预设的用预设色，其余自动分配）
    cmap_fallback = plt.get_cmap("Set2")
    color_lookup = dict(CATEGORY_COLORS)
    for i, c in enumerate(all_cats):
        if c not in color_lookup:
            color_lookup[c] = cmap_fallback(i % 8)

    n_cols = len(ordered_names)
    fig, (strip_ax, ax) = plt.subplots(
        2, 1, figsize=(max(5, 0.55 * n_cols), max(3, 0.5 * len(programs)) + 0.6),
        gridspec_kw={"height_ratios": [0.4, max(3, 0.5 * len(programs))], "hspace": 0.05},
        sharex=True,
    )

    # ── 顶部 category 颜色条 ──
    cat_codes = [all_cats.index(c) for c in col_categories]
    used_cats = [c for c in all_cats if c in set(col_categories)]
    strip_cmap = ListedColormap([color_lookup[c] for c in used_cats])
    code_remap = {all_cats.index(c): i for i, c in enumerate(used_cats)}
    strip_vals = np.array([[code_remap[code] for code in cat_codes]])
    strip_ax.imshow(strip_vals, aspect="auto", cmap=strip_cmap,
                     vmin=-0.5, vmax=len(used_cats) - 0.5)
    strip_ax.set_yticks([])
    strip_ax.set_xticks([])
    for spine in strip_ax.spines.values():
        spine.set_visible(False)
    legend_handles = [Patch(facecolor=color_lookup[c], label=c) for c in used_cats]
    strip_ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0, 1.05),
                     ncol=len(used_cats), fontsize=8, frameon=False)

    # ── 主热图 ──
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(programs)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if abs(val - 0.5) > 0.25 else "black", fontsize=7)

    # 组间分隔线
    boundaries = []
    for i in range(1, n_cols):
        if col_categories[i] != col_categories[i - 1]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        ax.axvline(b, color="black", linewidth=1.2)

    fig.suptitle("Program usage ~ contrast association (AUC)", y=0.98)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.32, top=0.88)
    fig.colorbar(im, ax=[strip_ax, ax], label="AUC (0.5 = no association)", fraction=0.025, pad=0.02)

    out_path = os.path.join(out_dir, "layer2_association_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path



def plot_program_umap_grid(data: dict, config: dict, out_dir: str) -> str:
    """全 program 的 UMAP 网格图（每个 program 一张小图，按 usage 染色）。

    第一张图固定是 cell_type 分类着色（若 config['meta_columns'] 里配置了
    celltype列），作为参照，方便直接对照某个program的高usage区域落在哪个
    细胞类型上。每个类型簇的中心位置直接标注类型名文字（白色描边保证在
    深色点上也清晰可读），不需要额外看图例对照颜色。

    需要 config["meta_columns"] 里配置了 umap1/umap2，没配置则跳过并提示。
    """
    meta_cols = config["meta_columns"]
    if "umap1" not in meta_cols or "umap2" not in meta_cols:
        print("  [Layer2] 未配置 UMAP 坐标列，跳过 UMAP 网格图")
        return ""

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    os.makedirs(out_dir, exist_ok=True)
    usage = data["usage"]
    meta = data["meta"]
    programs = data["programs"]
    common_idx = usage.index.intersection(meta.index)

    x = meta.loc[common_idx, meta_cols["umap1"]].values
    y = meta.loc[common_idx, meta_cols["umap2"]].values

    celltype_col = meta_cols.get("celltype")
    has_celltype = celltype_col is not None and celltype_col in meta.columns

    panels = (["__celltype__"] if has_celltype else []) + list(programs)
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if panel == "__celltype__":
            cats = meta.loc[common_idx, celltype_col].astype(str)
            uniq = sorted(cats.unique())
            cmap = plt.get_cmap("tab20", max(len(uniq), 1))
            color_map = {c: cmap(i) for i, c in enumerate(uniq)}
            point_colors = cats.map(color_map).tolist()
            ax.scatter(x, y, c=point_colors, s=3, linewidths=0)
            ax.set_title("cell_type (reference)")
            # 每个类型簇中心直接标文字，不用图例
            for c in uniq:
                mask = cats.values == c
                xm, ym = np.median(x[mask]), np.median(y[mask])
                ax.text(xm, ym, c, fontsize=8, fontweight="bold", ha="center", va="center",
                         path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        else:
            c = usage.loc[common_idx, panel].values
            sc = ax.scatter(x, y, c=c, cmap="viridis", s=3, linewidths=0)
            ax.set_title(panel)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Program usage on UMAP")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, "layer2_program_umap_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path



def plot_population_direction(
    population_direction: pd.DataFrame,
    out_dir: str,
    padj_threshold: float = 0.05,
    pval_threshold: float = 0.05,
    filename: str = "layer3_population_direction.png",
) -> str:
    """画方向检验的汇总条形图（delta_median，按方向着色，标注显著性）。

    传入的是 compute_population_direction() 的结果，不重新计算。
    样本量小、检验普遍不显著时（比如COVID这批数据），仍然按 delta_median
    大小排序展示，用星号区分"padj显著"/"仅raw pval显著"/"不显著"三档，
    不因为没有显著结果就不出图——描述性的方向倾向本身也是有信息量的。

    Parameters
    ----------
    population_direction : DataFrame
        compute_population_direction() 的返回值。
    out_dir : str
        图片保存目录（自动创建）。
    padj_threshold, pval_threshold : float
        显著性星号标注阈值。

    Returns
    -------
    str: 保存的图片路径。若 population_direction 为 None，打印提示并返回空字符串。
    """
    if population_direction is None:
        print("  [Layer3] population_direction 为 None，跳过画图")
        return ""

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    df = population_direction.copy()
    df = df[df["direction"] != "NA"].sort_values("delta_median")
    if len(df) == 0:
        print("  [Layer3] 没有任何 program 完成检验，跳过画图")
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



def plot_within_celltype_concordance(gene_tables: dict, program: str, out_dir: str) -> str:
    """Plot Layer 3b core-gene effects directly from the final Layer 0 table."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from urllib.parse import quote

    table = gene_tables[program]
    selected = table.loc[table['is_core'] & table['within_celltype_mean_diff'].notna()].copy()
    if selected.empty:
        return ''
    selected = selected.sort_values('within_celltype_mean_diff')
    colors = {'concordant': '#55A868', 'discordant': '#C44E52', 'inconclusive': '#999999', 'not_tested': '#DDDDDD'}
    fig, ax = plt.subplots(figsize=(7, max(3, len(selected) * .3)))
    ax.barh(range(len(selected)), selected['within_celltype_mean_diff'],
            color=[colors[label] for label in selected['concordance']])
    ax.set_yticks(range(len(selected)), labels=selected['symbol'])
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('Mean wPSI difference (disease group B − group A)')
    celltype = selected['within_celltype'].iloc[0]
    expected = selected['within_celltype_expected_direction'].iloc[0]
    ax.set_title(f'{program} within {celltype}; expected B−A: {expected}')
    ax.legend(handles=[Patch(facecolor=color, label=label) for label, color in colors.items()], fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'layer3_within_celltype_{quote(str(program), safe="")}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
