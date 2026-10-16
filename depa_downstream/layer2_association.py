"""
Layer 2 — 疾病/条件差异分析：program usage 与任意 contrast 的关联度。

对应原 notebook 的 Cell 3，但原版写死了"疾病 AUC"一个二分类。
这里改成对 config["contrasts"] 里的每一条独立跑一遍，函数本身不关心
某条 contrast 在语义上是"疾病 vs 健康"还是"细胞身份 vs 其余"——
全部当成"某一列的某种分组方式"来处理，保证跨数据集复用。

每条 contrast 用两类指标衡量 program usage 和分组的关联：
    - AUC（单变量）：仅用该 program 自己的 usage 值预测分组标签，
      衡量这个 program 单独看有多大区分力。
    - 逻辑回归系数 / 决策树重要性 / XGBoost重要性（多变量，可选）：
      把所有 program 的 usage 一起作为特征拟合分类器，衡量该 program
      在"控制了其他program"之后还剩多少独立贡献。

contrasts 支持三种 type（见 config_schema.py 文档），其中
"auto_one_vs_rest_all" 会在本模块内自动展开成多条 one_vs_rest，
不需要在 CONFIG 里手动列举某一列的每个取值。

每条 contrast 可以带一个可选的 "category" 字段（自由字符串，比如
"disease"/"identity"/"state"），用于画热图时分组展示。没写的话按
type 自动推断默认值：one_vs_rest/auto_one_vs_rest_all -> "identity"，
pairwise -> "state"。auto展开出的子条目会继承父条目的 category。
"""

from __future__ import annotations

import os

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


DEFAULT_CATEGORY_BY_TYPE = {
    "one_vs_rest": "identity",
    "auto_one_vs_rest_all": "identity",
    "pairwise": "state",
}

# 画图时类别的展示顺序 + 配色，不在这里列出的类别会被追加到末尾并自动分配颜色
CATEGORY_ORDER = ["disease", "identity", "state"]
CATEGORY_COLORS = {
    "disease": "#C44E52",
    "identity": "#4C72B0",
    "state": "#55A868",
}


def _resolve_category(ctr: dict) -> str:
    return ctr.get("category") or DEFAULT_CATEGORY_BY_TYPE.get(ctr["type"], "other")


def _expand_contrasts(contrasts: list[dict], meta: pd.DataFrame) -> list[dict]:
    """把 type="auto_one_vs_rest_all" 的条目展开成多条 one_vs_rest。

    展开后的每条名字直接用该分组的取值本身（如 "Platelet"），不再加
    "_vs_rest" 后缀——category信息（默认"identity"）已经能表明这是
    "该类型 vs 其余"的对比，不需要在名字里重复说明。
    category 会从父条目继承（父条目没写就用 one_vs_rest 的默认值 "identity"）。

    其余类型（pairwise / one_vs_rest）原样保留，只是补全 category 字段。
    """
    expanded = []
    for ctr in contrasts:
        category = _resolve_category(ctr)
        if ctr["type"] != "auto_one_vs_rest_all":
            expanded.append({**ctr, "category": category})
            continue
        col = ctr["column"]
        values = sorted(meta[col].dropna().unique().tolist())
        print(f"  contrast '{ctr['name']}' (auto_one_vs_rest_all, column={col!r}) "
              f"自动展开为 {len(values)} 条 [category={category}]: {values}")
        for v in values:
            expanded.append({
                "name": str(v),
                "column": col,
                "type": "one_vs_rest",
                "group": v,
                "category": category,
            })
    return expanded


def _build_label(meta: pd.DataFrame, contrast: dict) -> pd.Series:
    """根据一条 contrast 配置，从 meta 里构建 0/1 标签（Series，index=cell）。

    pairwise   : 只保留 group_a/group_b 两组细胞，其余细胞不参与
                 (label为1代表group_b，0代表group_a)
    one_vs_rest: group 组记为1，其余全部记为0
    """
    col = contrast["column"]
    if contrast["type"] == "pairwise":
        mask = meta[col].isin([contrast["group_a"], contrast["group_b"]])
        label = (meta.loc[mask, col] == contrast["group_b"]).astype(int)
        return label
    elif contrast["type"] == "one_vs_rest":
        # Unknown identities are not a biological "rest" group.
        label = (meta.loc[meta[col].notna(), col] == contrast["group"]).astype(int)
        return label
    else:
        raise ValueError(f"未知 contrast type: {contrast['type']}（auto_one_vs_rest_all 应已在此之前展开）")


def compute_contrast_association(
    data: dict,
    config: dict,
    program_direction: pd.DataFrame | None = None,
) -> dict:
    """对 config["contrasts"] 里每一条（含自动展开的），计算各 program usage 与该分组的关联度。

    Parameters
    ----------
    data : dict
        需要 data["usage"] (cells x programs) 和 data["meta"]。
    config : dict
        用到 config["contrasts"]，以及可选的 min_cells_per_group（默认20）。
    program_direction : DataFrame, optional
        Layer1 的输出，目前仅在结果里附带一列 direction 方便一起看，
        不参与关联度计算本身。可以为 None。

    Returns
    -------
    dict，key 是每条（展开后的）contrast 的 "name"，value 是一个 DataFrame：
        index=program，列：auc, lr_coef（逻辑回归系数，标准化后）,
        dt_importance（决策树特征重要性）, xgb_importance（若xgboost可用）,
        direction（若提供了program_direction）,
        n_group_a, n_group_b（各组细胞数）
        每个 DataFrame 附带 .attrs["category"]（该contrast的类别标签，
        画热图分组用）。
        若该条contrast因样本量不足被跳过，value 为 None。
    """
    usage = data["usage"]
    meta = data["meta"]
    programs = list(usage.columns)
    min_cells = config.get("min_cells_per_group", 20)

    contrasts = _expand_contrasts(config["contrasts"], meta)

    names_seen = [c["name"] for c in contrasts]
    dup_names = sorted({n for n in names_seen if names_seen.count(n) > 1})
    if dup_names:
        raise ValueError(
            f"contrasts展开后出现重名: {dup_names}（可能是手写的contrast名和"
            f"auto_one_vs_rest_all自动展开出的取值撞名），结果会互相覆盖，"
            f"请改名或检查配置"
        )

    results = {}
    for contrast in contrasts:
        name = contrast["name"]
        category = contrast["category"]
        print(f"\n=== contrast '{name}' [{category}] (column={contrast['column']!r}, type={contrast['type']!r}) ===")

        label = _build_label(meta, contrast)
        common_idx = usage.index.intersection(label.index)
        label = label.loc[common_idx]
        X_all = usage.loc[common_idx, programs]

        n_a = int((label == 0).sum())
        n_b = int((label == 1).sum())
        group_a_name = contrast.get("group_a", f"not_{contrast.get('group', '')}")
        group_b_name = contrast.get("group_b", contrast.get("group"))
        print(f"  {group_a_name}: {n_a} 细胞  |  {group_b_name}: {n_b} 细胞")

        if n_a < min_cells or n_b < min_cells:
            print(f"  ⚠ 某一组细胞数 < min_cells_per_group={min_cells}，跳过该 contrast")
            results[name] = None
            continue

        min_samples = config.get("min_samples_per_group")
        if min_samples is not None:
            sample_col = config["meta_columns"].get("sample")
            if sample_col is not None and sample_col in meta.columns:
                n_samples_a = meta.loc[label.index[label == 0], sample_col].nunique()
                n_samples_b = meta.loc[label.index[label == 1], sample_col].nunique()
                if n_samples_a < min_samples or n_samples_b < min_samples:
                    print(f"  ⚠ 某一组涉及的样本数({n_samples_a}/{n_samples_b}) < "
                          f"min_samples_per_group={min_samples}，可能是个体效应冒充组间差异，跳过该 contrast")
                    results[name] = None
                    continue

        y = label.values
        X = X_all.values

        # ── 单变量 AUC（每个program自己） ──────────────────────
        aucs = {}
        for i, prog in enumerate(programs):
            try:
                aucs[prog] = roc_auc_score(y, X[:, i])
            except ValueError:
                aucs[prog] = np.nan

        # ── 多变量：逻辑回归系数（标准化后，可比较大小） ─────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_scaled, y)
        lr_coefs = dict(zip(programs, lr.coef_[0]))

        # ── 多变量：决策树特征重要性 ─────────────────────────────
        dt = DecisionTreeClassifier(max_depth=3, random_state=0)
        dt.fit(X, y)
        dt_importances = dict(zip(programs, dt.feature_importances_))

        # ── 多变量：XGBoost特征重要性（可选，装了才跑） ──────────
        if _HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=100, max_depth=3, eval_metric="logloss", random_state=0,
                n_jobs=config.get("xgb_n_jobs", 2),
            )
            xgb.fit(X, y)
            xgb_importances = dict(zip(programs, xgb.feature_importances_))
        else:
            xgb_importances = {p: np.nan for p in programs}

        rows = []
        for prog in programs:
            row = {
                "program": prog,
                "auc": aucs[prog],
                "lr_coef": lr_coefs[prog],
                "dt_importance": dt_importances[prog],
                "xgb_importance": xgb_importances[prog],
                "n_group_a": n_a,
                "n_group_b": n_b,
            }
            if program_direction is not None and prog in program_direction.index:
                row["direction"] = program_direction.loc[prog, "direction"]
            rows.append(row)

        result_df = pd.DataFrame(rows).set_index("program")
        result_df = result_df.reindex(result_df["auc"].sub(0.5).abs().sort_values(ascending=False).index)
        result_df.attrs["category"] = category
        result_df.attrs["contrast"] = dict(contrast)
        top_line = result_df["auc"].sub(0.5).abs().sort_values(ascending=False).head(1)
        print(f"  最强关联: {top_line.index[0]} (|AUC-0.5|={top_line.iloc[0]:.3f})")

        results[name] = result_df

    return results


def resolve_disease_contrast(config: dict) -> dict | None:
    """Select an explicit disease contrast; refuse ambiguous multiple diseases."""
    requested = config.get('disease_contrast')
    candidates = [c for c in config['contrasts'] if c['type'] == 'pairwise'
                  and (c['name'] == requested if requested else _resolve_category(c) == 'disease')]
    if requested and not candidates:
        raise ValueError(f'disease_contrast {requested!r} is not a configured pairwise contrast')
    if len(candidates) > 1:
        raise ValueError('Multiple disease contrasts: set disease_contrast to a contrast name')
    return candidates[0] if candidates else None


def summarize_program_associations(assoc_results: dict, config: dict,
                                   programs: list | None = None) -> pd.DataFrame:
    """Select max |AUC-.5| cell identity per program, including depleted types.

    Count the positive (identity) group, not its complement. Ties are resolved
    by cell-type label for deterministic results. Only tested, finite AUCs
    participate; no cell type is inferred from an untested contrast.
    """
    disease = resolve_disease_contrast(config)
    if programs is None:
        programs = list(dict.fromkeys(p for result in assoc_results.values()
                                     if result is not None for p in result.index))
    celltype_column = config['meta_columns']['celltype']
    rows = []
    for program in programs:
        row = dict(program=program, disease_contrast=disease['name'] if disease else None,
                   disease_auc=np.nan, best_celltype=None, best_celltype_auc=np.nan,
                   best_celltype_n_cells=0, best_celltype_association=None,
                   association_status='no_valid_identity')
        disease_result = assoc_results.get(disease['name']) if disease else None
        if disease_result is not None and program in disease_result.index:
            row['disease_auc'] = disease_result.loc[program, 'auc']
        candidates = []
        for name, result in assoc_results.items():
            if result is None or program not in result.index:
                continue
            contrast = result.attrs.get('contrast', {})
            if contrast.get('column') != celltype_column or contrast.get('type') != 'one_vs_rest':
                continue
            auc = result.loc[program, 'auc']
            if pd.notna(auc) and np.isfinite(auc):
                candidates.append((abs(float(auc) - 0.5), str(contrast['group']), contrast['group'], result.loc[program]))
        if candidates:
            _, _, label, selected = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            row.update(best_celltype=label, best_celltype_auc=float(selected['auc']),
                       best_celltype_n_cells=int(selected['n_group_b']),
                       best_celltype_association=('enriched' if selected['auc'] > 0.5 else
                                                  'depleted' if selected['auc'] < 0.5 else 'neutral'),
                       association_status='ok')
        rows.append(row)
    return pd.DataFrame(rows).set_index('program')


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



# ────────────────────────────────────────────────────────────────
# 把program-contrast关联下钻到具体基因：gene x celltype wPSI
# ────────────────────────────────────────────────────────────────

def compute_gene_celltype_wpsi(
    data: dict,
    config: dict,
    leading_genes: dict,
    groupby_col: str | None = None,
    min_cells: int = 10,
):
    """把所有program的leading genes合并，计算每个基因在每个细胞类型(或其它
    meta分组)里的wPSI均值，用来把compute_contrast_association()给出的
    program-contrast关联强度下钻到"具体是哪些基因、在哪些细胞类型里的差异"。

    用mean而不是median聚合，呼应layer1_direction里
    _compare_gene_within_celltype_effect_size 的做法：单细胞wPSI零膨胀严重，
    中位数容易被钉死在0/1边界，均值更能反映组间的真实差异。

    Parameters
    ----------
    leading_genes : dict
        layer1_direction.compute_leading_genes() 的返回值。
    groupby_col : str, optional
        meta里的分组列名，默认取 config["meta_columns"]["celltype"]；
        传疾病分组那一列可以看program在疾病状态间的基因级差异。
    min_cells : int
        分组细胞数低于这个阈值的组跳过，避免小样本均值不稳。

    Returns
    -------
    (wpsi_matrix, gene_program_map, gene_symbol_map) 或 None
        wpsi_matrix: DataFrame index=gene_id, columns=分组取值
        gene_program_map: Series index=gene_id, value=所属program
        gene_symbol_map: Series index=gene_id, value=symbol
    """
    wpsi = data.get("wpsi")
    if wpsi is None:
        print("  [Layer2] 未提供 wPSI 矩阵，跳过 gene x celltype 计算")
        return None

    meta = data["meta"]
    if groupby_col is None:
        groupby_col = config["meta_columns"]["celltype"]

    gene_program_map, id_to_symbol = {}, {}
    for p, df in leading_genes.items():
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            gene_program_map.setdefault(row["gene_id"], p)
            id_to_symbol.setdefault(row["gene_id"], row["symbol"])

    if not gene_program_map:
        print("  [Layer2] 所有program都没有leading genes，跳过")
        return None

    gene_ids = pd.Index(gene_program_map.keys()).intersection(wpsi.index)
    n_missing = len(gene_program_map) - len(gene_ids)
    if n_missing:
        print(f"  [Layer2] {n_missing} 个leading gene不在wPSI矩阵里，已跳过")

    group_sizes = meta[groupby_col].value_counts()
    valid_groups = group_sizes[group_sizes >= min_cells].index.tolist()
    dropped = set(group_sizes.index) - set(valid_groups)
    if dropped:
        print(f"  [Layer2] 以下分组细胞数<{min_cells}，已跳过: {sorted(dropped)}")

    common_cells = wpsi.columns.intersection(meta.index)
    records = {}
    for gid in gene_ids:
        row = wpsi.loc[gid, common_cells]
        grouped_mean = row.groupby(meta.loc[common_cells, groupby_col]).mean()
        records[gid] = grouped_mean.reindex(valid_groups)

    wpsi_matrix = pd.DataFrame(records).T
    wpsi_matrix.index.name = "gene_id"
    gene_program_map = pd.Series(gene_program_map).reindex(wpsi_matrix.index)
    gene_symbol_map = pd.Series(id_to_symbol).reindex(wpsi_matrix.index)

    print(f"  gene x {groupby_col} wPSI矩阵: {wpsi_matrix.shape[0]} 个基因 x {wpsi_matrix.shape[1]} 个分组")
    return wpsi_matrix, gene_program_map, gene_symbol_map
