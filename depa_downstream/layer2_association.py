"""
Layer 2 — 疾病/条件差异分析：program usage 与任意 contrast 的关联度。

对 config["contrasts"] 里的每一条独立计算，函数本身不关心
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
) -> dict:
    """对 config["contrasts"] 里每一条（含自动展开的），计算各 program usage 与该分组的关联度。

    Parameters
    ----------
    data : dict
        需要 data["usage"] (cells x programs) 和 data["meta"]。
    config : dict
        用到 config["contrasts"]，以及可选的 min_cells_per_group（默认20）。
    Returns
    -------
    dict，key 是每条（展开后的）contrast 的 "name"，value 是一个 DataFrame：
        index=program，列：auc, lr_coef（逻辑回归系数，标准化后）,
        dt_importance（决策树特征重要性）, xgb_importance（若xgboost可用）,
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
