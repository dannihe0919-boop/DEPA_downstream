# DEPA downstream：Layer0–3 分析

从 cNMF 输出构建一张贯穿全流程的 program 基因表，再依次执行功能富集、usage 关联、群体方向检验和组内机制标注。最后将富集 term 与基因一致性交叉整合。原始数据不包含在仓库中。

```text
spectra / PAC 注释 ── Layer0 基因身份表 ── Layer1 富集 ──────────┐
                         │                                  │
usage + metadata ── Layer2 自动细胞类型汇总 ──┐                │
                         │                 │                │
Layer0 + wPSI ── Layer3a 群体方向 ── Layer3b 写回基因表 ── integrate_results
```

Layer1 与 Layer2 互不依赖。编号采用报告的叙事顺序；新流程不会让富集依赖方向检验，也不会把方向结果传入 usage 模型。

## 安装和运行

Python ≥3.10：

```bash
python -m pip install -e '.[notebook]'
# 可选：python -m pip install -e '.[xgboost]'
python -m depa_downstream.pipeline --config configs/covid19.yaml \
  --root-dir "$PWD" --output-root analysis_results/covid19
# 网络不可用时显式跳过富集：追加 --skip-enrichment
```

安装后也可使用 `depa-analyze`。`--root-dir` 只解析相对路径；配置中旧的 Mac 绝对路径需要用重复的 `--path KEY=PATH` 覆盖，例如：

```bash
depa-analyze --config configs/pre_t.yaml --output-root "$PWD/analysis_results/pre_t" \
  --path cnmf_dir=/data/pre_t/cNMF_output \
  --path h5ad_path=/data/pre_t/cNMF_output/PreT_DEAPA_counts.h5ad \
  --path feature_anno_path=/data/pre_t/cNMF_input/cnmf_feature_anno.csv \
  --path wpsi_matrix_path=/data/pre_t/PSI_results/movAPA_wPSI_matrix_df.rds \
  --path gene_symbol_map_path="$PWD/configs/pre_t_gene_symbol_map.csv"
```

两份主 Notebook 已改为新版顺序，配置只加载一次；Pre-T Notebook 通过一个 `DATA_ROOT` 覆盖旧路径。COVID Notebook 保留既有 P03 样本级补充分析，并改为直接读取 Layer0 核心基因。旧探索代码（不含旧单元输出）保存在 `notebooks/legacy/`；旧模块及导出函数继续保留作兼容，新流程不调用它们重新筛选基因。

Python 接口：

```python
from depa_downstream import load_cnmf_outputs
from depa_downstream.pipeline import load_analysis_config, run_analysis

config = load_analysis_config('configs/covid19.yaml', root_dir='.')
data = load_cnmf_outputs(config)
results = run_analysis(data, config, output_root='analysis_results/covid19')
gene_tables = results['gene_tables']  # {program: 最终 DataFrame}
```

分步接口见主 Notebook：`build_program_gene_tables` → `run_functional_enrichment` → `compute_contrast_association` / `summarize_program_associations` → `compute_population_direction` → `annotate_within_celltype` → `integrate_results`。`assemble_analysis_results` 将已计算的结果组织并整合，`export_analysis` 导出；两者不会重跑前面各层。

## 各层约定

| 层 | 输入 | 主要输出 |
| --- | --- | --- |
| 0 | spectra、PAC→gene 注释；可选 wPSI、symbol map | 每个 program 一张完整基因表 |
| 1 | Layer0 表的 `is_core` / `is_specific` | core / specific 两套富集表 |
| 2 | usage（内部为 **cells×programs**）、metadata、contrasts | AUC/模型指标和 program→best_celltype 汇总 |
| 3a | Layer0 核心标志、缓存的可靠性与 wPSI 中位数 | direction、pval、padj、expected_direction |
| 3b | Layer0、Layer2 汇总、3a、wPSI 与组内 usage | 同一张基因表上的组内效应及 concordance |
| 整合 | Layer1 + 最终 Layer0 | 每个 term 的一致性计数、symbol 和 gene ID |

**Layer0**：同一基因多个 PAC 的 `spectra_score` 和 `spectra_z` 按最大值聚合，与旧程序的核心定义一致；按 Z 降序、gene ID 同分排序生成唯一排名。`is_core` 使用 **Z ≥ 1.5**；`is_specific` 额外要求其余 program 的最大 gene Z **< 0.5**。检测率是非缺失 wPSI 中落在开区间 (0,1) 的比例，默认 ≥0.1。低可靠性基因仍保留在核心和富集列表中；缺少整个 wPSI 输入时可靠性标记为未知。`gene_wpsi_median`、检测率和有效细胞数只计算一次。

**Layer1**：背景从实际进入 spectra 且有 PAC 注释的基因动态生成，不写死 393。核心、专属列表及背景用相同的 symbol 映射和去重规则；未映射 ENSG 数量明确记录。通过 [Speedrichr 自定义背景协议](https://github.com/zqfang/GSEApy/blob/master/gseapy/enrichr.py)上传背景，一次运行复用同一个背景 ID；绝不回退到全基因组背景。默认 GO BP、KEGG、Hallmark、Reactome，支持配置覆盖。可用 symbol 数可能少于原始 gene ID 数，输出分别记录两种背景大小。

富集结果没有方向列。网络/协议失败是 `failed`，有效查询但无条目是 `empty`，基因过少是 `skipped_insufficient_genes`，离线模式是 `disabled`。`run_functional_enrichment` 默认遇错抛出；完整 pipeline 记录服务错误并完成其他独立分析，在 manifest 中标为 `partial`，CLI 返回 2。不要把无显著通路自动当作 cNMF 分解失败：数据库覆盖、背景大小和基因列表长度同样会影响结果。

**Layer2**：疾病维度保留配置中该 pairwise 对比的 AUC。细胞身份维度按 **|AUC−0.5| 最大**选择，同分按细胞类型标签排序。AUC<0.5 的类型同样会被选中，并标为 `depleted`；`best_celltype_n_cells` 是该类型正组的有效细胞数。缺失身份不归入 rest。多个 disease 对比时必须通过 `disease_contrast` 指定使用哪一个，不默认取首条。

**Layer3a 背景**：默认沿用旧检验的统计口径——在可靠 PAC universe 内**剔除当前前景**，使 MWU 两组不重叠；前景和背景都使用可用且可靠的基因。这是对方案中“pac_universe 全集作为背景”的具体实现约定。若确实需要包含前景的全集比较，可设 `direction_background_mode: full_universe`；输出记录所用模式，重叠的比较不具有独立两样本的解释。背景不再扩展到 PAC universe 外的全基因组。前景和背景默认都至少 5 个基因才检验，BH 在本次检验的 program 间校正。

`direction` 是前景相对背景的描述性方向；`is_significant` 单独记录 BH 显著性，不能把不显著的符号当作已确立机制。`expected_direction` 固定指 **usage 升高时** wPSI 应 `increase`（Distal）还是 `decrease`（Proximal）；零偏移或未检验为 `unknown`。

**Layer3b 符号**：`within_celltype_mean_diff` 始终是疾病 **B−A**。一致性比较使用的组间预期符号为：

```text
sign(wPSI 随 usage 升高的预期变化) × sign(该细胞类型内 mean usage(B) − mean usage(A))
```

这避免了把疾病组 usage 更低的 program 标反。原始 expected_direction、组间 expected_direction 和组内 usage 差值都保留。组内 usage 无变化或缺失时不推断正负一致性。全部核心基因都会得到状态标注，不再先按方向挑一个 leading-gene 子集。边界 0/1 保留用于非零比例和均值统计。

默认每组至少 20 个有效细胞；`min_samples_per_group`（若配置）还检查有效生物学样本数。均值差绝对值 ≥0.01 且 p<0.05 时判 concordant/discordant，否则为 inconclusive。基因不匹配、样本不足或缺少上游输入为 `not_tested`，具体原因在 `within_celltype_status`，不会混入 inconclusive。检出率、非零比例、均值、有效细胞/样本数、原始 p 和每个 program 内的 BH padj 一并写回。`within_significance: padj` 可改用 padj 判定；默认 `p` 与方案一致。

组内 MWU 是**细胞级探索性检验**；检查 donor 数不等于以 donor 为统计单位。生物学结论仍需样本级分析确认。保留的 P03 补充部分就是独立的样本级分析，不把细胞级 p 值冒充样本级证据。

**整合**：term 命中按去重后的 gene ID 计数。输出 `n_concordant`、`n_discordant`、`n_inconclusive` 及相应 symbol/ID 列；`n_unassessed` 单列未检验项。无法匹配的命中 symbol 以及一个 symbol 对应多个 ID 的歧义数量也单独记录。

## 输出与测试

```text
analysis_results/<dataset>/
  layer0/<program>.csv                      # 最终基因表，含 3b 新增列
  layer1/{core,specific}/<program>.csv       # 不含方向信息的原始富集
  layer2/contrasts/<contrast>.csv
  layer2/program_association_summary.csv
  layer3/population_direction.csv
  integrated/{core,specific}/<program>.csv  # 富集 × 一致性
  analysis_manifest.json                   # 参数、背景大小、跳过/失败状态
```

Notebook 还在各层目录中生成图。`status: completed` 表示调度完成；是否跳过方向或富集等步骤，必须查看 manifest 中各层状态。

```bash
python -m unittest discover -s tests -v
```

测试使用确定性合成数据和模拟 API，覆盖阈值边界、PAC 聚合、检测率保留、显式背景协议、AUC 正负与同分、usage 反向时的一致性、样本不足、缺失输入、结果整合和导出。真实数据未提供，真实数据的全量结论不属于这些测试的验证范围。
