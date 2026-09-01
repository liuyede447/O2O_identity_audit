# Goal 模式新增工作与全部结果总报告

日期：2026-09-01  
当前唯一论文数值权威：`evidence_freeze/final_evidence_v2`  
Final ledger：24 条 evidence、5 条直接排除记录、pending=0、16/16 final-freeze validation PASS  
Ledger SHA-256：`e2f504f2e048f842d26799048e9e6c8305000a662745a83ba611fc0b393d5e78`

## 1. 范围与最重要的总判断

`RUN_REGISTRY.csv` 共 53 条记录：9 条是为了完整对账而登记的既有结果，44 条是在本次 Goal 执行期间新建的 run。44 条 Goal 新增 run 的终态为：

- complete：33；
- failed：5；
- invalidated：5；
- incomplete：1。

33 条 complete 又分为：14 条 evidence-eligible、14 条 governance/upstream-only、5 条 smoke-only。

本轮没有启动新的 100/300 epoch 模型训练，也没有因结果不好重新训练或重新抽样。新增工作是只读 replay、CPU 统计分析、instrument qualification、连续边界扫描、既有 checkpoint 轨迹复放、冻结/预注册/lockbox 以及证据治理。`learning dynamics` 使用的是已经物理留存的 6 个 checkpoint；YOLOv10-S 结果是既有兼容 detector contract 的只读 replication，不是新训练。

当前 `main.tex` 对 final evidence v2 的覆盖情况：

- 已有且实质准确：5/24；
- 已出现但版本过时或不完整：2/24；
- 仅 Supplement 已有而主文没有：0/24；
- 完全缺失：17/24。

因此，以下结果不是“可选补充”，而是当前论文尚未吸收的新证据。正结果、零结果、负结果、失败和 invalidation 均在本报告中保留。

## 2. 测量工具与实现验证

| Evidence ID | 分母/范围 | 结果 | 结论边界 | 当前论文 |
|---|---:|---|---|---|
| `EV-INSTR-ORACLE-V1` | 6 oracle cases；7 property invariants | 6/6、7/7 PASS | 证明测试工具覆盖预设 native-path 情形，不是模型效应 | 完全缺失 |
| `EV-INSTR-MUTATION-V1` | 6 个预设 mutation | 6/6 被检测并杀死 | positive-control sensitivity，不是科学效应 | 完全缺失 |
| `EV-INSTR-PARITY-975-V1` | 300 base + 975 directional states | Native/production/NumPy mismatch=0 | 较窄三路 parity | 完全缺失 |
| `EV-INSTR-PARITY-5251-V1` | 300 base + 5,251 directional states | mismatch=0 | 更广 deterministic parity | 完全缺失 |
| `EV-BOUNDARY-REFERENCE-FORMAL-V2` | 30 图；30 forwards；4,408 production/reference trajectories | 4,408/4,408；mismatch=0；最大 radius 差=0；允许容差 0.001953126 | 连续边界工具资格验证，不是模型效应 | 完全缺失 |

Goal 新增 instrument/governance 总门：最终 current self-test bundle 和 final static qualification v3 均为 61/61 PASS。这个 61/61 只说明冻结、selection、sealed validator、analyzer、inventory、boundary geometry 的 fail-closed 实现通过测试，不是论文效应量。

负面/失败结果：约 59k 状态的 exhaustive parity 运行被主动停止，未保存安全 checkpoint，状态为 `incomplete`，不能作为证据。权威记录：`results/measurement_validation_20260831/parity_exhaustive_v1/INCOMPLETE.json`。

## 3. 主 stress-contract 结果

### 3.1 AI-TOD-v2 fixed 1 px 与 equivalent-side kappa=0.0625

分母：同一 outcome-blind 300 图 discovery sample；288 张贡献 common-valid GT；fixed 6,378 GT，normalized 6,377 GT；5,000 stratified image-cluster bootstrap。

| Contract | O2O 8–16 minus 16–32 px | O2M 8–16 minus 16–32 px | O2O-minus-O2M paired contrast |
|---|---:|---:|---:|
| fixed 1 px | +14.37 pp [10.84, 17.78] | +1.42 pp [−3.18, 6.31] | +12.95 pp [6.52, 19.16] |
| equivalent-side kappa=.0625 | +0.13 pp [−3.41, 3.15] | −17.45 pp [−22.72, −11.71] | +17.58 pp [11.06, 23.98] |

结果分类：

- 明确正向：fixed O2O、两合同 paired branch contrast；
- 明确负向：normalized O2M scale contrast；
- 零/不确定：fixed O2M、normalized O2O 的区间均包含 0；
- 科学含义：fixed-pixel 下的 O2O scale gradient 不是 stress-contract invariant；normalized stress 下 O2O 近零但 O2M 反向，branch response 明显分化；
- 边界：discovery result；prospective lockbox 没有 normative verdict。

Evidence：`EV-STRESS-CONTRACT-PRIMARY-V1`。当前论文已有且实质准确，但尚未绑定 claim ID/明确 discovery status。

### 3.2 两数据集 × 三个 kappa

| Dataset / kappa | O2O contrast | O2M contrast | Paired contrast |
|---|---:|---:|---:|
| AI-TOD-v2 .03125 | +1.34 [−0.72, 3.10] | −16.85 [−22.30, −11.00] | +18.19 [12.48, 23.83] |
| AI-TOD-v2 .0625 | +0.13 [−3.41, 3.15] | −17.45 [−22.72, −11.71] | +17.58 [11.06, 23.98] |
| AI-TOD-v2 .125 | +8.91 [4.21, 13.55] | −8.36 [−11.33, −5.41] | +17.27 [11.43, 23.45] |
| VisDrone .03125 | +5.14 [3.93, 6.36] | −29.65 [−31.73, −27.59] | +34.79 [32.19, 37.42] |
| VisDrone .0625 | +11.04 [9.33, 12.70] | −26.21 [−27.85, −24.57] | +37.25 [34.73, 39.85] |
| VisDrone .125 | +19.97 [17.73, 22.25] | −16.01 [−17.17, −14.84] | +35.98 [33.39, 38.56] |

单位均为 pp，括号为 95% CI。六行 O2M 均为负、六行 paired 均为正；absolute O2O response 随 kappa 和 dataset 改变。AI 小 kappa 的 O2O 包含 0，是明确的零/不确定结果。该矩阵是 post hoc frozen discovery grid，不支持普适 scale law，也不能把 AI kappa=.0625 与 primary row 重复计数。

Evidence：`EV-STRESS-CONTRACT-KAPPA-DATASET-V1`。当前论文已有且实质准确。

## 4. Eligibility、first-divergence 与 same-stride

### 4.1 Base-eligibility-lock counterfactual

分母：300 图；14,866 per-GT rows；59,228 directional rows。

- fixed O2O gap 减少 13.58 pp [11.21, 15.87]；
- fixed O2O gap 被移除比例 79.44% [70.51, 91.92]；
- normalized O2O gap 减少 3.15 pp [1.42, 4.83]。

这是明显的结构 gate attenuation，但不能单独称 eligibility 对整个现象具有完整因果解释。Evidence：`EV-ELIGIBILITY-LOCK-V1`。当前论文完全缺失。

### 4.2 Eligibility-stable observational subset

分母：fixed 保留 6,217/7,434 objects、28,144/29,616 directions；normalized 保留 6,999/7,432 objects、28,979/29,612 directions；paired pathway images=201。

- fixed conditional O2O scale contrast：+3.36 pp [1.11, 5.48]；
- normalized conditional contrast：+1.05 pp [−1.19, 2.96]，包含 0；
- normalized minus fixed eligibility fraction：−16.09 pp [−18.92, −13.34]。

条件化会改变 estimand，并可能引入 selection；必须与 eligibility-lock 联合解读。Evidence：`EV-ELIGIBILITY-STABLE-V2`。当前论文完全缺失。

### 4.3 Corrected pathway anatomy v2

分母：fixed 2,194 flips/201 contributing images；normalized 1,241 flips/159 images。

| Contract | Eligibility boundary | Within-set geometry rank reversal |
|---|---:|---:|
| fixed | 67.09% [63.02, 71.07] | 30.13% [26.14, 34.36] |
| normalized | 51.01% [46.06, 55.94] | 45.37% [40.62, 50.26] |

Normalization 将 first-divergence composition 从 eligibility 边界移向更高的 within-set geometry share。此 v2 取代旧图中的 24.7%/37.5% 和旧 Top-k 数值。Evidence：`EV-PATHWAY-ANATOMY-V2`。当前论文文字实质准确，但现有图过时，必须重画。

### 4.4 Strict same-stride sensitivity

分母：6,393 common-valid GT；3,835 strict paired GT。

- O2O scale contrast：+10.23 pp [7.30, 13.32]；
- O2M scale contrast：−0.12 pp [−6.83, 5.78]，接近 0 且不确定；
- O2O-minus-O2M specificity：+10.35 pp [3.54, 18.17]。

这表明 branch-specific pattern 不能仅由 cross-stride switching 概括；但 conditioning 是观察后的选择，不能排除 feature-pyramid-level contribution。Evidence：`EV-SAME-STRIDE-V1`。当前论文已有且准确，术语需从 feature-level 改为 cross-stride/feature-pyramid-level。

## 5. O2M Rank–Set decoupling

### 5.1 Fixed 1 px

分母：300 图；7,434 GT；29,616 directional replays。下列均为 tiny minus small：

- Jaccard：−4.19 pp [−4.89, −3.50]；
- base retention：−2.26 pp [−2.78, −1.78]；
- assigned-positive-set Top-1 flip：−2.38 pp [−5.14, +0.42]，包含 0；
- rank-only share among flips：−10.10 pp [−14.50, −6.20]。

Fixed stress 下 tiny assigned-positive sets 更不持久；但 Top-1 flip 的 tiny-small 差不确定。Evidence：`EV-RANK-SET-FIXED-V1`。当前论文完全缺失；旧 Supplement 的 Jaccard 表不是这个 frozen estimand。

### 5.2 Equivalent-side kappa=.0625

分母：300 图；7,432 GT；29,612 directional replays。

- Jaccard：+5.91 pp [5.18, 6.64]；
- base retention：+4.93 pp [4.54, 5.31]；
- assigned Top-1 flip：−12.80 pp [−16.02, −9.33]；
- rank-only share among flips：+22.50 pp [18.39, 26.33]。

Normalized stress 下 set-persistence 顺序反转；tiny assigned-set Top-1 flip 更少，但发生 flip 时 rank-only share 更高。Rank instability 与 positive-set turnover 不是同一状态。Evidence：`EV-RANK-SET-NORMALIZED-V1`。当前论文完全缺失。

## 6. O2O pre-resolution control

| Contract | Final loss-active minus pre-conflict Top-7 winner scale gap | Object fragility disagreement | Directional / object disagreements | Conflict coincidence |
|---|---:|---:|---:|---:|
| fixed | −0.077 pp [−0.367, +0.313] | 0.578% [0.301, 0.921] | 134 / 43 | 100% / 100% |
| normalized | −0.395 pp [−0.986, +0.034] | 0.578% [0.255, 0.971] | 96 / 43 | 100% / 100% |

两个 gap difference 均包含 0：final active 与 pre-conflict winner 的 scale gap 很接近。少量 disagreement 全部与 instrumented conflict stage 重合，支持路径定位，但不是 conflict resolver 的因果效应。

Evidence：`EV-PRERESOLUTION-FIXED-V1`、`EV-PRERESOLUTION-NORMALIZED-V1`。当前论文完全缺失。

## 7. Learning stages 与第二 detector contract

### 7.1 Six archived stages

分母：6 个物理留存 checkpoint，12 条 fixed/normalized trajectory rows。

- mAP50–95：0.1116–0.2804；
- fixed paired contrast：11.84–15.09 pp；
- normalized paired contrast：17.34–21.35 pp。

所有已存档阶段都观察到正向 paired divergence。但 epoch0.pt 是首个完成 epoch，不是随机初始化；epochs 100–280 的权重不存在，不能插值，也不能写 throughout training。正确表述是“present at all six physically archived stages spanning the first completed epoch to the final checkpoint”。Evidence：`EV-LEARNING-DYNAMICS-V1`。当前 Results 基本准确，但 Abstract/Contribution 过强。

### 7.2 YOLOv10-S normalized replication

分母：288 图；6,348 GT。

- O2O：+0.94 pp [−1.85, 3.61]，近零/不确定；
- O2M：−23.50 pp [−28.18, −18.90]；
- paired：+24.44 pp [18.89, 30.06]。

支持两个兼容 detector contracts 的 replication，不支持 architecture invariance。Evidence：`EV-YOLOV10-NORMALIZED-V1`。当前论文已有且准确。

## 8. Continuous boundary geometry

### 8.1 KM/Aalen–Johansen/RMSR

分母：300 图；7,431 focal GT；29,724 directional trajectories；O2O base-defined 29,048、undefined 676；O2M assigned-positive base-defined 28,996、undefined 728。RMSR 截断半径 tau=.125。

| Quantity | Point [95% CI] |
|---|---:|
| all O2O RMSR | 0.11992 [0.11940, 0.12038] |
| all O2M assigned-positive RMSR | 0.10055 [0.09895, 0.10210] |
| tiny O2O-minus-O2M RMSR | 0.01724 [0.01565, 0.01876] |
| small O2O-minus-O2M RMSR | 0.03282 [0.02904, 0.03614] |
| tiny O2O ever-boundary by tau | 9.03% [8.26, 9.90] |
| small O2O ever-boundary by tau | 6.39% [4.61, 8.49] |
| tiny O2M ever-boundary by tau | 32.66% [31.19, 34.14] |
| small O2M ever-boundary by tau | 48.15% [44.79, 51.31] |

O2O 的 restricted mean stability radius 更大；O2M boundary 更频繁；small 的 O2O-minus-O2M RMSR gap 更大。Right censor、competing censor、undefined 和 return transition 均被保留；这不是 causal training mechanism 或 hazard-ratio 结论。

Evidence：`EV-BOUNDARY-GEOMETRY-V2`。当前主文和 Supplement 完全缺失。

### 8.2 Margin 与固定候选对 crossing radius

29,724 trajectories 中：

- rho_R observed：1,592（5.35% [4.73, 6.04]）；
- competing-censored：7,437（25.02% [24.20, 25.83]）；
- right-censored：16,551（55.69% [52.37, 58.46]）；
- undefined：4,144（13.93% [11.23, 17.13]）。

Observed-case 条件相关：

- margin vs observed rho_R Spearman：0.628 [0.584, 0.670]；
- margin vs eligibility-boundary radius：0.153 [0.114, 0.194]；
- correlation difference：0.475 [0.405, 0.540]。

Margin 更具体地读出 fixed-pair q-order crossing radius，但这是 observed-case conditional construct validation，不是 censor-adjusted population correlation、因果关系、预测增益或 intervention efficacy。

Evidence：`EV-MARGIN-RHOR-CONSTRUCT-V2`。当前论文完全缺失。

## 9. Assignment spillover

分母：300 图；29,612 focal-direction events；3,549,250 nonfocal pair-events；shared-native-claim 5,679 pairs，nonshared 3,543,571 pairs；observed changes 52 vs 0。

- overall pair spillover：0.001465% [0.000824, 0.002383]；
- shared-claim pair：0.9159% [0.5787, 1.3248]；
- nonshared：本样本中 observed=0；
- shared minus nonshared：+0.9159 pp [0.5787, 1.3248]。

总体 spillover 极稀少，观察到的 52 个 change 全在 shared-claim pairs。Nonshared 的 0 不能写成结构性不存在、equivalence、exclusivity 或因果机制。

Pilot gate 结果也保留：30 图 pilot 只有 1/711,453 pair-event，且位于 shared-claim stratum；它触发了 formal 300-image extension，但 pilot 本身不是最终 evidence。

Evidence：`EV-SPILLOVER-FORMAL300-V1`。当前论文完全缺失。

## 10. Margin shape sensitivity

分母：6,410 GT；288 图；5 image-grouped OOF folds；每 outcome 5,000 bootstrap；spline knots 只从各 training fold 估计。

Spline minus linear：

| Outcome / metric | Difference [95% CI] |
|---|---:|
| fragility ROC-AUC | −0.00756 [−0.01411, −0.00138] |
| fragility PR-AUC | −0.00415 [−0.00813, −0.00109] |
| fragility log loss | −0.000903 [−0.002604, +0.000749] |
| coverage-based miss ROC-AUC | −0.000566 [−0.001222, +0.000022] |
| coverage-based miss PR-AUC | −0.005412 [−0.012983, −0.000266] |
| coverage-based miss log loss | −0.000596 [−0.007941, +0.005262] |

这是明确的负/零结果：spline 没有稳定的 metric-wide improvement，AUC/PR 多项更差；两个 log-loss 和 coverage ROC 包含 0。不能称等价，也不能称 margin 是最佳 predictor。

Evidence：`EV-MARGIN-SHAPE-SENSITIVITY-V1`。当前论文完全缺失；当前稿只包含旧 linear incremental analysis。

## 11. Sample efficiency 1,000 repeats

Full common-valid frame=288 images。每个 budget 做 1,000 次 outcome-blind stratified subsampling，不做 nested bootstrap。

| Budget | Median absolute error | P95 absolute error | Sign consistency | Branch-order consistency |
|---:|---:|---:|---:|---:|
| 25 | 6.39 pp | 18.38 pp | 97.5% | 97.5% |
| 50 | 4.52 pp | 13.63 pp | 99.6% | 99.6% |
| 100 | 3.43 pp | 8.41 pp | 100% | 100% |
| 150 | 2.45 pp | 6.13 pp | 100% | 100% |
| 200 | 1.74 pp | 4.60 pp | 100% | 100% |

100 图能稳定恢复 full normalized paired contrast 的符号与 branch ordering，但 p95 error 仍有 8.41 pp，只能支持 dominant-signature screening，不能作为精确估计、power 或样本量建议。

Evidence：`EV-SAMPLE-EFFICIENCY-1000-V1`。当前论文仍是旧 100 repeats、3.53/8.29 pp，Methods、图、表和 Discussion 均过时。

## 12. Prospective precision planning

分母：300 discovery images；strata 2/138/160；5,000 bootstrap；5 个 planned estimands。

| Planned estimand | Discovery point | Expected percentile 95% half-width | Direction recovery |
|---|---:|---:|---:|
| H1 normalized paired | +17.58 pp | 6.56 pp | 100% |
| H2 normalized O2M | −17.45 pp | 5.53 pp | 100% |
| H3 fixed minus normalized O2O | +14.24 pp | 1.73 pp | 100% |
| H4a fixed eligibility minus normalized | +16.09 pp | 2.78 pp | 100% |
| H4b normalized within minus fixed | +15.24 pp | 2.58 pp | 100% |

这只是 outcome access 前的 discovery-based planning diagnostics，不是 post-hoc power、MDE、observed power 或 lockbox precision guarantee。

Evidence：`EV-PRECISION-DESIGN-V1`。当前论文完全缺失。

## 13. Lockbox：选择、提取、失败与 exploratory correction

### 13.1 成功完成的治理/提取步骤

- instrument source commit/tag：`afa28c5587791d533992e8d7372cdec94c7fcb81` / `audit_instrument_v1.0`；
- preregistration commit/tag：`982657d84d3adaa34cfc0045744a73e6c0d3f4dd` / `audit_preregistration_v1.0`；
- 300-image outcome-blind lockbox，allocation t_only/s_only/t_and_s=160/2/138，discovery overlap=0；
- selection commit：`de2060c85702899e83cce77647f984087d9a4f1e`；
- fixed branch：8,057 per-GT / 32,061 directional rows；
- normalized branch：8,053 / 32,049；
- fixed anatomy：32,061 directional / 1,808 q-pair rows；
- normalized anatomy：32,049 / 1,102；
- sealed bundle READY SHA-256：`5392b04eb5076fb3f729b99584c7e9dd7cb96853e6be20a876d5b47a98400010`。

这些是治理/原始提取计数，不是效应结果。

### 13.2 Prospective analyzer 的正式状态

唯一一次 analyzer 调用在读取结果前原子写入了不可逆 receipt：

`8a5eaee6703c75fec23ec287cf77db2f833a4d85540f21ac2c147b78a0464c83`

随后在任何 H1–H5 estimate/verdict 写出前失败：

- selected=300，但四个 raw roles 只有同一 299 images；缺失图 `1063__1200_1200` 本应作为 zero-valued cluster；
- fixed common-valid GT=7,246，normalized=7,243；存在 3 fixed-only keys；
- frozen analyzer 在 zero-fill/交集处理之前强制 raw roster 和 support 完全相等。

因此：

- 没有 `LOCKBOX_VERDICT.json`；
- 没有 H1–H5 prospective PASS/FAIL；
- 不是“不显著”，而是 confirmatory verdict 根本不存在；
- prospective rerun、换样本、换 threshold 或恢复 gatekeeping 均禁止。

权威失败记录：`reports/PROSPECTIVE_LOCKBOX_ANALYZER_FAILED_V1.json`。当前主文与 Supplement 完全没有呈现，这是必须补上的负面结果。

### 13.3 Post-lockbox corrective exploratory

纠正规则是在 outcome access 后明确的：保留全部 300 clusters；缺 raw rows 的图为 zero-valued cluster；branch 使用 7,243 个 fixed∩normalized common-valid GT intersection，并披露 3 个 fixed-only keys。5,000 次 bootstrap 逐元素重放通过，23/23 validation PASS。

| Analogue | Point | Exploratory 95% CI | Formal status |
|---|---:|---:|---|
| H1 normalized paired | +16.69 pp | [12.47, 20.74] | `formal_pass=null` |
| H2 normalized O2M | −16.99 pp | [−20.62, −13.34] | `formal_pass=null` |
| H3 fixed minus normalized O2O | +14.00 pp | [12.51, 15.60] | `formal_pass=null` |
| H4 fixed eligibility minus normalized | +21.15 pp | [18.54, 23.44] | `formal_pass=null` |
| H5 normalized within minus fixed | +19.31 pp | [16.53, 21.94] | `formal_pass=null` |

五个方向均与 preregistration 一致，但任何一项都不能称 confirmed、replicated、significant、passed 或 formally tested。Evidence：`EV-LOCKBOX-CORRECTIVE-EXPLORATORY-V1`。

## 14. 全部 Goal 新增 run 的终态清单

### 14.1 Complete / evidence-eligible（14）

| Run ID | 结果角色 | 核心输出 |
|---|---|---|
| `20260831T152124_parity_deterministic_5k_v1` | instrument evidence | 300+5,251 states，0 mismatch |
| `20260831T161235_rank_set_fixed_v1` | discovery | fixed Rank–Set |
| `20260831T162551_rank_set_normalized_v1` | discovery | normalized Rank–Set |
| `20260831T163900_preresolution_fixed_v1` | discovery control | fixed pre-resolution |
| `20260831T164000_preresolution_normalized_v1` | discovery control | normalized pre-resolution |
| `20260831_boundary_reference_formal_v2` | instrument evidence | 4,408/4,408，0 mismatch/radius diff |
| `20260831_boundary_geometry_formal_v2` | discovery | KM/AJ/RMSR，5,000 bootstrap |
| `20260831_margin_shape_formal_v1` | supplementary diagnostic | 6,410 GT、5 OOF folds、5,000 bootstrap |
| `20260831_sample_efficiency_1000_v1` | supplementary engineering | 5 budgets × 1,000 repeats |
| `20260831_lockbox_precision_design_v1` | pre-freeze planning | 5 estimands × 5,000 bootstrap |
| `20260901_spillover_formal300_v1` | gated discovery | 3,549,250 pairs，52 changes all shared-claim |
| `20260901_post_lockbox_corrective_v1` | post-lockbox exploratory | 300 clusters、7,243 intersection、5,000 bootstrap |
| `20260901_final_evidence_v1` | evidence governance | 22-entry initial final freeze；已保留 |
| `20260901_final_evidence_v2` | manuscript authority | 24 entries、5 excluded、16/16 PASS |

### 14.2 Complete / governance or upstream-only（14）

| Run ID | 结果 |
|---|---|
| `20260831_boundary_primary_data_repeatability_v2` | boundary repeatability/hash 验证，不是科学 effect |
| `20260831T174200_boundary_discovery_v2` | upstream raw scanner：300 images、7,431 GT、29,724 directions |
| `20260831_spillover_pilot30_v1` | pilot gate：1/711,453 event；触发 formal extension |
| `20260831_lockbox_chain_selftests_pre_freeze_v3` | freeze 5/5、selector 7/7、validator 6/6、analyzer 19/19 |
| `20260901_lockbox_chain_selftests_pre_freeze_current` | 修复后 61/61 |
| `20260901_final_static_qualification_v3` | 最终 61/61 |
| `20260901_instrument_freeze_v1_final` | 399-file source/prereg seal；无 effect |
| `20260901_lockbox_selection_v1` | 300 images，160/2/138，overlap=0 |
| `20260901_lockbox_fixed_branch_v1` | raw rows 8,057/32,061；无 verdict |
| `20260901_lockbox_normalized_branch_v1` | raw rows 8,053/32,049；无 verdict |
| `20260901_lockbox_fixed_anatomy_v1` | raw rows 32,061 + q-pair 1,808 |
| `20260901_lockbox_normalized_anatomy_v1` | raw rows 32,049 + q-pair 1,102 |
| `20260901_lockbox_metadata_repackaging_v1` | 只修 selection metadata bytes；outcome hashes byte-identical |
| `20260901_sealed_bundle_ready_v1` | frozen validator PASS；READY=300 images |

### 14.3 Complete / smoke-only（5；不能作为论文效应）

- `20260831_rank_set_smoke_v2`；
- `20260831_preresolution_fixed_smoke_v1`；
- `20260831_boundary_radius_smoke_v1`；
- `20260831_boundary_reference_smoke_v2`（pre-domain-fix superseded）；
- `20260831_boundary_reference_smoke_v3`（48/48 instrument validation）。

### 14.4 Failed（5）

| Run ID | 失败原因 | 科学结果可用？ |
|---|---|---|
| `20260831_boundary_reference_smoke_v1` | production/reference selector subsets 不重叠 | 否；不是 semantic mismatch |
| `20260831T170918_boundary_discovery_v1` | base-outside GT 暴露缺失 domain precondition | 否；partial rows retained only as history |
| `20260901_instrument_freeze_v1_attempt1` | isolated finalize 错读 ROOT/results 而非 staged frozen_inputs | 否；pre-selection fail |
| `20260901_instrument_freeze_v1_attempt2` | Python bytecode cache 进入 frozen inventory，seal fail-closed | 否；pre-selection fail |
| `20260901_sealed_bundle_validation_attempt1` | selected_images float re-serialization 导致 byte hash mismatch | 否；READY 未创建，outcome 未开启 |

### 14.5 Invalidated（5）

| Run ID | 原因 | 后续替代 |
|---|---|---|
| `20260831_boundary_reference_formal_v1` | 未执行；依赖的 production v1 失败 | formal v2 |
| `20260831_boundary_geometry_formal_v1` | 未执行；依赖失败 | formal v2 |
| `20260901_final_static_qualification_v1` | self-test 有 false subcheck 仍报 overall PASS 的 fail-open defect | v3 61/61 |
| `20260901_final_static_qualification_v2` | 缺 no-bytecode invariant | v3 61/61 |
| `20260901_prospective_lockbox_analyzer_v1` | receipt 后 support-contract defect，verdict 前失败 | 无 prospective 替代；corrective only exploratory |

### 14.6 Incomplete（1）

- `20260831T130532_parity_exhaustive_v1`：主动停止、没有安全 checkpoint/final summary，不得恢复成证据。

### 14.7 既有但纳入最终对账的 legacy evidence（9）

这些不是 Goal 新建 run，但 final ledger 仍依赖：oracle/property、mutation、300+975 parity、eligibility-stable v2、eligibility-lock、corrected anatomy v2、same-stride、learning dynamics、YOLOv10 normalized replication。

## 15. 当前论文缺失/过时对照

24 个 final evidence IDs 中：

- 当前主文已有且准确：`EV-PATHWAY-ANATOMY-V2`、`EV-SAME-STRIDE-V1`、`EV-YOLOV10-NORMALIZED-V1`、`EV-STRESS-CONTRACT-PRIMARY-V1`、`EV-STRESS-CONTRACT-KAPPA-DATASET-V1`；
- 已有但过时/不完整：`EV-LEARNING-DYNAMICS-V1`（throughout training 过强）、`EV-SAMPLE-EFFICIENCY-1000-V1`（仍是旧 100 repeats）；
- 完全缺失：其余 17 条，包括全部 instrument qualification、eligibility lock/stable、Rank–Set、pre-resolution、boundary reference/geometry、margin–rho_R、spillover、margin shape、precision planning、prospective lockbox failure 和 corrective exploratory。

此外：

- `FN@IoU50/false negative` 应统一改为 `coverage-based miss` 或 `GT-miss@IoU50`；
- `feature-level switching` 应改成 `cross-stride (feature-pyramid-level) switching`；
- intervention 不应继续占摘要/主贡献；
- 当前 anatomy 图使用旧比例，当前 sample-efficiency 图使用旧 100-repeat 数据；两者必须重画；
- lockbox 不是“不显著”，而是没有合法 prospective verdict。

## 16. 权威文件

- 机器可读全部 24 条：`evidence_freeze/final_evidence_v2/CURRENT_EVIDENCE_LEDGER.json`；
- result-to-claim：`evidence_freeze/final_evidence_v2/RESULT_TO_CLAIM_MAPPING.json`；
- source-to-result：`evidence_freeze/final_evidence_v2/SOURCE_TO_RESULT_MAPPING.json`；
- 完整 run 状态：`RUN_REGISTRY.csv`；
- invalidated prospective record：`reports/PROSPECTIVE_LOCKBOX_ANALYZER_FAILED_V1.json`；
- exploratory corrective：`lockbox_runs/post_lockbox_corrective_v1/POST_LOCKBOX_CORRECTIVE_RESULTS.json`；
- final evidence v2 manifest：`evidence_freeze/final_evidence_v2/FINAL_EVIDENCE_MANIFEST.json`。

本报告不把任何 smoke、failed、invalidated、incomplete 或 post-lockbox exploratory 结果升级为 confirmatory evidence。
