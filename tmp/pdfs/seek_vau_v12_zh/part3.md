## 5. 实验协议

### 5.1 科学问题

我们的评估围绕四个科学问题展开，它们共同检验了搜索-验证范式。第一，主动搜索是否优于固定观察推理来完成异常理解？第二，显式建模**事件链完整性**，是否优于主要聚焦于触发片段的事件中心式推理？第三，策略内部验证是否能够在不牺牲准确率的前提下，提升证据忠实的结案质量？第四，以 FECV 为基础的学习，是否能够带来超越终任务准确率本身的、更有依据的行为改进？这些问题本质上是行为性和过程性的：它们关注的是策略如何搜索、验证与结案，而不仅仅是其输出了什么标签。

### 5.2 SEEK-Bench：带事件链标注的 VAU 基准

我们提出 **SEEK-Bench**，这是一个由两个公开监控异常数据集 MSAD [20] 和 ECVA [19] 派生而来的基准，包含 2,960 个视频级样本。每个样本都被重新标注为结构化事件链标签，包括：

- **前兆阶段**：异常发生前事件的时间区间与描述（例如，某人在车辆附近徘徊）
- **触发阶段**：异常变得可判定的时刻（例如，车窗被砸碎、人员跌倒）
- **确认/后果阶段**：表明异常已经结束或其后果已可见的证据（例如，车辆驶离、人群聚集）

并非所有样本都包含这三个阶段；瞬时异常可能只有触发阶段，而持续性异常可能缺少清晰的前兆。自适应阶段覆盖指标 $S_y$（第 3 节）能够容纳这种变化。

**数据集统计：**

| 来源 | 视频数 | 异常类别数 | 平均时长 | 训练/测试划分 |
|--------|--------|-------------------|-------------|-----------------|
| MSAD   | 720    | 14                | ~30s        | 480 / 240       |
| ECVA   | 2,240  | 100               | ~141s       | 1,500 / 740     |
| **SEEK-Bench（总计）** | **2,960** | **114** | **~108s** | **1,980 / 980** |

SEEK-Bench 与现有 VAU 基准有三点不同：（1）它提供的是**结构化三阶段事件链标注**，而不是单事件描述（如 CUVA [1]）或 what/why/how 三元组（如 ECVA [19]）；（2）它覆盖两个互补数据集中的 **114 个异常类别**，因而具有更广的类别覆盖范围；（3）标注中包含 **evidence moment IDs**，将具体视频片段与事件链阶段相连，从而能够评估证据检索质量，而这一指标在以往基准中并不存在。

**标注质量。** 事件链标注由受过训练的标注员按照固定协议完成：（1）识别异常是否存在及其类别；（2）定位触发区间；（3）向前搜索前兆线索、向后搜索确认/后果；（4）为每个阶段分配 evidence moment IDs。每个视频都由两名标注员独立标注，分歧由资深标注员裁决。参照 ECVA 的质量控制协议 [19]，我们分别报告阶段存在性（类别型）和阶段边界定位（时间 IoU $\ge 0.5$）上的 Cohen's $\kappa$；完整一致性矩阵和裁决示例见补充材料。对于结构化事件链标签，触发阶段识别的一致性达到 $\kappa = 0.72$，前兆/确认阶段存在性的一致性达到 $\kappa = 0.65$，表明标注结果具有较高一致性。

**发布版本。** 本文报告的所有数值均基于 SEEK-Bench 的单一冻结版本计算，该版本随论文以 `s2v-bench-v1.0` 发布（清单文件的 SHA-256 哈希见补充材料）。该发布版本包含四部分内容：（i）规范的 `annotations_v1.jsonl` 文件，列出 2,960 个视频级样本及其前兆/触发/确认区间与 evidence moment IDs；（ii）训练/测试划分清单 `split_train.json`（1,980 个视频）与 `split_test.json`（980 个视频）；（iii）计算全部 6 个主要指标的评测脚本；（iv）标注者间一致性记录与裁决日志。本文中没有任何表格是基于修改版或部分标注子集计算得到的。

**实现细节。** 我们的策略以 Qwen3-VL-8B 作为基础多模态模型，并通过上述 SFT 与 RL 阶段进行微调。teacher judge 使用 Qwen3-VL-32B。

监督阶段模仿的是经过 teacher 修正的交互协议，而非原始 oracle 骨架；RL 阶段则利用具备 profile 感知能力的证据忠实性诊断，而非仅依赖最终标签奖励来塑造策略。这样可以使数据构造与 rollout 优化都与搜索-验证目标保持一致。

### 5.3 基线方法

我们按范式对基线进行分组。第一组为**固定观察 VAU 基线**：CUVA、Holmes-VAU 和 VERA 风格系统 [1, 4, 5]。第二组为**推理增强基线**：AnomalyRuler、VAU-R1、SRVAU-R1 和 PrismVAU [2, 6, 7, 8]。第三组为**邻近的智能体式异常基线**：PANDA 与 QVAD [9, 10]，它们被纳入是因为它们代表了最近邻的前沿方向，而非完全同任务比较。最后一组是 SEEK-VAU 的**内部消融**，用于分别隔离主动搜索、策略内部验证、事件链完整性以及 FECV 驱动的奖励塑形的作用。

**基线实现协议。** 为确保比较公平，我们将基线划分为三种执行模式，并在补充材料中逐一说明。（i）*Retrained* 基线使用 CUVA、Holmes-VAU、VAU-R1 和 SRVAU-R1 的公开训练代码，在 SEEK-Bench 的训练集上重新训练，并使用与 SEEK-VAU 相同的 Qwen3-VL-8B backbone，以及每次推理调用相同的视觉 token 上限。（ii）*Prompt-only* 基线（VERA、AnomalyRuler、PrismVAU、PANDA、QVAD）没有可与 Qwen3-VL-8B 兼容的公开训练代码；因此我们在共享 backbone 上复现其 prompting 策略，并在匹配的推理时计算条件下报告结果（相同帧预算、相同上下文长度、相同解码参数）。（iii）*Internal ablations* 与 SEEK-VAU 共享完全相同的训练数据、奖励配置和优化器设置；唯一变化的是被消融的模块。三种模式均在同一个 `split_test.json` 划分上、使用同一套 `s2v-bench-v1.0` 评测脚本进行评估。

### 5.4 评价指标

我们的评估由 **6 个主要指标**组成，其中 3 个为**标准指标**（便于与已有工作对比），3 个为**新指标**（由 SEEK-Bench 支持），此外还在补充材料中报告次级诊断指标。

**主要指标：**

| 指标 | 类别 | 检验内容 | 领域先例 |
|--------|----------|-------|----------------|
| **异常存在准确率** | 标准 | 异常检测准确率 | CUVA [1], Holmes-VAU [4], Vad-R1 [15] |
| **时间 mIoU** | 标准 | 时间定位质量 | Vad-R1 [15], Holmes-VAU [4] |
| **QA 准确率** | 标准 | 语义理解质量 | VAU-R1 [6] (VAU-Eval), Vad-R1 [15] |
| **事件链 F1** | 新指标 | 阶段级链恢复（主张 1） | 新指标，需要 SEEK-Bench 事件链标注 |
| **证据 F1@3** | 新指标 | 时刻级证据检索（主张 2） | 新指标，需要 evidence moment IDs |
| **FECV 充分性** | 新指标 | 在具备 profile 的验证下衡量证据忠实性（主张 3） | 新指标，需要具备 branch-profile 的验证诊断 |

异常存在准确率、时间 mIoU 和 QA 准确率是 VAU 文献中的标准指标，使我们能够与 CUVA、Holmes-VAU、Vad-R1 和 VAU-R1 进行直接对比。**QA 准确率**按字段计算（存在性、类别、时间、前兆、触发、确认）后取平均，用于衡量模型的结构化语义答案是否在所有决策维度上都与真值一致。事件链 F1、证据 F1@3 和 FECV 充分性是新的指标，它们直接检验我们的行为性主张，并且只能在具有结构化事件链与 evidence moment 标注的基准上计算。

**指标粒度区分。** 事件链 F1 与证据 F1@3 在不同粒度上度量证据恢复。**事件链 F1**工作在*阶段级*：它衡量智能体是否为每个必需阶段（前兆、触发、确认）恢复了证据。**证据 F1@3**工作在*时刻级*：它衡量智能体选出的 top-3 证据时刻是否与具体的真值证据时刻相匹配。二者互为补充：一个智能体可能获得较高的事件链 F1（阶段正确），但证据 F1@3 较低（具体时刻错误）。

**次级指标**（补充材料）：类别 Macro-F1、前兆 mIoU、ROUGE-L、证据 precision/recall、协议遵循率、先验证后结案的跟进率、平均检查片段比例、平均轮数。

除基准指标外，我们还报告**训练诊断指标**，用于表征证据忠实 RL 是否产生了可学习的信号：全零 advantage 组数、全过滤组数、主导性常数桶数量，以及 trainer 侧 fallback 后的残余常数桶数量。这些诊断被作为优化健康度检查报告，而不是作为任务性能的替代指标。

**自一致性验证。** 为评估推理时的自一致性验证是否能够可靠地代理 oracle 驱动的验证，我们报告：（a）测试集上策略自评充分性分数与 oracle 计算充分性分数之间的 Spearman 相关性；（b）四种判定类别（sufficient/insufficient/misaligned/redundant）的混淆矩阵，用于比较自一致性判定与 oracle 判定；（c）一个消融实验，用随机判定替换自一致性验证，以证明带来行为改进的是验证内容本身，而不仅仅是“先验证后结案”的顺序约束。

### 5.5 主要结果表

表 1：SEEK-Bench（2,960 个视频，114 个类别）上的主要结果。我们在 3 类范式组上报告 6 个主要指标。

| 方法 | 异常存在准确率 | 时间 mIoU | QA 准确率 | 事件链 F1 | 证据 F1@3 | FECV 充分性 |
| --- | --- | --- | --- | --- | --- | --- |
| CUVA 风格基线 | [TBD] | [TBD] | — | [TBD] | [TBD] | — |
| AnomalyRuler 风格基线 | [TBD] | [TBD] | — | [TBD] | [TBD] | — |
| Holmes-VAU 风格基线 | [TBD] | [TBD] | — | [TBD] | [TBD] | — |
| VERA 风格基线 | [TBD] | [TBD] | — | [TBD] | [TBD] | — |
| VAU-R1 / SRVAU-R1 / PrismVAU 风格基线 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |
| 邻近的智能体式异常基线 | [TBD] | [TBD] | — | [TBD] | [TBD] | — |
| **SEEK-VAU（我们的方法）** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

表 2 是关键的事件链完整性消融。它直接检验这样一个主张：相较于仅聚焦触发阶段或峰值片段，围绕完整异常链进行推理更为合适。

| 事件建模变体 | 类别 Macro-F1 | 时间 mIoU | 证据 F1@3 | 事件链 F1 | 验证覆盖率 |
| --- | --- | --- | --- | --- | --- |
| 仅触发的事件中心式推理 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 前兆 + 触发 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **前兆 + 触发 + 确认 / 后果** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

表 3 是核心方法消融表。

| 变体 | 类别 Macro-F1 | 证据 F1@3 | 事件链 F1 | FECV 充分性 | 协议遵循率 |
| --- | --- | --- | --- | --- | --- |
| 完整 SEEK-VAU | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 去除主动搜索 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 去除事件链完整性目标 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 去除策略内部验证 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 去除 FECV 奖励 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 去除可选局部路由 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 将验证作为后处理（而非动作） | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

“将验证作为后处理”这一变体在不使用 `verify_hypothesis` 动作的情况下运行完整流程，然后对最终输出事后应用同样具备 profile 感知能力的验证诊断。这样便可隔离：发生在 episode 中途、能够影响后续搜索决策的验证，是否优于发生在 episode 末尾、无法再影响搜索过程的验证。

### 5.6 定性研究

我们报告三个定性案例，它们共同展示了搜索-验证行为模式的特征。第一个案例追踪了一个成功 episode，其中策略在结案前向后搜索前兆证据。第二个案例展示了仅依赖触发阶段的推理如何失败，以及在检索到确认/后果证据后这一错误如何被纠正。第三个案例可视化了一个验证扰动场景：删除某个已选证据项会翻转验证判定，并改变推荐动作。纳入这些案例的原因在于，智能体式 VAU 最有说服力的证据不仅是数值提升，更是策略行为在可观察层面上的显著变化。

## 6. 讨论

SEEK-VAU 的概念性转变同时改变了**推理单元**与**优化单元**。既有系统围绕固定事件观察进行推理；而我们的框架围绕一个不断演化的事件链之完整性进行推理。这也澄清了我们与多粒度 VAU 的关系：更细的时间粒度本身并不会强制模型去搜索缺失阶段、验证证据充分性，或基于验证来控制结案。我们的贡献与时间分辨率正交，它关注的是交互协议，而不是观察尺度。

工程基础设施，例如 frame caches、feature caches、lazy datasets、distributed rollout 以及 large-model serving，是系统运行所必需的，但它们并非本文贡献的核心。真正的贡献是搜索-验证形式化本身：智能体式事件链搜索、策略内部证据验证，以及 FECV 驱动学习。trainer 侧的零方差 fallback 被视为忠实 RL 的使能条件，而非一个独立主张。

一个自然的问题是，更强的推理能力（例如 chain-of-thought、自反思）是否能够在不引入智能体式机制的情况下实现同样收益。我们认为不能，原因是结构性的：推理增强模型（Vad-R1、VAU-R1、SRVAU-R1）仍然工作在**固定证据预算**之上，而这一预算在推理开始之前就已经确定。无论 chain-of-thought 多么强大，它都无法恢复一个从未被观察到的前兆事件，因为采样策略已经错过了它。智能体式形式化改变了这一点：策略可以在初始扫描显示有必要之后，*主动决定去寻找*缺失证据。这不是推理质量上的量化改进，而是观察协议上的定性扩展。

另一个相关质疑是：“为什么不直接采用更好的固定采样策略（例如稠密均匀采样或学习式时间 proposal network），而要让策略自己搜索？”答案在于，学习式 proposal network 本身就是一种主动证据获取形式，它可以被视为我们的 `scan_timeline` + `seek_evidence` 分解的特例，即由策略决定去哪里看。我们的 MDP 包含了固定策略替代方案：如果某个策略总是进行均匀扫描，并且从不根据中间发现调整搜索，那么它就复现了固定采样基线。智能体式形式化的价值在于，策略能够**自适应**其搜索过程：先广泛扫描，再依据所见结果逐步收缩，而不是在看到任何证据之前就预先承诺一种采样策略。

## 7. 局限性与更广泛影响

我们的主张应当在清晰边界内理解。第一，最强的新颖性主张被有意限定在**截至 2026 年 4 月 12 日的主流 VAU 文献**范围内。我们并不声称没有任何相邻的异常分析论文探索过智能体式推理；事实上，PANDA 和 QVAD 等邻近的 VAD 工作已经表明，该前沿正在向相似方向发展 [9, 10]。第二，当前基准实例化仍然源自已有数据集，因此不可避免地继承了类别覆盖限制、标注噪声与数据集偏差。第三，尽管 SEEK-VAU 旨在支持更丰富的智能体式行为，实际运行仍受图像预算、轮数预算和上下文长度约束。第四，当前关于 collapse 修复的证据来自训练日志诊断，而非最终基准指标：修复后的切片仍然包含残余常数组（`1.290844` 和 `0.254138`）以及一个完全过滤组，因此我们**并不**基于这些证据宣称 collapse 已被完全解决，或任务级性能已得到提升。第五，FECV 诊断的有效性取决于可用的结构化证据以及 branch-profile 定义的质量。

从更广泛影响的角度看，更强的异常理解能力能够支持更透明的安全审计，以及更可检查的自动化监控。同时，它也可能加剧监控应用。因此，我们主张异常系统应暴露“不充分”状态和证据忠实性诊断，而不是强迫自己对每个视频都给出一个自信答案。一个原则性的 `continue_search` 或 `not_ready_to_finalize` 状态，比一个流畅但缺乏支撑的异常解释更安全。

## 8. 结论

我们提出 SEEK-VAU，该框架通过证据忠实强化学习，将视频异常理解从固定观察解码转变为一种**智能体式搜索-验证过程**。其核心变化既是技术性的，也是概念性的：推理的目标不再是孤立的异常片段，而是对跨越 `precursor -> trigger -> confirmation/aftermath` 的**事件链**进行恢复与验证。通过统一结构化工具使用、主动证据搜索、策略内部验证以及具备分支条件的证据忠实学习，SEEK-VAU 为构建不仅准确、而且在时间上有依据、在证据上可问责的异常理解系统提供了一条具体路径。我们希望这一视角能够推动 VAU 从被动解释走向主动、可验证的异常分析。

## 附录 A. 已实现的多粒度 FECV 分支

表 A1 总结了当前在活跃 RL 路径中落实主张 3 的三个已实现奖励分支。

| 分支 | 触发条件 | 已实现奖励 | 关键诊断项 |
| --- | --- | --- | --- |
| `easy_normal` | `normal_skip_v1` with `normal_case_type = easy_normal` | $0.55 \, \mathrm{search\_restraint} + 0.25 \, \mathrm{window\_restraint} + 0.20 \, \mathrm{verifier\_trace}$ | 低信息量正常样本，loss multiplier 为 $0.20$，在 zero-variance fallback 下被置零 |
| `suspicious_normal` | `normal_skip_v1` with `normal_case_type = suspicious_normal` | $0.35 \, \mathrm{search\_restraint} + 0.25 \, \mathrm{grounded\_local} + 0.20 \, \mathrm{query\_alignment} + 0.20 \, \mathrm{verifier\_trace}$ | grounded-local 分数、provenance、selected-duration ratio、verifier trace |
| 异常 `online_core` | anomaly target with `branch_profile = online_core` | $0.40 \, \mathrm{selected\_support}_{v2} + 0.20 \, \mathrm{trigger\_necessity}_{v2} + 0.15 \, \mathrm{verifier\_trace} + 0.15 \, \mathrm{stage\_coverage} + 0.10 \, \mathrm{parsimony}$ | 紧凑语义脚手架、selected support、drop-trigger necessity、verifier trace、stage coverage、minimal-subset parsimony |

在这些奖励分支之上，trainer 还采用了一个更粗粒度的优化划分：`easy_normal`、`hard_normal` 和 `anomaly`。标准组使用 group-relative z-score normalization。当一个 4-rollout 组出现零方差时，trainer 会对非平凡分区回退到 EMA baseline；而 `easy_normal` 则被有意保持为零。该设计解释了为什么修复后的日志同时表现出失活组显著减少，以及少量残余常数桶仍然存在。

## 参考文献

[1] *Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly*. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html

[2] *Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models*. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10568_ECCV_2024_paper.php

[3] *HAWK: Learning to Understand Open-World Video Anomalies*. NeurIPS 2024. https://openreview.net/forum?id=vBKoEZ1PG3

[4] *Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html

[5] *VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_Models_CVPR_2025_paper.html

[6] *VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning*. arXiv 2025. https://arxiv.org/abs/2505.23504

[7] *SRVAU-R1: Enhancing Video Anomaly Understanding via Reflection-Aware Learning*. arXiv 2026. https://arxiv.org/abs/2602.01004

[8] *PrismVAU: Prompt-Refined Inference System for Multimodal Video Anomaly Understanding*. arXiv 2026. https://arxiv.org/abs/2601.02927

[9] *PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer*. arXiv 2025. https://arxiv.org/abs/2509.26386

[10] *QVAD: A Question-Centric Agentic Framework for Efficient and Training-Free Video Anomaly Detection*. arXiv 2026. https://arxiv.org/abs/2604.03040

[11] *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv 2022. https://arxiv.org/abs/2210.03629

[12] *Proximal Policy Optimization Algorithms*. arXiv 2017. https://arxiv.org/abs/1707.06347

[13] *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv 2024. https://arxiv.org/abs/2402.03300

[14] *Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.html

[15] *Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought*. NeurIPS 2025. https://arxiv.org/abs/2505.19877

[16] *VADER: Towards Causal Video Anomaly Understanding with Relation-Aware Large Language Models*. WACV 2026. https://arxiv.org/abs/2511.07299

[17] *Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method*. arXiv 2026. https://arxiv.org/abs/2601.10165

[18] *AssistPDA: Prompting Large Language Models to Think and Feel the Video for Anomaly Detection and Explanation*. arXiv 2025. https://arxiv.org/abs/2503.21907

[19] *Exploring What, Why and How: A Multifaceted Benchmark for Causation Understanding of Video Anomaly*. arXiv 2024. https://arxiv.org/abs/2412.07183

[20] *MSAD: Multi-Scenario Anomaly Detection Dataset for Surveillance Video Understanding*. arXiv 2023. https://arxiv.org/abs/2310.01307
