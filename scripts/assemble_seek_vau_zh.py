#!/root/miniconda3/bin/python
from pathlib import Path


PART2 = r"""## 3. 问题表述

我们考虑一个视频异常理解回合，其由视频 $V$、任务查询 $q$ 和结构化目标异常案例 $y$ 构成。目标案例并不只是一个类别标签。它还包括异常是否存在、类别、时间上锚定的区间、证据时刻以及语义解释。在我们的实现中，这些字段被具体化为运行时回合中的结构，同时支持监督重放与在线 rollout。

在步骤 $t$，策略维护状态 $s_t = (h_t, E_t, M_t, c_t)$，其中包含对话历史 $h_t$、当前证据账本 $E_t$、由先前扫描得到的时间图 $M_t$，以及当前工作假设 $c_t$（一个结构化断言，包含异常类别、时间区间与严重程度估计）。动作空间被限制为四个可执行动作：

1. `scan_timeline`，在视频时间线上执行广覆盖搜索与定位。
2. `seek_evidence`，为当前假设检索更有针对性的候选证据。
3. `verify_hypothesis`，测试所选证据子集是充分、不足、错配还是冗余。
4. `finalize_case`，输出结构化异常决策。

该实现中的一条关键语义规则是，`scan_timeline` **并不**构成证据本身。它是一种广义搜索操作。证据账本由 `seek_evidence` 填充，因为只有被检索到的证据项才允许支撑验证与终结。这一区分对于训练和评估都至关重要，否则模型就可能模糊粗粒度扫描与实际证据承诺之间的差别。

核心任务目标是恢复一条连贯的异常事件链。设恢复得到的事件链表示为三个有序阶段集合，

$$
C = \{C_{\mathrm{pre}}, C_{\mathrm{trg}}, C_{\mathrm{conf}}\},
$$

其中，$C_{\mathrm{pre}}$ 表示前兆证据，$C_{\mathrm{trg}}$ 表示触发证据，$C_{\mathrm{conf}}$ 表示确认或后果证据。事件链完整性意味着最终决策不仅类别正确，还应当由一条阶段覆盖与目标异常相匹配的事件链来支撑。

**形式化 MDP。** 我们将 VAU 回合形式化为一个 Markov 决策过程

$$
M = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma),
$$

其中：
- **$\mathcal{S}$** 是联合状态空间，

  $$
  s_t = (h_t, E_t, M_t, c_t),
  $$

  其中 $h_t$ 为对话历史，$E_t$ 为证据账本，$M_t$ 为粗粒度时间图，$c_t$ 为当前工作假设。
- **$\mathcal{A}$** = {`scan_timeline`, `seek_evidence`, `verify_hypothesis`, `finalize_case`} 是离散动作集合。
- **$\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$** 是环境转移（工具执行与上下文更新）。
- **$\mathcal{R}$** 是轨迹奖励（定义如下）。
- **$\gamma \in (0, 1]$** 是折扣因子。

我们将该 MDP 实例化为一个**回合式、无折扣（$\gamma = 1$）的决策过程**，其固定轮次预算为 $T_{\max} = 10$。状态表示是完整对话历史的串联，其中包括工具调用参数与工具返回观测，策略（一个因果语言模型）以自回归方式对其进行处理。我们并不声称它在经典意义上严格满足 Markov 性；相反，这里的 MDP 形式化主要作为一个操作性框架，用于定义动作空间、奖励结构与训练目标。给定工具执行，转移 $\mathcal{T}$ 是确定性的：每个动作都会产生一个工具观测，并将其追加到对话中，从而相应更新 $E_t$ 和 $M_t$。在当前激活的 RL 配置中，GRPO 对每个 prompt 采样 **4 个 generation**，并在每个 4-rollout 组内计算相对优势。

**奖励函数。** 轨迹奖励分解为：

$$
R(\tau) = w_{\mathrm{acc}} R_{\mathrm{acc}}(\tau) + w_{\mathrm{fecv}} R_{\mathrm{fecv}}(\tau) + w_{\mathrm{prot}} R_{\mathrm{protocol}}(\tau).
$$

默认权重为 **$w_{\mathrm{acc}} = 1.0$、$w_{\mathrm{fecv}} = 0.35$ 和 $w_{\mathrm{prot}} = 0.05$**。这一权重比例反映了有意的设计选择：**正确性是首要信号**（权重 1.0），因为与“证据好但答案错”的策略相比，“答案对但证据差”的策略仍然更可取，错误答案无法仅凭忠实证据被挽回。证据忠实性（权重 0.35）是次级信号，其设置足够高，使得在 GRPO 的优势归一化下，两条正确性相同但证据质量不同的轨迹能够获得可区分的奖励。协议遵循性（权重 0.05）充当轻量正则项，它推动策略遵循先验证后终结的顺序，但不会压过正确性信号。我们在表 3 中通过比较 $w_{\mathrm{fecv}} \in \{0.0, 0.15, 0.35, 0.50\}$ 的奖励权重敏感性消融来验证这一权重配置的稳健性。

各组成部分的具体定义如下：

**答案正确性奖励。** $R_{\mathrm{acc}}$ 对当前训练配置中保留的两类封闭式问题族的逐字段得分取平均：(i) *decision*，即异常是否存在与类别的二元匹配；(ii) *temporal grounding*，即预测异常区间与目标异常区间之间的 interval IoU。每一类先在内部求平均，然后 $R_{\mathrm{acc}}$ 取各有效问题族的等权平均。针对阶段摘要的开放式语义评分仅在测试时评估（第 5.4 节，QA Accuracy），而不作为训练奖励信号，从而使奖励模型免受 judge 噪声影响，并降低不同轨迹之间的梯度方差。

**证据忠实性奖励。** $R_{\mathrm{fecv}}$ 是一个按分支条件路由的分数，它根据轨迹 $\tau$ 所分配的难度分支 $b(\tau)$，将其送入三条奖励路径之一。之所以需要这种分支，是因为证据忠实性在正常回合和异常回合中具有不同的操作含义：正常回合因克制且有依据的引用而获得奖励，异常回合则因完整且在因果上必要的证据而获得奖励。各分支公式如下：

$$
R_{\mathrm{fecv}}(\tau)=
\begin{cases}
R_{\mathrm{easy}}(\tau), & b(\tau)=\texttt{easy\_normal}, \\
R_{\mathrm{susp}}(\tau), & b(\tau)=\texttt{suspicious\_normal}, \\
R_{\mathrm{online}}(\tau), & b(\tau)=\texttt{anomaly\_online\_core}.
\end{cases}
$$

对于 **easy normal** rollout，奖励偏好克制的搜索与稳定的验证：

$$
R_{\mathrm{easy}}
=
0.55 \, \mathrm{search\_restraint}
+ 0.25 \, \mathrm{window\_restraint}
+ 0.20 \, \mathrm{verifier\_trace}.
$$

此外，这些样本还通过 $0.20$ 的 easy-normal loss multiplier 被下调权重，以防止琐碎的正常案例主导梯度。

对于 **suspicious normal** rollout，奖励强调有依据的局部证据：

$$
R_{\mathrm{susp}}
=
0.35 \, \mathrm{search\_restraint}
+ 0.25 \, \mathrm{grounded\_local}
+ 0.20 \, \mathrm{query\_alignment}
+ 0.20 \, \mathrm{verifier\_trace},
$$

其中

$$
\mathrm{grounded\_local}
=
0.35 \, \mathrm{window\_restraint}
+ 0.20 \, \mathrm{provenance}
+ 0.25 \, (1-\mathrm{selected\_duration\_ratio})
+ 0.20 \, \mathrm{verifier\_trace}.
$$

对于遵循紧凑 `online_core` profile 的 **异常** rollout，奖励变为：

$$
R_{\mathrm{online}}
=
0.40 \, \mathrm{selected\_support}_{v2}
+ 0.20 \, \mathrm{trigger\_necessity}_{v2}
+ 0.15 \, \mathrm{verifier\_trace}
+ 0.15 \, \mathrm{stage\_coverage}
+ 0.10 \, \mathrm{parsimony}.
$$

其中，$\mathrm{selected\_support}_{v2} = 0.75 \cdot \mathrm{decision\_field\_support} + 0.25 \cdot \mathrm{stage\_text\_support}$，$\mathrm{trigger\_necessity}_{v2}$ 表示移除 trigger 证据后 decision field 下降的最大值，$\mathrm{stage\_coverage}$ 衡量所需阶段的恢复情况，$\mathrm{parsimony}=1-\lvert \text{minimal\_subset} \rvert/\lvert \text{full\_set} \rvert$。训练器仍保留针对非 `online_core` 异常 profile 的旧式兼容路径，但本文的方法阐述聚焦于上述三个奖励分支，因为它们定义了当前系统中的核心学习行为。

**结构化终结奖励。** $R_{\mathrm{protocol}}$ 将先验证后终结这一约束编码为一个三值信号：

$$
R_{\mathrm{protocol}} =
\begin{cases}
-1, & \text{if finalization is premature or never happens}, \\
+1, & \text{if verification explicitly recommends finalization and the policy finalizes}, \\
+0.75, & \text{otherwise}.
\end{cases}
$$

这里，“过早终结”指 `finalize_case` 先于 `verify_hypothesis` 发生。该设计直接惩罚过早终结，并奖励符合协议的案例闭环。

**事件链完整性。** 对于恢复得到的事件链 $C$ 与目标异常 $y$，阶段覆盖指标定义为：

$$
\operatorname{stage\_coverage}(C, y) =
\frac{\left|\left\{ s \in S_y : C_s \neq \varnothing \land \operatorname{temporally\_valid}(C_s) \right\}\right|}{|S_y|}.
$$

其中，$S_y \subseteq \{\mathrm{pre}, \mathrm{trg}, \mathrm{conf}\}$ 是目标异常 $y$ 中被标注为存在的阶段集合。对于仅存在 trigger 证据的瞬时异常，$S_y = \{\mathrm{trg}\}$，此时完整覆盖只要求恢复 trigger。这一自适应分母缓解了固定三阶段形式化与“并非所有异常都具有全部阶段”这一现实之间的张力。谓词 $\operatorname{temporally\_valid}(C_s)$ 要求阶段 $s$ 中的证据时刻在时间上有序，并且与异常区间一致。覆盖率为 $1.0$ 表示所有被标注的阶段都已由时间有效的证据填充。

**通过验证扰动定义的证据忠实性。** 当且仅当从所选证据集合 $E$ 中移除证据项 $e$ 会使验证结论从 sufficient 变为 insufficient 时，$e$ 被视为*必要的*：

$$
e \text{ is necessary}
\;\Leftrightarrow\;
\operatorname{verdict}(\text{claim}, E) = \mathtt{sufficient}
\land
\operatorname{verdict}(\text{claim}, E \setminus \{e\}) = \mathtt{insufficient}.
$$

不满足该条件的证据会被归类为冗余，并应触发一个带更严格阶段约束的定向 `seek_evidence` 调用。

我们注意到，并非所有异常都能被干净地分解为三个阶段。瞬时异常（例如突发爆炸）可能几乎没有前兆证据，而缓慢发展的异常（例如设备逐步退化）可能缺乏清晰的触发时刻。事件链形式化能够容纳这些情况：$\operatorname{stage\_coverage}$ 是一个软指标，策略会因为恢复了实际存在的阶段而得到奖励，而不会因缺失并不存在的阶段而受到惩罚。在实践中，MSAD 基准包含多种不同类型的异常，因此天然提供了事件链完整性要求上的变化。

只有当策略同时满足两个条件时，它才算成功。第一，它必须是**决策正确**的，即最终案例在异常是否存在、类别、时间与语义上都与目标异常一致。第二，它必须是**证据忠实**的，即所选证据子集在已实现的验证扰动下确实是必要且充分的。这正是为什么验证是动作空间的一部分，而不是事后附加步骤。一个系统如果基于错误或冗余证据给出了正确标签，它仍未真正解决异常理解问题。

## 4. SEEK-VAU：方法

SEEK-VAU 是一个面向视频异常理解的受约束工具使用策略，建立在 ReAct [11] 所确立的工具使用范式之上。在每一轮，策略都会基于对话状态、当前证据账本以及先前观察到的时间上下文进行推理，然后从四个可执行动作中选择其一：`scan_timeline`、`seek_evidence`、`verify_hypothesis` 或 `finalize_case`。这种动作设计是本方法的核心抽象。它迫使策略将广覆盖的时间搜索与证据承诺相分离，在生成结构化异常报告之前显式暴露其认为案例是否已经就绪。

```mermaid
flowchart TD
    accTitle: SEEK-VAU 方法概览
    accDescr: 用于事件链搜索、验证与结构化终结的工具使用策略。

    video["视频 + 任务查询"] --> policy["多模态策略"]
    policy --> scan["scan_timeline"]
    policy --> seek["seek_evidence"]
    policy --> verify["verify_hypothesis"]
    policy --> finalize["finalize_case"]

    scan --> plan["粗粒度时间图"]
    seek --> ledger["证据账本\nwindow ids / evidence ids / 阶段提示"]
    plan --> ledger
    ledger --> chain["恢复事件链完整性\n前兆 -> 触发 -> 确认 / 后果"]
    chain --> verify
    verify --> verdict["充分 / 不足 / 错配 / 冗余"]
    verdict --> action["scan_timeline / seek_evidence / finalize_case"]
    action --> policy
    finalize --> report["结构化异常案例\n类别 + 区间 + 证据 ids + 语义答案"]

    classDef core fill:#e7f0ff,stroke:#2f5aa8,stroke-width:2px,color:#183153
    classDef decision fill:#eef8ea,stroke:#397d2b,stroke-width:2px,color:#183b12
    classDef reportbox fill:#fff3df,stroke:#b26b00,stroke-width:2px,color:#5a3800

    class video,policy,scan,seek,verify,ledger,chain core
    class verdict,action decision
    class finalize,report reportbox
```

### 4.1 Agentic Event-Chain Search

第一个设计选择，是将搜索内化到策略之中。`scan_timeline` 执行广覆盖的时间搜索与粗定位，而 `seek_evidence` 则为当前假设收集更有针对性的证据。这一区分是有意为之：`scan_timeline` 不被视为证据，因为广义扫描不应与证据承诺混为一谈。当 feature cache 与 proposal runtime 被挂载后，`seek_evidence` 会变为查询引导式，并能够主动检索异常链中缺失的阶段，而不再依赖固定的 observation bundle。

这改变了观察预算的使用方式。在固定观察的 VAU 中，预算在推理开始之前就已经被消耗；而在 SEEK-VAU 中，预算是在推理过程中被消耗的。如果当前上下文揭示了 trigger 但没有 precursor，策略可以向后搜索；如果 aftermath 证据仍然缺失，它可以向前搜索。因此，事件链完整性不再只是一个标注 schema，而是一个 rollout 时的目标。

选择这四个动作，反映了对异常调查过程的一种最小而完整的分解。我们将 `scan_timeline` 与 `seek_evidence` 分开，是因为如果把粗粒度时间探索与证据承诺混为一体，就会模糊“我查看过这个区域”和“我将其作为支持性证据提交”之间的差异。在消融实验（表 3）中，将这两个动作合并成单一 `search` 操作会使 event-chain F1 降低 [TBD] 点，验证了这种分离在经验上是有益的。类似地，将 `verify_hypothesis` 设计为显式动作，而不是 `finalize_case` 内部的隐式步骤，会迫使策略在提交最终报告前显式暴露其不确定性。

**视觉预算约束。** SEEK-VAU 在固定视觉预算下运行：每次工具调用（`scan_timeline` 或 `seek_evidence`）至多从请求的时间窗口中采样 $K = 8$ 个关键帧。在一个 $T_{\max} = 10$ 轮的回合中，agent 最多检查 $10 \times 8 = 80$ 帧，这仍显著低于穷尽式观看。这一预算约束使得搜索-验证形式化并非平凡问题：agent 必须在 scan、seek 和 verify 动作之间战略性地分配其有限的视觉观察。不同于在推理开始前就耗尽全部帧预算的固定观察基线，SEEK-VAU 会自适应地分配预算，在模糊区域投入更多帧，在明显正常的片段上投入更少帧。我们将 mean inspected clip ratio 作为一个次级效率指标进行报告，以量化这种自适应分配。

### 4.2 策略内部的证据验证

第二个设计选择，是将验证做成显式的策略动作。`verify_hypothesis` 接收一个 claim 以及所选窗口、evidence ids 和结构化证据时刻，并返回一个结构化 verdict，例如 `sufficient`、`insufficient`、`misaligned` 或 `redundant`，同时给出建议的下一步。这个紧凑的验证接口使策略系统不仅能够表达“我认为发生了什么”，还能够表达“我当前的证据是否已经足以终结”。

这也是本方法与既有固定观察推理最明显的分野。只会不断累积支持性证据的策略，往往会倾向于过度收集与过度解释。相比之下，将验证作为动作，关注的是所选证据是否真的必要、是否已有一个更小的子集就足够、以及离题证据是否应当使当前 claim 失效。在训练时，这些检查由 oracle 标注支撑；在推理时，它们作为自一致性探针运作，虽然弱于 oracle verification，但足以阻止过早终结。在我们的表述中，这些检查不是可选的诊断工具，而是“忠实地理解一个异常案例”这一目标本身的一部分。

#### 4.2.1 Profiled Verification Protocol

当前实现**并不会**对每个样本统一施加一个六分支 verifier。相反，验证采用两类与奖励设计对齐的 profile family。

对于 **normal** 目标，策略会进入 `normal_skip_v1`。该路径有意跳过代价高昂的完整反事实重放，而是从 rollout trace 中重建已选窗口，然后将案例分类为 `easy_normal` 或 `suspicious_normal`。这一区分是当前训练配方的核心：easy-normal 案例被视为低信息量轨迹，而 suspicious-normal 案例则根据其是否保持克制、有依据且与 verifier 一致来计分。

对于 **anomaly** 目标，策略优先采用紧凑的 `online_core` profile。与其在主训练循环中实例化一个庞大的分支集合，`online_core` 只保留语义上必需的骨架：

- `decision`
- `covered_stages`
- `missing_required_stages`
- `stage_selected_moment_ids`
- `event_chain_summary`

这种紧凑表示已足以计算驱动当前异常奖励实现的连续诊断量：

1. 来自 `full_selected` 分支的**选中支持度**
2. 来自 `drop_trigger` 分支的**触发必要性**
3. 来自 `minimal_subset` 分支的**简约性**
4. 来自最新 verifier turn 以及恢复阶段元数据的**验证轨迹**与**阶段覆盖**

旧式的非 `online_core` 异常 profile 仍保留在代码库中以维持向后兼容，但它们不属于当前激活的训练叙事，其定义被推迟到附录中给出。

每次验证调用仍会返回一个 verdict，例如 `sufficient`、`insufficient`、`misaligned` 或 `redundant`，并伴随结构化的连续诊断量。在当前系统中，这些诊断量比单一的类别通过/失败比特更重要：它们决定了一条轨迹会被视为低权重的 easy-normal、会获得 suspicious-normal 的 grounded-evidence 奖励，还是会进入以异常为中心的 `online_core` 奖励路径。

**训练与推理的分离。** 在 RL 训练期间，来自验证的诊断量是基于结构化分支字段与 verifier 元数据计算得到的，从而确保奖励不是一种自由形式的文本启发式。在推理时，策略仍会针对自身选出的证据执行自一致性验证。尽管这种自一致性弱于 oracle verifier，但它仍然有助于为终结设置门控，并在交互循环中显式暴露证据不足状态。

```
算法 1：SEEK-VAU 推理回合
输入：视频 V、查询 q、轮次预算 T_max
初始化：证据账本 E ← ∅，时间图 M ← ∅，当前工作假设 c ← ∅，轮次 t ← 0
while t < T_max do:
    action ← π(s_t | history, E, M, c)  // 策略选择动作
    if action = scan_timeline:
        M ← M ∪ TemporalProposal(V, query=q)  // 粗粒度时间候选
        // 扫描结果提供线索，但不进入证据账本
    elif action = seek_evidence:
        e_new ← RetrieveEvidence(V, query=q, proposals=M)
        E ← E ∪ {e_new}  // 证据连同阶段提示一起写入账本
    elif action = verify_hypothesis:
        verdict, next_step ← VerifyEvidence(c, E)
        if next_step = "finalize": action_hint ← finalize_case
        elif next_step = "search": action_hint ← scan_timeline or seek_evidence
        // verdict 通过策略影响下一步动作选择，而不是作为独立动作执行
    elif action = finalize_case:
        return StructuredReport(category, interval, evidence_ids, explanation)
    t ← t + 1
return StructuredReport(...)  // 预算耗尽
```

需要注意的是，`verify_hypothesis` 会返回一个推荐的下一步，但策略保留完全自主性：该推荐会被编码进下一轮的状态中，而不是被自动执行。这样既保持了四动作空间的简洁性，又允许验证引导后续行为。

### 4.3 以 FECV 为基础的学习

训练目标分为两个阶段。监督微调并不直接模仿原始 oracle skeleton，而是由 teacher judge 将其改写为 **教师改写轨迹监督**，从而教授一种与协议一致的搜索-验证-终结交互模式。随后，强化学习沿着 rollout → FECV → reward → GRPO 的路径进行，并以第 3 节引入的按分支条件定义的 FECV 奖励作为核心监督信号。

**教师裁判。** teacher judge 是一个更强的冻结多模态模型（例如 GPT-4o 或 Qwen3-VL-32B），它将原始 oracle skeleton 改写为更干净的交互轨迹。Oracle skeleton 是从真实标注中导出的基于规则的动作序列；teacher judge 会纠正顺序错误、补充缺失的验证步骤，并提升证据选择质量。

在**默认奖励配置**下，主要奖励组成包括**答案正确性奖励**、**按分支定义的证据忠实性奖励**以及**结构化终结奖励**。可选的局部路由信号不再被当作单独的科学主张；相反，它们被折叠进真正起作用的奖励分支中：对于 suspicious normal，是 query alignment 与 grounded-local evidence；对于异常 `online_core` rollout，则是 stage coverage 与 verifier trace。整体优化目标依然简单：一条轨迹不仅应当是正确的，而且应当是**基于忠实证据而正确**的。

**已实现的分支结构。** 当前 RL 路径区分三个核心奖励分支。

- **`easy_normal`** 使用 `normal_skip_v1`，接收第 3 节中的低信息量 normal 奖励，并通过 $0.20$ 的 loss multiplier 有意降低权重。
- **`suspicious_normal`** 同样使用 `normal_skip_v1`，但会因为克制且有依据的证据选择而通过 `grounded_local` 与 verifier-trace 项获得奖励。
- **anomaly `online_core`** 使用第 4.2.1 节中的紧凑异常 profile，并通过 selected support、trigger necessity、verifier trace、stage coverage 与 parsimony 获得奖励。

在训练器层面，这些奖励路径又对应一个更轻量的优化 partition 方案：`easy_normal`、`hard_normal` 和 `anomaly`。标准组使用组内相对 z-score 优势。当一个 4-rollout 组的方差为零时，训练器会对非平凡 partition 回退到 EMA baseline；`easy_normal` 则有意保持为零。这一区分很重要：奖励分支定义了**奖励什么**，而训练器分区定义了**如何挽救或抑制塌缩组**。

**训练细节。** Oracle skeleton 通过将真实标注按规则对齐到四动作协议而构造：首先是一个覆盖整段视频的 `scan_timeline`，随后是针对每个已标注事件链阶段（precursor、trigger、confirmation）的 `seek_evidence` 调用，对收集到的证据执行一次 `verify_hypothesis`，最后使用真实标签调用 `finalize_case`。teacher judge（Qwen3-VL-32B）将这些机械式序列改写为更自然的交互轨迹，纠正动作顺序，加入具备上下文感知的搜索查询，并改进证据描述。SFT 使用标准的 next-token prediction，并只对 assistant-turn 施加 loss masking，system、user 和 tool message 都不计入 loss。当前激活的 GRPO 训练使用学习率 $5 \times 10^{-7}$、KL 系数 $0.0$、**每个 prompt 采样 4 个 generation**、每个回合最多 **10 轮**、**3 张 H200 GPU**、bf16，以及 **DeepSpeed ZeRO-2**。我们将 collapse-fix 视为已实现学习设计的一部分，而非事后的工程性补丁，因为它直接决定了以 FECV 为基础的监督能否产生可训练信号。

```mermaid
flowchart LR
    accTitle: 以 FECV 为基础的学习流程
    accDescr: 训练路径结合了教师裁判改写的 SFT 与以 FECV 为基础的强化学习。

    runtime["运行时回合"] --> oracle["Oracle 骨架"]
    oracle --> teacher["教师裁判改写\n教师改写轨迹监督"]
    teacher --> sft["步骤级 SFT"]
    sft --> rollout["策略 rollout"]
    rollout --> fecv["FECV 配置\n忠实性 + 验证诊断"]
    fecv --> reward["奖励聚合\n答案正确性 + 证据忠实性 + 结构化终结"]
    reward --> grpo["GRPO 更新"]

    reward --> local["可选辅助局部路由\n搜索 / 证据 / 查询 / 阶段 / 教师"]
    local --> grpo

    classDef data fill:#e8f1fb,stroke:#35608f,stroke-width:1.5px,color:#1d3557
    classDef train fill:#eef8e6,stroke:#4f772d,stroke-width:1.5px,color:#31572c
    classDef rewardbox fill:#fff4d6,stroke:#b7791f,stroke-width:1.5px,color:#7c2d12

    class runtime,oracle,teacher data
    class sft,rollout,grpo train
    class fecv,reward,local rewardbox
```"""


def main():
    root = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3")
    part1 = root / "tmp/pdfs/seek_vau_v12_zh/part1.md"
    part3 = root / "tmp/pdfs/seek_vau_v12_zh/part3.md"
    out = root / "docs/paper_drafts/seek_vau_neurips_v12_zh.md"

    text1 = part1.read_text(encoding="utf-8").strip()
    text3 = part3.read_text(encoding="utf-8").strip()
    final = "\n\n".join([text1, PART2.strip(), text3]) + "\n"

    final = final.replace("evidence moment IDs", "证据时刻 ID")
    final = final.replace("teacher judge", "教师裁判")
    final = final.replace("teacher 改写", "教师改写")
    final = final.replace("rollout", "rollout")
    final = final.replace("profile", "profile")
    final = final.replace("branch-profile", "branch-profile")

    out.write_text(final, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
