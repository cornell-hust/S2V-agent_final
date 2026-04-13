• 坦白说：现在这组比较“可以讲”，但还不够硬。
  如果论文只拿 8-frame fixed baseline 对比 14-turn / max 28 images context 的 S2V-Agent，审稿人很可能会质疑：

  > 你的提升到底来自 agentic search，还是因为模型看了更多帧？

  这个质疑是合理的。

  ## 结论先说

  ### 当前比较的合理性

  合理，但只能支持这个说法：

  > S2V-Agent improves over a fixed-observation baseline under an interactive evidence-acquisition protocol.

  也就是：我们的方法作为一个系统，比固定观察范式更强。

  ### 当前比较不够支持的说法

  不能强说：

  > S2V-Agent 在相同视觉预算下优于 baseline。

  因为严格讲，当前不相同：

  - baseline：固定 8 张图
  - agent：最多 14 turns，context 最多 28 张图，累计看过的视觉证据可能更多

  所以如果论文里写“fair visual budget comparison”，会被打。

  ———

  # 审稿人可能怎么攻击

  ## Attack 1: “You simply give your method more visual information”

  他们会说：

  > The fixed baseline only sees 8 frames, while S2V-Agent can inspect multiple windows over 14 turns. The improvement may come from visual coverage rather than agentic reasoning.

  这个是最危险的。

  ## Attack 2: “The baseline is too weak”

  他们会问：

  > Why not compare against dense uniform sampling, 28-frame fixed observation, or a stronger fixed-frame Qwen3-VL baseline?

  如果没有这些，baseline 会显得 strawman。

  ## Attack 3: “Event-Chain F1 naturally favors agentic methods”

  因为 agent 被训练/设计成 precursor-trigger-confirmation 结构，而 baseline 只是单次回答。审稿人可能说 metric 与方法绑定太强。

  ———

  # 怎么让比较变得有说服力

  我建议论文里至少分成两类实验。

  ## 1) Paradigm Comparison：范式对比

  这张表可以保留。

  目的不是证明同预算，而是证明：

  > Fixed-observation decoding is structurally limited; active search-to-verify gives better event-chain recovery.

  这里可以放：

  - Fixed Qwen3-VL-8B, 8 frames
  - Fixed Qwen3-VL-8B, 28 frames
  - prior VAU-style baselines
  - S2V-Agent

  但表述必须小心：

  > We compare against fixed-observation baselines under their non-interactive observation protocol. This evaluates the practical advantage of search-to-verify as an interaction paradigm, not a strictly equal-frame ablation.

  也就是承认：这是范式比较。

  ———

  ## 2) Budget-Controlled Comparison：预算控制对比

  这张表很关键，建议一定加。

  目标是回答：

  > 如果给 baseline 同样多的视觉预算，它还输吗？

  至少加这几行：

  | Method | Observation Policy | Total Visual Budget | Adaptive? | Verify? |
  |---|---|---:|---|---|
  | Fixed-8 | uniform preview | 8 frames | No | No |
  | Fixed-28 | uniform preview | 28 frames | No | No |
  | Fixed-112 | uniform dense / staged fixed sampling | up to 112 frame slots | No | No |
  | Random Search Agent | random windows | same visual calls as S2V | Yes, random | Optional/No |
  | S2V w/o Verify | learned/adaptive search | same turns | Yes | No |
  | S2V-Agent | learned/adaptive search | same turns | Yes | Yes |

  这里面最重要的是：

  ### Fixed-28 baseline

  因为 agent 的 context 上限是 28 张图。
  所以至少要让 baseline 一次性看 28 张 uniform frames。

  这能回答：

  > 是不是因为 baseline 只看 8 张太少？

  ### Fixed-112 baseline

  agent 最多 14 turns，每次视觉 tool 最多 8 帧。极端上限是 112 帧槽位。
  虽然实际上不是每轮都看 8 帧，而且上下文只保留 28 张，但 reviewer 可能会用这个上界攻击。

  所以可以做一个“generous fixed baseline”：

  - uniformly sample 112 frames
  - 或者 sample 28 frames but from 4 segments summarized as contact sheets
  - 让 Qwen3-VL 一次性看尽可能多

  如果 S2V 还赢，论证非常强。

  ### Random Search Agent

  这个特别重要。它控制了“多轮看更多帧”的因素。

  做法：

  - 也允许 14 turns
  - 也每次最多 8 帧
  - 但时间窗口随机选，或者固定均匀扫
  - 不根据当前 hypothesis 自适应选择

  如果 S2V > Random Search，说明提升不是只来自多看帧，而来自策略性搜索。

  ### S2V w/o Verify

  这对应 claim 2：

  - 有搜索
  - 没有验证
  - 看看 Evidence F1、Event-Chain F1、FECV/verification diagnostics 是否下降

  这能证明 verify action 不是摆设。

  ———

  # 我建议你们怎么改当前 baseline

  ## 当前 baseline：Fixed-8

  保留。它代表最朴素 fixed-observation baseline。

  但不能只报它。建议再加两个 baseline。

  ## 新增 baseline 1：Fixed-28

  配置：

  - num_preview_frames = 28
  - max_total_images = 28
  - 单次直接推理
  - no tools
  - no verify

  这和 agent 的上下文图像上限对齐。

  它回答：

  > If the fixed baseline sees as many images as the agent can retain in context, does it still fail to recover event chains?

  ## 新增 baseline 2：Fixed-112 或 Fixed-Budget-Search

  更强但可选。

  两种做法：

  ### 方案 A：Fixed-112 Dense Baseline

  - 一次性 uniform sample up to 112 frames
  - 如果模型上下文吃不下，就做 contact sheet / tiled image
  - 这是 generous baseline

  优点：强对照。
  缺点：实现和视觉编码方式可能引入新变量。

  ### 方案 B：Fixed-28 Multi-Segment Baseline

  - 把视频分成 4 个固定 segment
  - 每个 segment 取 7 帧
  - 总共 28 张
  - 单轮回答

  这个更干净，和 max_total_images=28 对齐。

  我更推荐先做 Fixed-28 uniform，够用了。

  ———

  # 论文里应该怎么表述才安全

  不要说：

  > We use the same visual budget as baselines.

  应该说：

  > We match the local observation granularity at 8 frames per observation. Since S2V-Agent is an interactive method, it is additionally constrained by a trajectory-level budget of 14 turns and a 28-image context cap. To isolate the effect of adaptive search from raw visual
  > coverage, we include fixed-observation baselines with 8 and 28 uniformly sampled frames, as well as budget-controlled ablations.

  这句话很稳。

  中文意思：

  - 单次观察粒度对齐：8 帧
  - agent 有交互预算：14 turns / 28 image context
  - 为排除“只是看更多帧”的质疑，加 fixed-28 和预算控制 ablation

  ———

  # 最建议的实验表结构

  ## Table 1: Main Results

  可以放最终系统比较。

  | Method | Visual Protocol | Existence Acc. | Temporal mIoU | QA Acc. | Event-Chain F1 | Evidence F1@3 |
  |---|---|---:|---:|---:|---:|---:|
  | Qwen3-VL Fixed-8 | 8 uniform preview frames | | | | | |
  | Qwen3-VL Fixed-28 | 28 uniform preview frames | | | | | |
  | S2V-Agent | 8/frame tool, 14 turns, 28 image context cap | | | | | |

  ## Table 2: Budget-Controlled Ablation

  专门回答预算问题。

  | Variant | Per-Observation Frames | Context Image Cap | Max Turns | Adaptive Search | Verify | Event-Chain F1 | Evidence F1@3 |
  |---|---:|---:|---:|---|---|---:|---:|
  | Fixed-8 | 8 | 8 | 1 | No | No | | |
  | Fixed-28 | 28 | 28 | 1 | No | No | | |
  | Uniform-Search | 8 | 28 | 14 | No | No/Yes | | |
  | S2V w/o Verify | 8 | 28 | 14 | Yes | No | | |
  | S2V-Agent | 8 | 28 | 14 | Yes | Yes | | |

  这张表会非常有说服力。

  ———

  # 如果资源有限，最少要补哪些

  如果时间紧，我建议最低限度补：

  1. Qwen3-VL Fixed-28 baseline
  2. S2V w/o Verify ablation
  3. 报告 mean inspected clip ratio
  4. 报告 mean_num_turns
  5. 报告 mean visual frames observed 或至少 num_visual_tool_calls × 8

  这样可以基本挡住 reviewer 的主要攻击。

  ———

  # 最终判断

  ## 现在这样直接投，风险中高

  因为 8-frame baseline vs multi-turn agent，预算差距明显。

  ## 加 Fixed-28 后，风险明显下降

  因为你能说：

  > Even when the fixed baseline sees as many images as our context cap, it lacks adaptive evidence acquisition and underperforms on event-chain/evidence metrics.

  ## 再加 Random/Uniform Search 后，说服力很强

  因为你能证明：

  > Not all additional visual budget helps; adaptive search and verification matter.

  ———

  # 我的建议

  你们应该把当前 baseline 扩展成至少两个 fixed baselines：

  - Qwen3-VL Fixed-8
  - Qwen3-VL Fixed-28

  然后把 agent 视觉预算写清楚：

  - local observation: 8 frames/tool
  - context cap: 28 images
  - max turns: 14
  - adaptive search: yes
  - verify: yes

  这样审稿人会更容易接受。