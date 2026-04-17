# RL 阶段移除 `severity` / `counterfactual_type`

## Summary

- 范围仅限 active RL 训练链路。
- active RL 不再要求模型在 `finalize_case` 中生成 `severity` 或 `counterfactual_type`。
- active RL 的 reward / FECV 路径不再计算或消费这两个字段。
- 旧 artifact / 旧 payload 若仍包含这两个字段，训练侧宽松兼容读取，但会忽略它们。

## Boundaries

- 不改 SFT。
- 不改 fixed baseline eval。
- 不改论文草稿。
- 不全局删除共享 helper 中对这两个字段的支持，避免误伤非 RL 路径。

## Implementation Notes

- RL-only 输出契约收缩在 active RL rollout item 准备阶段完成，运行时重写 `tool_io.finalize_case_schema` 与 system prompt，无需重做已 materialized 数据。
- active RL reward 仅保留：
  - `accuracy_reward`
  - `fecv_evidence_faithfulness_reward`
  - `protocol_finalize_reward`
- `severity` / `counterfactual_type` 在 active RL 中不再进入：
  - `accuracy_reward`
  - `structured_oracle_v1` FECV summary 的训练相关字段
  - `finalized_case` 的最终保存结果（若 schema 未声明，则被剥离）

## Compatibility

- 旧 rollout / 旧 `finalize_case` payload 若仍带这两个字段，不报错。
- active RL 新路径仅在 schema 未声明时将其剥离，不影响旧数据读取。

## Verification

- `py_compile` 覆盖修改文件。
- RL 相关单测覆盖：
  - active RL reward 忽略 `severity` / `counterfactual_type`
  - structured-oracle FECV 可省略 `counterfactual_type_supported`
  - active RL rollout item 的 finalize schema / system prompt 收缩
  - `finalize_case` 在 RL schema 下会剥离这两个字段
