# 确定性工作流状态机（阶段 12）

AnalysisTeam 只使用下列分析状态，且不允许跳转或由 LLM 决定下一个阶段：

```text
CREATED
  → VALIDATING_INPUT
  → EXTRACTING_GRAPH
  → PREDICTING_RISK
  → ASSESSING_COMPLIANCE
  → REVIEWING
  → GENERATING_REPORT
  → COMPLETED
```

- 每个成功 Agent 回合会先形成结构化检查点，再与下一个合法状态一起持久化，并追加审计事件。
- `pause_after_step` 在某个已持久化检查点后停止；恢复时只运行该检查点之后的 Agent。
- Agent 失败时不保存失败回合为检查点，只保存安全 `ErrorRecord` 和重试计数。
- 每个阶段最多允许配置的重试次数；超过上限后转入 `FAILED`。后续修改型流程可使用 `NEEDS_MANUAL_REVIEW`，但不属于本分析链。
