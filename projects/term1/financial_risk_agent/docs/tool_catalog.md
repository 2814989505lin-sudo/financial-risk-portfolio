# 工具目录

## 已实现的只读工具

只读工具由 `ToolRegistry` 与 `SafeToolGateway` 显式注册。它们只接受 Pydantic 参数模型，写入安全审计事件，且拒绝 Shell、SQL、代码执行、脚本和未注册工具。

## 已实现的受控修改工具

| 工具 | 风险级别 | 固定参数 | 审批 | 幂等 | 执行后校验 | 状态 |
|---|---|---|---|---|---|---|
| `update_case_status` | MEDIUM | `case_id`、`new_status`、`expected_version` | 必须 | 是 | 重读状态、版本和内容哈希 | 已实现 |
| `update_case_label` | MEDIUM | `case_id`、`new_label`、`expected_version` | 必须 | 是 | 重读标签、版本和内容哈希 | 已实现（阶段 18 本轮） |

## 未实现工具

`persist_graph_artifact`、`create_review_case`、`add_to_watchlist`、`export_sensitive_report`、`send_risk_alert`、`replace_active_model` 和 `delete_case_data` 当前没有执行入口，也不会注册到工具表。后续每轮只能新增一个工具，并同时提供独立参数模型、风险策略、审批检查、执行后校验和测试。

## 永久禁止能力

不得提供 `eval`、`exec`、任意 Python、Shell、SQL、Docker、用户脚本、动态插件或动态导入工具。它们不会因为审批而变为可执行能力。
