# 金融风险智能分析 Agent：技术架构

## 1. 项目目标

在既有“新闻采集 → LLM 标注/图谱抽取 → RGCN 风险分类”的成果上，构建一个可交互、可审计、可人工干预的金融风险智能分析应用。

系统处理新闻或已有案例，形成实体关系图谱、RGCN 风险预测、合规解释和最终风险报告；涉及数据修改、外部通知、敏感导出等操作时，必须经人工审批。

## 2. 技术约束

- Python 3.12。
- AutoGen 负责多 Agent 工作流编排。
- Streamlit 提供用户界面。
- LLM 使用 OpenAI 兼容接口，初始模型配置为 qwen3.7-max。
- 禁止 eval、exec、compile、Shell、Docker、任意 Python 或用户脚本执行。
- 所有高风险操作必须先经人工审批。
- 所有数据修改必须写入追加式审计日志。
- 每次修改执行后必须重新读取权威数据源进行校验。
- 既有工业项目代码、数据集和模型权重只能通过受控适配器读取；应用不得直接导入或执行历史脚本。
- 历史资产接入前必须完成凭据轮换、数据集清单和模型可复现性校验。

## 3. 架构总览

~~~mermaid
flowchart TD
    UI["Streamlit 用户界面"] --> APP["应用服务层"]
    APP --> CTRL["确定性 Workflow Controller"]
    CTRL --> ANALYSIS["AutoGen AnalysisTeam：RoundRobinGroupChat"]

    ANALYSIS --> INTAKE["Intake Agent：输入整理"]
    INTAKE --> GRAPH["Graph Agent：图谱抽取"]
    GRAPH --> RGCN["RGCN Agent：风险预测"]
    RGCN --> COMPLIANCE["Compliance Agent：合规解释"]
    COMPLIANCE --> CRITIC["Critic Agent：证据复核"]
    CRITIC --> DRAFT["Report Draft Agent：报告草稿"]

    CTRL --> POLICY["Policy Engine：风险分级"]
    POLICY -->|"只读或低风险"| FINAL["Final Report Agent"]
    POLICY -->|"高风险"| APPROVAL["人工审批中心"]
    APPROVAL -->|"拒绝"| FINAL
    APPROVAL -->|"批准"| MUTATION["AutoGen MutationTeam：RoundRobinGroupChat"]
    MUTATION --> EXECUTOR["Executor Agent：受控执行"]
    EXECUTOR --> VERIFY["Verifier Agent：重新校验"]
    VERIFY -->|"通过"| FINAL
    VERIFY -->|"失败"| RECOVERY["回滚或人工处理"]
    RECOVERY --> FINAL

    EXECUTOR --> GATEWAY["Safe Tool Gateway"]
    GATEWAY --> DATA["数据集 / 图谱 / RGCN 模型 / 案例库"]

    APP --> AUDIT["不可变审计日志"]
    CTRL --> AUDIT
    APPROVAL --> AUDIT
    GATEWAY --> AUDIT
    VERIFY --> AUDIT
~~~

## 4. 编排策略

### 4.1 总体原则

采用“AutoGen RoundRobinGroupChat + 外层确定性状态机”的混合方式。

- AutoGen 负责 Agent 间的顺序协作、上下文传递和 Agent 状态保存/恢复。
- 外层 Workflow Controller 负责阶段转换、审批暂停、恢复、重试上限和安全决策。
- LLM 不拥有数据库直连能力，也不能自行决定是否执行高风险操作。
- 所有工具必须通过 Safe Tool Gateway 调用。

### 4.2 RoundRobinGroupChat

本项目的主分析流程固定：输入整理、图谱抽取、RGCN 预测、合规解释、证据复核、生成报告草稿。采用固定顺序，避免 LLM 决定下一位 Agent 而导致流程不稳定。

~~~text
AnalysisTeam = RoundRobinGroupChat(
    Intake Agent
    → Graph Agent
    → RGCN Agent
    → Compliance Agent
    → Critic Agent
    → Report Draft Agent
)
~~~

- 正常分析设置 max_turns=6，每个 Agent 执行一次。
- 单个 Agent 内部允许有限次数的白名单工具调用；设置严格超时、重试和最大工具调用次数。
- Agent 之间传递经过校验的结构化对象或产物引用，不以自由文本保存业务状态。

### 4.3 两个 Team 的划分

不要将审批、执行、校验和分析全部放进一个 RoundRobin 循环。否则无需修改数据时，执行 Agent 仍会被轮询；审批拒绝后也难以安全跳过执行阶段。

因此拆分为：

1. AnalysisTeam：固定完成风险分析。
2. Workflow Controller：根据操作风险决定直接报告、进入审批或终止。
3. MutationTeam：仅在审批通过后启动。
4. Final Report Agent：根据分析、审批、执行与校验结果生成最终报告。

~~~text
MutationTeam = RoundRobinGroupChat(
    Executor Agent
    → Verifier Agent
    → Final Report Agent
)
~~~

### 4.4 审批暂停与恢复

高风险操作不使用长期阻塞的控制台输入，也不在 Streamlit 进程内等待。

~~~text
发现高风险操作
→ 生成审批单
→ 绑定 ActionProposal 哈希
→ 保存业务状态与 AutoGen Team 状态
→ 状态设为 AWAITING_APPROVAL
→ 本次运行结束
→ Streamlit 显示批准/拒绝
→ 验证审批仍有效
→ 批准后启动 MutationTeam
~~~

任何操作参数、目标资源或预期影响发生变化时，ActionProposal 哈希变化，原审批自动失效。

## 5. Agent 职责

| Agent | 职责 | 允许的能力 |
|---|---|---|
| Coordinator / Controller | 管理状态与阶段，不产生业务修改 | 任务状态读写、风险策略判断 |
| Intake Agent | 校验输入、生成不可变输入快照 | 只读数据工具、输入校验 |
| Graph Agent | 提取实体、关系、事件和证据位置 | 图谱抽取工具、图谱读取工具 |
| RGCN Agent | 调用现有 RGCN 推理，产出风险概率 | RGCN 推理工具、模型元数据读取 |
| Compliance Agent | 基于证据提出合规解释与行动建议 | 只读检索工具、结构化建议输出 |
| Critic Agent | 检查证据充分性、模型和解释一致性 | 只读校验工具、退回建议 |
| Report Draft Agent | 生成分析草稿 | 只读结果、模板渲染 |
| Executor Agent | 通过受控网关执行已批准操作 | 仅允许获批的修改工具 |
| Verifier Agent | 重新读取目标数据，验证实际效果 | 只读验证工具、回滚申请 |
| Final Report Agent | 汇总最终结论、审批和校验结果 | 只读结果、模板渲染 |

Agent 不能直接连接数据库、文件系统或外部服务；必须调用经过注册和校验的工具。

## 6. 状态机

### 6.1 状态枚举

~~~text
CREATED
VALIDATING_INPUT
EXTRACTING_GRAPH
PREDICTING_RISK
ASSESSING_COMPLIANCE
REVIEWING
AWAITING_APPROVAL
EXECUTING
VERIFYING
GENERATING_REPORT
COMPLETED
REJECTED
FAILED
NEEDS_MANUAL_REVIEW
~~~

### 6.2 WorkflowState

~~~text
WorkflowState
├── schema_version
├── task_id
├── session_id
├── correlation_id
├── status
├── current_phase
├── created_at / updated_at
├── input
│   ├── source_type
│   ├── article_id
│   ├── title
│   ├── content_ref
│   ├── content_hash
│   └── source_metadata
├── graph_artifact
│   ├── artifact_id
│   ├── entities
│   ├── relations
│   ├── evidence_refs
│   └── schema_version
├── risk_prediction
│   ├── model_version
│   ├── model_artifact_ref
│   ├── checkpoint_hash
│   ├── feature_pipeline_version
│   ├── node_type_vocab_hash / relation_type_vocab_hash
│   ├── class_order
│   ├── credit_probability
│   ├── market_probability
│   ├── operational_probability
│   ├── fraud_probability
│   └── prediction_hash
├── compliance_assessment
├── proposed_actions[]
├── approvals[]
├── execution_receipts[]
├── verification_results[]
├── retry_counters
├── error_records[]
├── autogen_state_ref
├── audit_event_refs[]
└── final_report_ref
~~~

业务状态必须持久化；Streamlit 的 session_state 只保存当前页面会话信息和任务编号。

### 6.3 ActionProposal

每个可能修改数据或对外产生影响的建议，都先转化为不可直接执行的结构化操作提案：

~~~text
action_id
tool_name
operation_type
risk_level
target_resource
validated_arguments
reason
evidence_refs
expected_effect
preconditions
requires_approval
idempotency_key
proposal_hash
proposed_by
~~~

## 7. 安全设计

### 7.1 禁止任意代码执行

系统不注册、不实现、不调用以下能力：

- eval、exec、compile。
- AutoGen 的 Python 代码执行工具。
- Shell、终端、Docker 命令执行工具。
- 用户上传脚本、动态模块导入或动态插件加载。
- 任意 SQL 执行工具。
- 由 LLM 动态生成的新工具。

所有工具均为预先注册的 Python 函数，参数通过 Pydantic 严格校验；数据访问通过固定 Repository 方法和参数化查询完成。

### 7.2 操作风险分级

| 风险级别 | 示例 | 执行策略 |
|---|---|---|
| READ_ONLY | 读取新闻、图谱、模型预测、审计记录 | 自动执行并审计 |
| LOW | 保存临时草稿、本地分析缓存 | 自动执行、审计、重新校验 |
| MEDIUM | 修改人工标签、更新案例状态 | 由策略配置决定是否审批 |
| HIGH | 删除数据、覆盖图谱、发送外部警报、敏感导出、替换模型 | 必须人工审批 |
| PROHIBITED | 任意代码、命令、SQL 或脚本执行 | 永久拒绝 |

### 7.3 人工审批

审批页面必须显示：

- 操作名称和目标数据。
- 修改前后预期值。
- 风险等级和风险原因。
- 支持该建议的证据。
- 操作参数摘要。
- 是否可回滚。
- 审批有效期。

审批记录至少包含：审批人、时间、决定、备注、绑定的 proposal_hash、失效时间。

### 7.4 历史资产安全边界

- 历史采集、LLM、Neo4j、训练和 Notebook 脚本均视为不可信的离线资产，不得作为应用运行时模块导入。
- 发现历史源码、Notebook 或产物中存在明文凭据时，必须先撤销或轮换凭据；仅从版本控制中删除并不足以解除风险。
- 旧数据集、图谱 JSON 和模型权重进入应用前，必须登记来源路径、内容哈希、创建时间、Schema/特征版本和许可或敏感级别。
- 旧模型权重仅在模型包清单完整、校验哈希匹配且兼容性测试通过后才能用于只读推理；否则只作为研究或演示资产。
- Safe Tool Gateway 只暴露新项目注册的 Repository 和适配器函数，不暴露历史脚本、Shell 命令、Notebook 或 Neo4j 导入器。

## 8. 审计与校验

### 8.1 审计日志

所有读取、Agent 结论、操作提案、审批、执行、失败、回滚和校验都记录为追加式事件。数据修改必须由审计服务包裹，禁止绕过。

~~~text
AuditEvent
├── event_id
├── sequence_number
├── task_id / correlation_id
├── event_type
├── actor_type / actor_id
├── timestamp
├── target_resource
├── action_id / approval_id
├── arguments_digest
├── before_digest / after_digest
├── result / error_code
├── previous_event_hash
└── event_hash
~~~

审计表仅允许追加写入；使用事件哈希链增强篡改可检测性。日志中不得保存 API Key、完整敏感内容或未经脱敏的凭据。

### 8.2 修改后的重新校验

工具返回成功不代表操作完成。每次修改都必须执行：

1. 写入修改前版本和内容哈希。
2. 使用幂等键执行获批操作。
3. 保存执行凭证和实际修改范围。
4. 由 Verifier 从权威数据源重新读取目标数据。
5. 检查目标、字段、版本、预期效果和业务约束。
6. 成功时写入 VERIFIED 审计事件。
7. 失败时可回滚则执行补偿；否则进入 NEEDS_MANUAL_REVIEW。

校验通过与否由确定性代码决定，LLM 只能解释校验结论，不能替代校验逻辑。

## 9. 工具清单

### 9.1 只读工具

| 工具 | 用途 |
|---|---|
| load_article | 读取指定新闻或案例 |
| search_dataset | 按实体、时间和风险类型检索 |
| load_graph_artifact | 读取已有知识图谱 |
| extract_graph | 调用固定图谱抽取流程 |
| predict_risk_rgcn | 调用现有 RGCN 模型 |
| get_entity_history | 查询实体历史风险 |
| get_model_metadata | 获取模型版本、阈值和指标 |
| get_case_history | 查询案例、审批和处理历史 |
| get_audit_trail | 查询审计事件 |
| validate_evidence | 检查结论与原文证据的映射 |

### 9.2 修改工具

下列工具只通过 Safe Tool Gateway 暴露：

| 工具 | 风险 |
|---|---|
| save_analysis_draft | 低 |
| persist_graph_artifact | 中 |
| update_case_label | 中 |
| update_case_status | 中 |
| create_review_case | 中 |
| add_to_watchlist | 高 |
| export_sensitive_report | 高 |
| send_risk_alert | 高 |
| replace_active_model | 高 |
| delete_case_data | 高 |

### 9.3 治理工具

| 工具 | 用途 |
|---|---|
| validate_tool_arguments | 校验工具参数和数据模型 |
| classify_operation_risk | 判定操作风险等级 |
| create_approval_request | 创建审批并暂停任务 |
| verify_approval | 校验审批人、有效期、操作哈希 |
| execute_allowed_action | 执行获准操作 |
| verify_mutation | 重新读取并验证修改结果 |
| rollback_or_compensate | 校验失败后的恢复 |
| append_audit_event | 追加审计事件 |

## 10. LLM 与既有模型集成

### 10.1 LLM 适配层

所有 LLM 调用通过 LLMProvider 统一封装：

- OpenAI 兼容 base_url、API Key、模型名均从环境变量读取。
- 初始模型为 qwen3.7-max。
- 密钥不进入代码、日志、审计事件或任务状态。
- 应用启动时执行连通性、Tool Calling、结构化输出、超时和上下文长度自检。
- 保留模型能力配置和回退策略，避免接口差异影响核心流程。

### 10.2 RGCN 适配层

RGCN 不由 LLM 替代。RGCN Agent 通过确定性推理适配器调用既有模型，返回：

- 模型版本。
- 四类风险概率：信用、市场、操作、欺诈。
- 使用阈值。
- 图谱特征/输入产物哈希。
- 推理时间和异常状态。

每个可部署模型必须以版本化模型包交付，而不是单独的 checkpoint。模型包至少包含：权重及其哈希、网络结构参数、风险类别顺序、节点/关系词表及哈希、特征构建器与嵌入模型版本、校准阈值、训练/验证数据集哈希和评估指标。适配器拒绝元数据不完整、类别顺序不一致或特征维度不匹配的模型包。

LLM 负责图谱抽取、解释、建议和报告；知识图谱与 RGCN 保持为核心确定性能力。

## 11. 数据与持久化

- 本地 MVP 使用 SQLite，开启事务并通过 Repository 接口访问。
- 多用户或生产环境切换 PostgreSQL。
- 审计日志、任务状态、审批记录、操作凭证、校验结果独立持久化。
- 数据集、模型、图谱和报告均保存在项目目录内部。
- 原始数据不可原地覆盖；所有转换形成可追溯的新版本或产物引用。
- 历史数据、图谱和模型以只读导入清单管理；清单记录来源、导入时间、内容哈希、Schema 版本和转换产物引用。
- 训练、验证、测试切分及词表只能由训练集建立；验证和测试数据不得参与词表、特征或阈值拟合。

## 12. 推荐目录结构

~~~text
financial_risk_agent/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── case_analysis.py
│   │   ├── approvals.py
│   │   ├── task_history.py
│   │   └── audit_trail.py
│   └── components/
├── src/
│   └── financial_risk_agent/
│       ├── config/
│       ├── core/
│       │   ├── enums.py
│       │   ├── schemas.py
│       │   ├── state.py
│       │   └── security.py
│       ├── orchestration/
│       │   ├── team.py
│       │   ├── workflow.py
│       │   ├── selector.py
│       │   ├── termination.py
│       │   └── state_store.py
│       ├── agents/
│       │   ├── coordinator.py
│       │   ├── intake.py
│       │   ├── graph.py
│       │   ├── rgcn.py
│       │   ├── compliance.py
│       │   ├── critic.py
│       │   ├── report.py
│       │   ├── executor.py
│       │   └── verifier.py
│       ├── tools/
│       │   ├── registry.py
│       │   ├── gateway.py
│       │   ├── readonly/
│       │   └── mutations/
│       ├── services/
│       │   ├── policy.py
│       │   ├── approval.py
│       │   ├── audit.py
│       │   ├── execution.py
│       │   └── verification.py
│       ├── domain/
│       │   ├── risk/
│       │   ├── knowledge_graph/
│       │   ├── compliance/
│       │   └── reporting/
│       ├── integrations/
│       │   ├── llm_client.py
│       │   ├── rgcn_adapter.py
│       │   ├── graph_adapter.py
│       │   └── legacy/
│       │       ├── dataset_adapter.py
│       │       ├── graph_artifact_adapter.py
│       │       └── model_bundle.py
│       └── storage/
│           ├── database.py
│           ├── repositories/
│           └── migrations/
├── prompts/
│   ├── graph/
│   ├── compliance/
│   ├── critic/
│   └── report/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── samples/
├── artifacts/
│   ├── models/
│   │   └── <model_version>/manifest.json
│   ├── graphs/
│   └── reports/
├── var/
│   ├── runtime/
│   ├── audit/
│   └── exports/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   ├── workflow/
│   └── end_to_end/
└── docs/
    ├── architecture.md
    ├── security.md
    ├── state_machine.md
    └── tool_catalog.md
~~~

## 13. 实施顺序

1. 建立项目骨架、配置管理和 Pydantic 状态模型。
2. 清理历史凭据，并为旧数据、图谱和模型建立只读资产清单。
3. 实现任务持久化和追加式审计日志。
4. 接入历史新闻、图谱产物和模型包的只读适配器；先验证转换和可复现性，再启用推理。
5. 实现 Safe Tool Gateway、LLM Provider 和只读 AnalysisTeam。
6. 实现 Streamlit 案例分析、任务历史和报告展示。
7. 实现 Policy Engine、人工审批、受控修改和执行后重新校验。
8. 增加安全、集成和端到端测试。

## 14. 当前决策记录

- 编排方案：采用 RoundRobinGroupChat，而非让 LLM 动态决定流程。
- Team 划分：AnalysisTeam 与 MutationTeam 分离。
- 高风险操作：审批前不执行；审批绑定操作哈希。
- 校验：执行后必须从权威数据源重新读取验证。
- 安全边界：Agent 不直连数据源，不提供任意代码执行能力。
- 业务差异化：LLM 负责理解和解释，知识图谱与 RGCN 保持为核心确定性能力。
- 历史资产边界：旧脚本不进入运行时；只复用经过测试的纯转换逻辑、数据格式和模型定义。
- 模型治理：模型权重必须与特征、词表、类别顺序、阈值和数据版本共同发布。
