# 金融风险智能分析 Agent：分阶段开发计划

本文基于 `docs/architecture.md` 拆分实现工作。执行原则：每次只实现一个阶段；该阶段的定向测试和已有回归测试通过后，才进入下一阶段。

## 阶段 1：项目骨架与基础测试环境

**目标**  
输入：当前项目目录和架构文档。  
输出：可安装、可导入、可运行测试的 Python 3.12 项目骨架；不实现业务功能。

**需要创建或修改的文件**

- `pyproject.toml`
- `README.md`
- `.env.example`
- `.gitignore`
- `src/financial_risk_agent/__init__.py`
- `src/financial_risk_agent/config/settings.py`
- `tests/conftest.py`
- `tests/unit/test_settings.py`

**测试方式**

```bash
python -m pytest tests/unit/test_settings.py
python -c "import financial_risk_agent"
```

**验收标准**

- 项目可在 Python 3.12 环境安装和导入。
- 测试框架可运行。
- 配置可从环境变量读取；缺少必需配置时返回明确错误。
- API Key 不存在默认值，也不写入日志。
- 不包含 Shell、代码执行或动态插件能力。

**依赖的前置阶段**  
无。

## 阶段 2：核心枚举与业务数据模型

**目标**  
输入：状态枚举、`WorkflowState`、`ActionProposal`、审批和执行数据结构。  
输出：经过严格校验、可序列化和反序列化的 Pydantic 模型。

**需要创建或修改的文件**

- `src/financial_risk_agent/core/enums.py`
- `src/financial_risk_agent/core/schemas.py`
- `src/financial_risk_agent/core/state.py`
- `src/financial_risk_agent/core/security.py`
- `tests/unit/test_schemas.py`
- `tests/unit/test_state.py`
- `tests/unit/test_security.py`

**测试方式**

```bash
python -m pytest tests/unit/test_schemas.py tests/unit/test_state.py tests/unit/test_security.py
```

**验收标准**

- 架构中的全部状态和风险等级都有枚举定义。
- 非法概率、空任务编号、未知状态和非法工具名会被拒绝。
- 状态对象可完成 JSON 往返转换。
- `proposal_hash` 和内容哈希具有确定性。
- 序列化结果不包含密钥或凭据。

**依赖的前置阶段**  
阶段 1。

## 阶段 2.5：历史资产安全清理与导入清单

**目标**  
输入：工业项目一中已有的采集、标注、图谱、训练脚本、数据集和模型权重。  
输出：已轮换或撤销的历史凭据，以及仅记录元数据和内容哈希的只读资产清单；本阶段不导入或执行历史脚本。

**需要创建或修改的文件**

- `docs/security.md`
- `docs/legacy_asset_inventory.md`
- `data/manifests/legacy_assets.json`
- `tests/security/test_legacy_asset_manifest.py`

**测试方式**

```bash
python -m pytest tests/security/test_legacy_asset_manifest.py
```

并以不输出凭据内容的扫描确认：新项目目录、`.env.example`、日志模板和资产清单均不含明文密钥。

**验收标准**

- 已发现的 Guardian、LLM/OpenAI 兼容服务和 Neo4j 历史凭据已由负责人撤销或轮换。
- 清单为每个可接入数据集、图谱 JSON 和模型权重记录来源、内容哈希、创建时间、Schema 或特征版本及敏感级别。
- 清单不保存 API Key、密码、原始敏感文本或完整凭据。
- 新项目不导入、执行或注册任何历史 Python、Shell、Notebook 或 Neo4j 导入脚本。

**依赖的前置阶段**  
阶段 1、阶段 2；凭据轮换需要具备外部服务管理权限的负责人执行。

## 阶段 3：SQLite 数据库与任务状态持久化

**目标**  
输入：合法的 `WorkflowState`。  
输出：能够创建、读取和更新任务状态的 SQLite Repository；暂不接入 Agent。

**需要创建或修改的文件**

- `src/financial_risk_agent/storage/database.py`
- `src/financial_risk_agent/storage/repositories/task_repository.py`
- `src/financial_risk_agent/orchestration/state_store.py`
- `src/financial_risk_agent/storage/migrations/001_tasks.sql`
- `tests/integration/test_task_repository.py`
- `tests/integration/test_state_store.py`

**测试方式**

```bash
python -m pytest tests/integration/test_task_repository.py tests/integration/test_state_store.py
```

**验收标准**

- 可创建任务，并按 `task_id` 重新读取。
- 状态更新在事务中完成。
- 非法状态转换不能写入数据库。
- 应用重新启动后任务仍可恢复。
- Streamlit 会话不是业务状态的唯一存储位置。

**依赖的前置阶段**  
阶段 2、阶段 2.5。

## 阶段 4：追加式审计日志与哈希链

**目标**  
输入：读取、状态变化、Agent 结论或操作事件。  
输出：仅追加写入、可验证完整性的审计事件链。

**需要创建或修改的文件**

- `src/financial_risk_agent/services/audit.py`
- `src/financial_risk_agent/storage/repositories/audit_repository.py`
- `src/financial_risk_agent/storage/migrations/002_audit_events.sql`
- `tests/unit/test_audit_hashing.py`
- `tests/integration/test_audit_repository.py`
- `tests/security/test_audit_redaction.py`

**测试方式**

```bash
python -m pytest tests/unit/test_audit_hashing.py tests/integration/test_audit_repository.py tests/security/test_audit_redaction.py
```

**验收标准**

- 审计事件只允许追加，不提供更新和删除接口。
- `sequence_number` 连续且任务内唯一。
- 每个事件正确引用前一事件哈希。
- 修改历史事件后，完整性检查必然失败。
- API Key、令牌和完整敏感内容不会进入日志。

**依赖的前置阶段**  
阶段 3。

## 阶段 5：历史新闻与案例只读适配器

**目标**  
输入：文章编号、案例编号、历史 Guardian CSV 或测试数据文件，以及阶段 2.5 的资产清单。  
输出：标准化、不可变的文章输入快照、内容哈希和导入产物引用；历史文件保持只读。

**需要创建或修改的文件**

- `src/financial_risk_agent/integrations/legacy/dataset_adapter.py`
- `src/financial_risk_agent/storage/repositories/article_repository.py`
- `src/financial_risk_agent/tools/readonly/load_article.py`
- `src/financial_risk_agent/tools/readonly/search_dataset.py`
- `data/samples/sample_articles.json`
- `data/manifests/legacy_assets.json`
- `tests/unit/test_article_normalization.py`
- `tests/integration/test_article_tools.py`
- `tests/integration/test_legacy_dataset_adapter.py`

**测试方式**

```bash
python -m pytest tests/unit/test_article_normalization.py tests/integration/test_article_tools.py tests/integration/test_legacy_dataset_adapter.py
```

**验收标准**

- 可读取样例新闻并生成标准输入模型。
- 相同内容始终产生相同哈希。
- 缺失内容、错误编码和未知文章编号有明确错误。
- 工具不能修改源数据。
- 历史列名可映射为统一的文章 Schema，且数据集版本和来源可追溯。
- 每次读取都会产生审计事件。

**依赖的前置阶段**  
阶段 2.5、阶段 4。

## 阶段 6：既有知识图谱产物转换适配器

**目标**  
输入：阶段 5 的文章快照，以及历史 `nodes / edges / docs` 或文档级图 JSON。  
输出：符合统一 Schema 的实体、关系、证据引用和图谱谱系信息；暂不调用 LLM。

**需要创建或修改的文件**

- `src/financial_risk_agent/domain/knowledge_graph/models.py`
- `src/financial_risk_agent/integrations/legacy/graph_artifact_adapter.py`
- `src/financial_risk_agent/tools/readonly/load_graph_artifact.py`
- `tests/fixtures/graph_cases/`
- `tests/unit/test_graph_schema.py`
- `tests/integration/test_legacy_graph_artifact_adapter.py`

**测试方式**

```bash
python -m pytest tests/unit/test_graph_schema.py tests/integration/test_legacy_graph_artifact_adapter.py
```

**验收标准**

- 每个实体和关系均符合 Schema。
- 历史 `nodes / edges / docs` 与文档级图均可转换到新 Schema。
- 原始证据缺失的历史共现边显式标记为低证据强度，不能伪装成原文证据。
- 同一固定输入可生成相同的转换产物和谱系哈希。
- 转换过程不会执行或导入历史脚本。

**依赖的前置阶段**  
阶段 5。

## 阶段 6B：图谱抽取契约与证据定位

**目标**  
输入：阶段 5 的文章快照和模拟的结构化抽取响应。  
输出：用于后续 Graph Agent 的实体、关系、原文证据位置和拒绝规则；本阶段不连接真实 LLM。

**需要创建或修改的文件**

- `src/financial_risk_agent/integrations/graph_adapter.py`
- `src/financial_risk_agent/tools/readonly/extract_graph.py`
- `prompts/graph/`
- `tests/unit/test_graph_evidence_locator.py`
- `tests/integration/test_graph_extraction_contract.py`

**测试方式**

```bash
python -m pytest tests/unit/test_graph_evidence_locator.py tests/integration/test_graph_extraction_contract.py
```

**验收标准**

- 抽取响应必须通过统一 GraphArtifact Schema。
- 关键关系必须能定位到输入快照的原文范围；无法定位时被拒绝或降级。
- 输入长度、实体数量、关系数量和重试预算均有上限。
- 测试通过模拟 Provider 离线运行。

**依赖的前置阶段**  
阶段 5、阶段 6。

## 阶段 6C：确定性图谱验证与清理

**目标**  
输入：阶段 6 或 6B 的 GraphArtifact。  
输出：经实体类型、谓词类型、置信度和证据规则验证后的图谱，以及独立的清理记录。

**需要创建或修改的文件**

- `src/financial_risk_agent/services/graph_validation.py`
- `src/financial_risk_agent/domain/knowledge_graph/rules.py`
- `tests/unit/test_graph_validation_rules.py`
- `tests/integration/test_graph_validation.py`

**测试方式**

```bash
python -m pytest tests/unit/test_graph_validation_rules.py tests/integration/test_graph_validation.py
```

**验收标准**

- 非法实体类型、主体/客体类型组合和低置信度强关系会被拒绝或显式降级。
- 降级、删除和类型修正均产生结构化清理记录，不修改原始图谱产物。
- 校验结论完全由确定性代码产生。

**依赖的前置阶段**  
阶段 6、阶段 6B。

## 阶段 7：RGCN 模型包封装与兼容性验证

**目标**  
输入：既有 RGCN 权重、模型定义、图数据样本和阶段 2.5 的资产清单。  
输出：带完整元数据的、可信的只读模型包；元数据缺失或不兼容的旧 checkpoint 被明确拒绝。

**需要创建或修改的文件**

- `src/financial_risk_agent/core/schemas.py`
- `src/financial_risk_agent/core/state.py`
- `src/financial_risk_agent/domain/risk/models.py`
- `src/financial_risk_agent/integrations/legacy/model_bundle.py`
- `artifacts/models/<model_version>/manifest.json`
- `tests/fixtures/rgcn/`
- `tests/unit/test_model_manifest.py`
- `tests/integration/test_legacy_model_compatibility.py`

**测试方式**

```bash
python -m pytest tests/unit/test_model_manifest.py tests/integration/test_legacy_model_compatibility.py
```

**验收标准**

- 模型包包含 checkpoint 哈希、网络结构、类别顺序、节点/关系词表及哈希、特征构建版本、嵌入模型版本、阈值、数据集哈希和指标。
- 类别顺序固定为信用、市场、操作、欺诈；不一致时加载失败。
- 固定模型包与固定图样本的兼容性测试产生可复现输出。
- 只有训练集可以建立词表和拟合特征；验证、测试数据不得参与。
- 单独的旧 checkpoint、缺失阈值或特征元数据的模型不进入推理注册表。

**依赖的前置阶段**  
阶段 2.5、阶段 6C。

## 阶段 7B：RGCN 确定性推理适配器

**目标**  
输入：阶段 6C 的已验证图谱产物和阶段 7 的可信模型包。  
输出：四类风险概率、模型版本、阈值、输入哈希和推理状态。

**需要创建或修改的文件**

- `src/financial_risk_agent/integrations/rgcn_adapter.py`
- `src/financial_risk_agent/tools/readonly/predict_risk_rgcn.py`
- `src/financial_risk_agent/tools/readonly/get_model_metadata.py`
- `tests/unit/test_risk_prediction.py`
- `tests/integration/test_rgcn_adapter.py`

**测试方式**

```bash
python -m pytest tests/unit/test_risk_prediction.py tests/integration/test_rgcn_adapter.py
```

**验收标准**

- 四类概率均处于 `[0, 1]`，并按模型包记录的类别顺序映射到固定风险字段。
- 输出包含模型版本、阈值、输入产物哈希和模型包引用。
- 模型文件缺失、哈希不匹配、特征维度不匹配或版本不兼容时安全失败。
- 推理不依赖 LLM，且不重新拟合词表、阈值或特征。

**依赖的前置阶段**  
阶段 6C、阶段 7。

## 阶段 8：只读 Safe Tool Gateway

**目标**  
输入：经过注册的只读工具名及结构化参数。  
输出：统一校验、调用、超时、错误处理和审计后的工具结果。

**需要创建或修改的文件**

- `src/financial_risk_agent/tools/registry.py`
- `src/financial_risk_agent/tools/gateway.py`
- `src/financial_risk_agent/services/policy.py`
- `tests/unit/test_tool_registry.py`
- `tests/integration/test_readonly_gateway.py`
- `tests/security/test_gateway_denials.py`

**测试方式**

```bash
python -m pytest tests/unit/test_tool_registry.py tests/integration/test_readonly_gateway.py tests/security/test_gateway_denials.py
```

**验收标准**

- 只有预注册工具可以调用。
- 参数必须通过 Pydantic 校验。
- 未知工具和修改型工具默认拒绝。
- `eval`、`exec`、Shell、SQL 和用户脚本请求永久拒绝。
- 每次调用均记录成功、失败、耗时和参数摘要。

**依赖的前置阶段**  
阶段 4、阶段 5、阶段 6C、阶段 7B。

## 阶段 9：LLM Provider 与启动自检

**目标**  
输入：OpenAI 兼容接口配置和结构化请求。  
输出：统一的 LLM 调用接口及可独立运行的能力自检；暂不创建多 Agent 流程。

**需要创建或修改的文件**

- `src/financial_risk_agent/integrations/llm_client.py`
- `src/financial_risk_agent/config/model_capabilities.py`
- `src/financial_risk_agent/services/startup_check.py`
- `tests/unit/test_llm_client.py`
- `tests/integration/test_llm_contract.py`

**测试方式**

```bash
python -m pytest tests/unit/test_llm_client.py
python -m pytest tests/integration/test_llm_contract.py
```

第二条测试仅在提供测试用 API 配置时运行；否则应自动跳过。

**验收标准**

- 默认模型配置为 `qwen3.7-max`，但允许通过环境变量覆盖。
- 支持经配置选择任意 OpenAI 兼容 Provider；不迁移或复用历史脚本中的默认密钥。
- 支持超时、有限重试和结构化输出校验。
- 密钥不会出现在异常、状态或审计日志中。
- 自检能区分连接、Tool Calling 和结构化输出故障。
- 单元测试不依赖真实外部接口。

**依赖的前置阶段**  
阶段 1、阶段 2、阶段 2.5。

## 阶段 10：单个分析 Agent 的结构化契约

**目标**  
输入：文章、图谱、RGCN 预测和已有分析结果。  
输出：六个分析 Agent 各自的结构化结果；暂不进行完整编排。

**需要创建或修改的文件**

- `src/financial_risk_agent/agents/intake.py`
- `src/financial_risk_agent/agents/graph.py`
- `src/financial_risk_agent/agents/rgcn.py`
- `src/financial_risk_agent/agents/compliance.py`
- `src/financial_risk_agent/agents/critic.py`
- `src/financial_risk_agent/agents/report.py`
- `prompts/graph/`
- `prompts/compliance/`
- `prompts/critic/`
- `prompts/report/`
- `tests/unit/agents/`

**测试方式**

```bash
python -m pytest tests/unit/agents
```

**验收标准**

- 每个 Agent 只获得其职责需要的工具。
- Agent 结果必须通过对应 Schema 校验。
- Agent 不直接访问文件、数据库或外部服务。
- Critic 能识别缺失证据和概率—解释不一致。
- 测试使用模拟 LLM，可离线运行。

**依赖的前置阶段**  
阶段 6B、阶段 6C、阶段 8、阶段 9。

## 阶段 11：AnalysisTeam 固定顺序编排

**目标**  
输入：合法的文章输入快照。  
输出：按固定顺序生成图谱、风险预测、合规意见、复核结果和报告草稿。

**需要创建或修改的文件**

- `src/financial_risk_agent/orchestration/team.py`
- `src/financial_risk_agent/orchestration/termination.py`
- `src/financial_risk_agent/orchestration/selector.py`
- `tests/workflow/test_analysis_team.py`
- `tests/workflow/test_analysis_team_limits.py`

**测试方式**

```bash
python -m pytest tests/workflow/test_analysis_team.py tests/workflow/test_analysis_team_limits.py
```

**验收标准**

- 执行顺序固定为 Intake → Graph → RGCN → Compliance → Critic → Report Draft。
- 正常流程最多六个 Agent turn。
- 工具调用次数、重试次数和超时均有限制。
- Agent 间传递结构化对象或产物引用。
- 单个 Agent 失败时流程安全停止，并保存错误记录。

**依赖的前置阶段**  
阶段 10。

## 阶段 12：确定性 Workflow Controller

**目标**  
输入：任务编号、持久化状态及 AnalysisTeam 结果。  
输出：可暂停、恢复和失败重试的确定性状态机。

**需要创建或修改的文件**

- `src/financial_risk_agent/orchestration/workflow.py`
- `src/financial_risk_agent/agents/coordinator.py`
- `src/financial_risk_agent/orchestration/state_store.py`
- `docs/state_machine.md`
- `tests/workflow/test_state_transitions.py`
- `tests/workflow/test_pause_resume.py`
- `tests/workflow/test_failure_recovery.py`

**测试方式**

```bash
python -m pytest tests/workflow
```

**验收标准**

- 只允许预定义状态转换。
- 每次转换都持久化并写入审计日志。
- 进程中断后可从上次完成阶段恢复。
- 已完成阶段不会因恢复而重复执行。
- 达到重试上限后进入 `FAILED` 或 `NEEDS_MANUAL_REVIEW`。

**依赖的前置阶段**  
阶段 3、阶段 4、阶段 11。

## 阶段 13：只读案例分析界面

**目标**  
输入：文章或已有案例编号。  
输出：可以创建任务、启动只读分析并查看结果的 Streamlit 最小界面。

**需要创建或修改的文件**

- `app/streamlit_app.py`
- `app/pages/case_analysis.py`
- `app/components/task_status.py`
- `app/components/report_view.py`
- `tests/unit/test_ui_view_models.py`
- `tests/end_to_end/test_readonly_case_analysis.py`

**测试方式**

```bash
python -m pytest tests/unit/test_ui_view_models.py
python -m pytest tests/end_to_end/test_readonly_case_analysis.py
streamlit run app/streamlit_app.py
```

**验收标准**

- 用户可选择样例案例并创建任务。
- 页面刷新不会丢失任务状态。
- 可显示图谱摘要、四类风险概率、证据和报告草稿。
- UI 不直接调用数据库或模型。
- 当前阶段不显示或执行任何修改按钮。

**依赖的前置阶段**  
阶段 12。

## 阶段 14：ActionProposal、风险策略与审批服务

**目标**  
输入：分析阶段提出的行动建议。  
输出：结构化 ActionProposal、确定性风险级别和可暂停任务的审批单。

**需要创建或修改的文件**

- `src/financial_risk_agent/services/policy.py`
- `src/financial_risk_agent/services/approval.py`
- `src/financial_risk_agent/storage/repositories/approval_repository.py`
- `src/financial_risk_agent/storage/migrations/004_approvals.sql`
- `tests/unit/test_policy.py`
- `tests/unit/test_proposal_hash.py`
- `tests/integration/test_approval_service.py`

**测试方式**

```bash
python -m pytest tests/unit/test_policy.py tests/unit/test_proposal_hash.py tests/integration/test_approval_service.py
```

**验收标准**

- 所有建议先转换为不可直接执行的 ActionProposal。
- `PROHIBITED` 操作无条件拒绝。
- `HIGH` 操作必定进入 `AWAITING_APPROVAL`。
- 审批绑定 `proposal_hash`、审批人和有效期。
- 参数、目标或预期影响变化后，旧审批自动失效。
- 拒绝后任务进入 `REJECTED`，且不执行修改。

**依赖的前置阶段**  
阶段 12。

## 阶段 15：首个受控修改工具及执行后校验

**目标**  
输入：已批准的单个中风险操作，例如 `update_case_status`。  
输出：幂等执行凭证、权威数据源重读结果和确定性校验结论。本阶段只实现一种修改操作。

**需要创建或修改的文件**

- `src/financial_risk_agent/tools/mutations/update_case_status.py`
- `src/financial_risk_agent/services/execution.py`
- `src/financial_risk_agent/services/verification.py`
- `src/financial_risk_agent/storage/repositories/case_repository.py`
- `src/financial_risk_agent/storage/repositories/execution_repository.py`
- `src/financial_risk_agent/storage/migrations/005_executions.sql`
- `tests/integration/test_update_case_status.py`
- `tests/security/test_execution_requires_approval.py`
- `tests/integration/test_mutation_verification.py`

**测试方式**

```bash
python -m pytest tests/integration/test_update_case_status.py tests/security/test_execution_requires_approval.py tests/integration/test_mutation_verification.py
```

**验收标准**

- 无有效审批时修改请求必定失败。
- 实际调用参数必须与获批提案完全一致。
- 同一幂等键重复提交不会重复修改。
- 执行后必须从 Repository 重新读取数据。
- 只有确定性校验通过后才能标记 `VERIFIED`。
- 修改前后哈希、执行凭证和校验结果全部进入审计链。

**依赖的前置阶段**  
阶段 8、阶段 14。

## 阶段 16：MutationTeam 与失败恢复

**目标**  
输入：审批通过的 ActionProposal。  
输出：Executor → Verifier → Final Report 的受控流程，以及校验失败后的补偿或人工处理状态。

**需要创建或修改的文件**

- `src/financial_risk_agent/agents/executor.py`
- `src/financial_risk_agent/agents/verifier.py`
- `src/financial_risk_agent/orchestration/team.py`
- `src/financial_risk_agent/services/verification.py`
- `src/financial_risk_agent/services/recovery.py`
- `src/financial_risk_agent/storage/migrations/006_execution_recovery.sql`
- `tests/workflow/test_mutation_team.py`
- `tests/workflow/test_verification_failure.py`
- `tests/workflow/test_compensation.py`

**测试方式**

```bash
python -m pytest tests/workflow/test_mutation_team.py tests/workflow/test_verification_failure.py tests/workflow/test_compensation.py
```

**验收标准**

- MutationTeam 只在审批有效时启动。
- Executor 只能调用提案指定的工具和参数。
- Verifier 不依赖 Executor 返回值判断成功，而是重读权威数据。
- 可回滚操作校验失败时执行补偿并再次验证。
- 不可回滚或补偿失败时进入 `NEEDS_MANUAL_REVIEW`。
- 最终报告明确记录审批、执行、校验和恢复结果。

**依赖的前置阶段**  
阶段 15。

## 阶段 17：审批、历史和审计界面

**目标**  
输入：待审批任务、历史任务及审计事件。  
输出：可批准或拒绝提案，并查看任务与审计记录的 UI。

**需要创建或修改的文件**

- `app/pages/approvals.py`
- `app/pages/task_history.py`
- `app/pages/audit_trail.py`
- `app/components/proposal_summary.py`
- `app/components/audit_timeline.py`
- `tests/unit/test_approval_view_models.py`
- `tests/end_to_end/test_approval_flow.py`

**测试方式**

```bash
python -m pytest tests/unit/test_approval_view_models.py
python -m pytest tests/end_to_end/test_approval_flow.py
```

**验收标准**

- 审批页显示目标、参数摘要、风险原因、证据、预期变化、可回滚性和有效期。
- 批准前再次验证 `proposal_hash`。
- 重复点击批准不会重复执行。
- 页面刷新或应用重启后审批状态不丢失。
- 审计页面可按任务和事件类型查询，但不能编辑记录。

**依赖的前置阶段**  
阶段 16。

## 阶段 18：扩展修改工具与全链路安全验收

**目标**  
输入：已经通过单一修改工具验证的完整架构。  
输出：逐个增加剩余修改工具，并完成安全、集成和端到端回归测试。每次只新增一种工具。

**需要创建或修改的文件**

- `src/financial_risk_agent/tools/mutations/` 下的其余白名单工具
- `docs/security.md`
- `docs/tool_catalog.md`
- `tests/security/test_prohibited_capabilities.py`
- `tests/security/test_approval_bypass.py`
- `tests/security/test_sensitive_data.py`
- `tests/end_to_end/test_readonly_flow.py`
- `tests/end_to_end/test_approved_mutation_flow.py`
- `tests/end_to_end/test_rejected_mutation_flow.py`
- `tests/end_to_end/test_manual_review_flow.py`

**测试方式**

每增加一个工具，先执行该工具的定向测试，再运行全量测试：

```bash
python -m pytest tests/unit tests/integration tests/security tests/workflow
python -m pytest tests/end_to_end
```

**验收标准**

- 每种修改工具都有独立参数模型、风险级别和验收测试。
- 高风险工具无法绕过审批。
- 系统不存在任意代码、命令、SQL 或动态工具执行入口。
- 读取、拒绝、批准、执行失败、校验失败和人工处理路径均有端到端测试。
- 全量测试通过后，才将新增工具加入正式注册表。

**依赖的前置阶段**  
阶段 17。
