# 历史资产导入清单说明

本清单只描述可由新应用通过受控适配器读取的资产。历史脚本、Notebook、Shell 编排、Neo4j 导入器和本地虚拟环境均不属于运行时模块。

| 资产类别 | 准入状态 | 用途 | 约束 |
|---|---|---|---|
| Guardian 训练、验证、测试 CSV | `read_only_candidate` | 新闻输入与回归测试样本 | 使用内容哈希校验；保留来源和列映射。 |
| 文档级图 JSON | `read_only_candidate` | RGCN 输入和图谱转换测试 | 仅作为旧格式输入；转换后使用新 GraphArtifact。 |
| `nodes / edges / docs` 图谱 JSON | `read_only_candidate` | 图谱读取与证据迁移测试 | 无原文证据的共现边必须降级。 |
| 既有 RGCN checkpoint | `research_only_pending_model_bundle` | 模型兼容性研究 | 不可直接提供推理，需先完成模型包和兼容性测试。 |
| 采集、LLM、spaCy、Neo4j、GAT、Shell 和 Notebook 脚本 | `excluded_runtime` | 研究参考 | 不可被新应用导入、执行或注册为工具。 |

机器可读的资产路径与内容哈希以 `data/manifests/legacy_assets.json` 为准。清单中不保存凭据、原始新闻正文或模型内部权重。
