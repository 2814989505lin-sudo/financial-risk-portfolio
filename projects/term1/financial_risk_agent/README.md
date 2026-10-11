# Financial Risk Agent

面向新闻与案例的金融风险分析应用。系统将逐步提供可审计的图谱抽取、RGCN 风险预测、合规解释、人工审批和受控执行能力。

## 当前状态

当前已完成可审计的核心模型、历史资产适配、确定性 RGCN 测试运行时、只读工具网关、LLM 契约、固定分析流程与最小只读案例分析界面。生产案例分析服务需由部署宿主注入；当前历史生产 checkpoint 仍不会注册为推理模型。

## 本地开发

使用 Python 3.12 或 3.13 创建虚拟环境后，安装项目及开发依赖：

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

复制 `.env.example` 为 `.env`。应用在当前工作目录读取该文件；系统环境变量优先于 `.env` 中的同名配置。仅在需要调用 LLM 时填写密钥，且不要提交 `.env`、`.env.*` 或密钥文件。

验证安装与默认配置：

```bash
python -c "from financial_risk_agent.config import Settings; print(Settings.from_environment().to_safe_dict())"
```

启动 Streamlit 界面：

```bash
streamlit run app/streamlit_app.py
```

界面不会直接访问数据库或模型。部署宿主必须向 Streamlit session 注入已配置的 `case_analysis_service`；未注入时界面仅显示服务未配置提示。

运行基础测试：

```bash
python -m pytest tests/unit/test_settings.py
```

运行格式与静态检查：

```bash
python -m ruff format --check .
python -m ruff check .
```
