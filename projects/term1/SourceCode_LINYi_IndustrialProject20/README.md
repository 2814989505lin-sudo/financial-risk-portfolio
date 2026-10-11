# News graph classification with RGCN

Research pipeline for collecting news, labeling financial risks with an LLM, extracting document-level graphs, training an RGCN classifier and comparing graph/MLP depths.

| Stage | Files |
| --- | --- |
| Collection and labeling | `0.news_collecting.py`, `1.labeling.py`, `llm_config.py` |
| Dataset split | `2.splitting.py` |
| Graph extraction | `3.llm_train.py`, `3.llm_val.py`, `3.llm_test.py` |
| Classification | `RGCN_4.py` |
| Ablation and plotting | `4_ablation_RGCN_MLP_depth.py`, `ablation_draw.py` |

`ablation_rgcn_depth_mlp_fast.csv` and its PNG contain recorded validation results. They are not evidence of performance on an independent production dataset.

The scripts expect local CSV and graph JSON files. Those full datasets and the checkpoint are excluded. Collection uses `GUARDIAN_API_KEY`; LLM settings are read from environment variables by `llm_config.py`. Inspect that file for the supported names. Dependencies used by this research code include requests, python-dateutil, pandas, scikit-learn, PyTorch, torch-geometric, transformers, langchain-openai, langchain-core, pydantic and matplotlib. No locked dependency file was present, and end-to-end execution is not verified in this package.
