# News events, risk scoring and community evolution

The main chain is news collection and sampling → structured events → company-month risk representation and scoring → monthly risk communities and their evolution.

The `方案一` folder contains the pipeline, tests, schema and prompt templates. `数据抓取/长文抓取.py` contains the long-form news collector. Old collection alternatives and full news corpora are excluded. Recorded aggregate figures are in `../../results/term2/`.

## Preparation

From `方案一`, install the existing pipeline dependency list:

```bash
python -m pip install -r requirements.txt
```

This original list is not a complete environment lock. Collection, plotting and optional fine-tuning modules have additional dependencies; consult [reproduction notes](../../docs/reproduction.md).

Copy `.env.example` in the Term 2 root to `.env` and fill only the providers you intend to use. LLM calls use external services.

## Synthetic input

`方案一/data/news_input.synthetic.jsonl` and `方案一/data/few_shot_examples.synthetic.json` use fictional companies and text created for this showcase. They illustrate the input contract; they are not a training dataset or validation evidence.

Example extraction command, from `方案一`:

```bash
python pipeline/run_docfee_news.py --input data/news_input.synthetic.jsonl --schema config/schema_news_v0.yaml --output output/demo/news_events.jsonl --few-shot-path data/few_shot_examples.synthetic.json
```

A configured LLM provider is required. This command was not executed during packaging. Use each stage's `--help` to inspect its required inputs. Existing unit tests construct their own small inputs; no full-suite pass is claimed.
