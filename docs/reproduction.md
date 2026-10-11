# Reproduction boundaries

## Verified during packaging

Python source was parsed without executing the application. Files were selected explicitly, duplicates were compared by content, and known local credentials were checked against the showcase files. Original datasets and environments were not modified.

## Prerequisites not included

| Module | Boundary |
| --- | --- |
| Term 1 research | Full news CSVs, extracted graphs, model checkpoint and a dependency lock are not included. Some scripts use fixed local filenames. |
| Agent | Its declared Python range is 3.12–3.13. `data/manifests/legacy_assets.json` was already absent in the original material, so manifest-dependent integration/security tests cannot be claimed to pass. Production host injection and historical checkpoint limitations are described in its README. |
| Term 2 | Original raw corpora, company aliases, large representations, checkpoints and generated outputs are not included. Synthetic examples only illustrate input structure. |

The current local interpreter used for syntax checks is Python 3.14, outside the Agent's declared supported range. No complete runtime or full test suite was verified.

The Term 2 pipeline's original dependency list is retained. The selected collection script uses pandas, beautifulsoup4 and tushare; figures use matplotlib. Collection requires `TUSHARE_TOKEN` and an authorized `TUSHARE_NEWS_API_URL` in environment variables. The hardcoded token and private endpoint were removed only from this showcase copy. These optional paths need a separately verified environment before running them.

Historical Agent documentation is preserved as source material. Statements about legacy manifests refer to the full research workspace, not to a bundled manifest in this showcase.
