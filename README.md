# CALM – Calibrated Adaptive Learning Method

Code for **CALM A Self Adaptive Knowledge Data Framework for Universal Metallic Materials Exploration**.

CALM combines a calibrated ensemble of GPR, random forest, bootstrap LightGBM and MLP models with LLM-derived materials knowledge. Feedback-guided shortlist reranking (FSR) updates candidate rankings using measured anchors and persistent local references. Extreme-weighted rank correlation (EWRC) adjusts how strongly the knowledge ranking influences experimental selection:

$$
a(x)=(1-\lambda)\operatorname{rankpct}(UCB(x))+
\lambda\operatorname{rankpct}(s_K(x)),\qquad
\lambda=\operatorname{clip}\left(0.6\max(\rho_{\mathrm{EW}},0),0,0.6\right).
$$

Here, the knowledge rank includes FSR updates, and $UCB=\mu_{\mathrm{ens}}+1.96\sigma_{\mathrm{ens}}$. Knowledge scores and observed targets are compared in the registered optimization direction. Scoring uses no retrieval-augmented generation (RAG).

## Installation

The reference environment uses **Python 3.9.21** and **PyTorch 2.7.0 with CUDA 12.8** on Linux or Windows. In an activated Python environment, install PyTorch first, then the remaining pinned dependencies:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The PyTorch command follows the [official version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v270).

## Run an experiment

Copy [`common/llm/.env.example`](common/llm/.env.example) to `common/llm/.env` and set:

```dotenv
CALM_API_BASE_URL=https://api.example.com/v1
CALM_LLM_MODEL=deepseek-v4-flash-ascend
CALM_API_KEY=
```

Use an API key with access to the selected model and an endpoint supporting OpenAI-compatible chat completions with JSON-object responses. The values above are the reference service and model; change both for another provider. Shell environment variables take precedence over `.env`. For multiple keys, use distinct keys numbered consecutively from `CALM_API_KEY_1` through `CALM_API_KEY_4`, and select one with `--api-key-slot`. Credentials stay local.

Start with one paired run across all ten strategies. This still performs remote LLM scoring and may incur API charges:

```bash
python run.py --dataset 2xxx_T6__UTS --n-runs 1
```

Run the default experiment or view available options:

```bash
python run.py --dataset 2xxx_T6__UTS
python run.py --help
```

Defaults are **10 paired runs**, **10 strategies**, an initial sample of **10%** of the candidate pool, batches of **2%**, and a total measurement budget of **40%**, including initialization. The final batch is truncated to the remaining budget. Each strategy updates from its own measured samples while sharing the paired initialization and query schedule.

The strategy identifiers are `Random`, `GPR-UCB`, `RFR-UCB`, `GBDT-UCB`, `NN-UCB`, `Adaptive-UCB`, `LLM-rank`, `LLM-dynamic`, `Hybrid-static` and `Hybrid`. `Hybrid` denotes CALM; `Hybrid-static` disables FSR while retaining adaptive knowledge weighting. The main search metric is normalized Regret AUC, where lower values are better.

Default results are saved in `outputs/calm_method_v1/<dataset>/`; parameter overrides such as `--n-runs 1` use `outputs/calm_method_v1/_custom/<hash>/<dataset>/`. Resume requires matching code, data and configuration.

| Output | Contents |
|---|---|
| `convergence.json` | Per-run acquisition and best-so-far trajectories |
| `stats.json` | Summary metrics, including `norm_regret_auc_mean` and `optimum_hit_rate_by_budget` |
| `convergence_plot.png` | Convergence curves |

## Project structure

```text
CALM/
├── run.py              # Experiment entry point
├── calm/               # Acquisition loop, strategies and metrics
├── common/             # Shared models, features and LLM scoring
├── dabase/             # Curated model-ready inputs
├── requirements.txt    # Dependency versions
└── DATA_SOURCES.md     # Dataset attribution and recorded terms
```

## Data

Seven curated CSVs are included for Al UTS/EL, Cu EC, Mg UTS/YS, Steel UTS and Ti UTS/YS. Other registered tasks require separately supplied inputs, including HEA-Invar and MPEA. The registered HEA-Invar objective minimizes thermal expansion. This repository does not include all manuscript datasets or result archives.

Composition units and missing values follow each dataset's column definitions; unknown values must not be interpreted as zero. Dataset sources, licenses and preprocessing are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

To add a dataset, deduplicate by candidate inputs, then register a `DatasetConfig` in [`calm/runner.py`](calm/runner.py) with its CSV path, target column, composition unit, optimization direction, expected valid row count and other measured-property columns. These other properties must be excluded from model and LLM inputs. For unreported composition values, set `composition_missing_policy=REPORTED_COMPOSITION_POLICY`.

Supply matching `.columns.json` and `.dedup.json` files with column roles, hashes and row counts recalculated for the new CSV; see the [Al CSV](dabase/curated/al_alloys.csv), [column definitions](dabase/curated/al_alloys.columns.json) and [deduplication record](dabase/curated/al_alloys.dedup.json). Run the registered task using `--dataset <TASK_ID>`.

## Citation

Please cite **CALM A Self Adaptive Knowledge Data Framework for Universal Metallic Materials Exploration** and the [CALM repository](https://github.com/Tyler-gt/CALM).

## License

The code is released under the [MIT License](LICENSE). Datasets retain the terms listed in [DATA_SOURCES.md](DATA_SOURCES.md).
