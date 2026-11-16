# CALM – Calibrated Adaptive Learning Method

Code for **CALM A Self Adaptive Knowledge Data Framework for Universal Metallic Materials Exploration**.

![CALM framework: data and knowledge fusion, EWRC adaptive weighting, and FSR feedback updating](assets/calm-framework.png)

*Figure 2 from the paper: the CALM exploration loop, EWRC-based adaptive weighting, and feedback-guided shortlist reranking (FSR).*

CALM combines a calibrated GPR/RFR/bootstrap-LightGBM/MLP ensemble with LLM-derived materials knowledge. Feedback-guided shortlist reranking (FSR) updates knowledge ranks using measured anchors and persistent candidate references. Extreme-weighted rank correlation (EWRC) adapts the knowledge weight:

$$
a(x)=(1-\lambda)\operatorname{rankpct}(UCB(x))+
\lambda\widetilde{R}^{K}(x),\qquad
\lambda=\operatorname{clip}\left(0.6\max(\rho_{\mathrm{EW}},0),0,0.6\right).
$$

Here, $\widetilde{R}^{K}$ is the updated knowledge percentile rank. FSR reorders shortlisted candidates within their original static rank positions; candidates outside the shortlist retain their ranks.

Inputs are raw compositions, available processing/testing conditions, categorical encodings and missingness indicators. X-only screening removes unusable or constant columns while protecting varying raw inputs from correlation-based removal. The shared MLP has a 64→32→16→1 GELU architecture with MC dropout. LLM scoring uses no retrieval augmentation.

## Installation

Use a **Python 3.9** environment; the reference environment used **3.9.21**. Download or clone this repository and run the commands below from its root directory. To reproduce only the bundled figures and tables, use the [lightweight installation below](#reproduce-the-paper-figures).

For new experiments, install **PyTorch 2.7.0** for your operating system and hardware using the [official instructions](https://pytorch.org/get-started/previous-versions/#v270). For example, a CPU-only installation on Linux or Windows:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

CUDA is optional. The reference environment used the CUDA 12.8 build; choose a compatible build for your own hardware. The model uses CUDA when available and otherwise runs on the CPU.

<details>
<summary>Windows Conda troubleshooting: duplicate OpenMP runtimes</summary>

If a Windows Conda environment reports duplicate OpenMP runtimes (`OMP: Error #15`), set [MKL's sequential threading layer](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2026-0/dynamic-select-the-interface-and-threading-layer.html) before starting Python:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
```

This setting applies to the current shell and runs MKL operations in single-threaded mode.

</details>

## Supplementary materials

The revised English [Supplementary Information PDF](support/CALM_Supplementary_Information_English.pdf), [editable Word version](support/CALM_Supplementary_Information_English.docx), separate S14 Methods and Supplementary Data S1 are available in [support/](support/). [Download the complete package](support/CALM_English_Support_Package.zip) or use the [data index](support/DATA_INDEX.md) to locate the accompanying repository results.

## Reproduce the paper figures

Prepared inputs for all nine tasks and the frozen results needed for the manuscript figures and tables are included. This workflow needs only NumPy, pandas and Matplotlib, without PyTorch, a GPU or API access:

```bash
python -m pip install -r requirements-figures.txt
python reproduce/figures.py --output-dir outputs/paper_figures
```

The command generates main Figures 1–9, Tables 1–2, supplementary Figures S18–S22 and Tables S15.1–S15.3 and S16.1 from the bundled paper results. See [`results/README.md`](results/README.md) for the result files and [`results/figure_data/README.md`](results/figure_data/README.md) for the figure inputs. Figures 1 and 2 are authored schematics copied from the manuscript assets; quantitative plots use a portable font and the manuscript's numerical values and panel definitions.

## API configuration

The paper used **DeepSeek V4 Flash**, recorded under the deployment identifier `deepseek-v4-flash-ascend`, at temperature 0 with retrieval disabled.

For new experiments, copy [`common/llm/.env.example`](common/llm/.env.example) to `common/llm/.env` and configure your own provider account. Example using the [official DeepSeek API](https://api-docs.deepseek.com/):

```dotenv
CALM_API_BASE_URL=https://api.deepseek.com
CALM_LLM_MODEL=deepseek-flash
CALM_API_KEY=
```

Set `CALM_LLM_MODEL` to a model ID available from your provider. The official `deepseek-flash` alias follows the provider's current model version and differs from the paper's recorded deployment. Other providers must support OpenAI-compatible chat completions with JSON-object responses. Shell variables override `.env`; configuration uses the `CALM_*` variables. Reproducing the bundled paper figures requires no API access.

## Run the benchmark

For a selected task, run one or all ten paired repetitions using the recorded paper seeds:

```bash
python run.py --dataset 2xxx_T6__UTS --n-runs 1
python run.py --dataset 2xxx_T6__UTS
```

New runs use remote LLM scoring and may incur API charges. The default main comparison includes **CALM and eight baselines**: Random, GPR-UCB, RFR-UCB, GBDT-UCB, NN-UCB, Adaptive-UCB, LLM-rank and LLM-dynamic. To include CALM-static, as in the supplementary comparison:

```bash
python run.py --dataset 2xxx_T6__UTS --experiment supplementary
```

Initialization uses 10% of the candidate pool, batches use 2%, and the total measurement budget is 40%, including initialization. Counts follow the paper's rounding rules and the final batch is truncated. Strategies share initial sets, inputs and measurement schedules, then learn from their own observations.

The runner directly uses the actual per-run seed lists in [`calm/benchmark.py`](calm/benchmark.py), in the order shown below. `--n-runs` uses the first requested entries; `--seed <value>` generates a new paired experiment. Result files are described in [`results/README.md`](results/README.md). Fresh API responses and the revised implementation can produce different trajectories; use the bundled results to reproduce the manuscript figures.

<details>
<summary>Actual seeds used in the paper runs</summary>

| Task | Seeds, in run order |
|---|---|
| `Cu_alloys__EC` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `HEA_Invar__TEC` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `MPEA__UTS` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 669991378 |
| `mg_alloy__UTS` | 1782176804, 1028655772, 465679224, 1114009941, 678236246, 1400879312, 1283215481, 1709079418, 1324328782, 78442560 |
| `Ti_alloy__YS` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `Ti_alloy__UTS` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `mg_alloy__YS` | 1608637542, 1273642419, 669991378, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `2xxx_T6__EL` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 670094950, 1914837113 |
| `2xxx_T6__UTS` | 1608637542, 1273642419, 1935803228, 787846414, 996406378, 1201263687, 423734972, 415968276, 669991378, 1914837113 |

</details>

## Module ablations

Use `--experiment` to run a single CALM configuration from the five-way paper ablation:

| Value | FSR | Knowledge weighting |
|---|---|---|
| `full` | Enabled | EWRC, γ=2 |
| `no-fsr` | Disabled | EWRC, γ=2 |
| `no-ewrc` | Enabled | Fixed λ=0.3 |
| `no-fsr-no-ewrc` | Disabled | Fixed λ=0.3 |
| `spearman` | Enabled | Ordinary Spearman, γ=0 |

```bash
python run.py --dataset 2xxx_T6__UTS --experiment no-ewrc
python run.py --help
```

Ablations retain the same ensemble, knowledge branch and paired initialization, and use their own measured feedback.

## Data

The eight bundled CSVs support all nine tasks in the manuscript.

| Task ID | Property | Direction | Candidates |
|---|---|---|---:|
| `Cu_alloys__EC` | Electrical conductivity | Maximize | 1486 |
| `HEA_Invar__TEC` | Thermal expansion coefficient | Minimize | 709 |
| `MPEA__UTS` | Ultimate tensile strength | Maximize | 362 |
| `mg_alloy__UTS` | Ultimate tensile strength | Maximize | 289 |
| `Ti_alloy__YS` | Yield strength | Maximize | 248 |
| `Ti_alloy__UTS` | Ultimate tensile strength | Maximize | 185 |
| `mg_alloy__YS` | Yield strength | Maximize | 298 |
| `2xxx_T6__EL` | Elongation | Maximize | 134 |
| `2xxx_T6__UTS` | Ultimate tensile strength | Maximize | 138 |

See [DATA_SOURCES.md](DATA_SOURCES.md) for sources, composition units and preparation. Missing values remain explicit in source tables and LLM inputs; numerical surrogate inputs use median imputation and missingness indicators. All target/auxiliary-property columns are excluded from model and LLM inputs. Al UTS and EL share one table.

For another dataset, register a `DatasetConfig` in [`calm/runner.py`](calm/runner.py) with `data_path` pointing to a prepared CSV, and use an explicit `--seed` for new runs. Supply a deduplicated CSV in its documented composition unit plus both `.columns.json` and `.dedup.json` records with matching hashes and row counts; see the [Al CSV](dabase/curated/al_alloys.csv), [column definitions](dabase/curated/al_alloys.columns.json) and [deduplication record](dabase/curated/al_alloys.dedup.json). Set `expected_target_valid_rows`, the target, direction, composition unit and auxiliary-property exclusions explicitly; use `REPORTED_COMPOSITION_POLICY` when composition fields are unreported.

Both data records are required and validated before API calls. Direct users of `LLMReviewer` must also supply `element_cols` and `process_cols` explicitly; `process_cols=[]` is valid.

## Outputs and code

Default results go to `outputs/calm_paper_v1/<dataset>/`; other experiments and overrides use `outputs/calm_paper_v1/_custom/<hash>/<dataset>/`. Static knowledge scores are shared across comparisons under `outputs/calm_paper_v1/_static/`. Resume requires matching code, environment, data, configuration and the same static scoring generation.

Repeating a completed run preserves its results. Existing outputs from older code are also preserved and cannot be resumed with a different implementation. For an independent rerun with a fresh static-scoring cache, choose a new root:

```bash
python run.py --dataset 2xxx_T6__UTS --output-root outputs/rerun_01
```

| File | Contents |
|---|---|
| `convergence.json` | Per-run selections and best-so-far trajectories |
| `stats.json` | nAUC, Hit@budget, first-hit summaries and paired nAUC differences |
| `convergence_plot.png` | Mean trajectories with sample-standard-deviation bands |

`run.py` is the entry point; `calm/` contains the acquisition loop, strategies and metrics; `common/` contains raw-feature filtering, the MLP and LLM scoring. The main search metric is normalized regret AUC (lower is better). Summary comparisons report paired mean differences and sample standard deviations.

## Citation and license

Please cite **CALM A Self Adaptive Knowledge Data Framework for Universal Metallic Materials Exploration** and the [CALM repository](https://github.com/Tyler-gt/CALM).

Code is released under the [MIT License](LICENSE). Datasets retain the terms in [DATA_SOURCES.md](DATA_SOURCES.md).
