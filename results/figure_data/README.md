# Frozen figure inputs

These compact extracts supplement the complete paired benchmark and ablation results in the parent directory. They reproduce the figures and tables in the final manuscript and its English supplementary Sections S15-S17 without API requests, model training or research-archive files.

From the repository root, install the lightweight plotting dependencies and run:

```bash
python -m pip install -r requirements-figures.txt
python reproduce/figures.py --output-dir outputs/paper_figures
```

The command writes main Figures 1-9, supplementary Figures S18-S22, Tables 1-2, Tables S15.1-S15.3 and Table S16.1. Quantitative figures are saved as PNG, PDF and SVG; tables and selected plotted values are CSV. Use `--figures 6 7 S22 --skip-tables` for a subset, or `--tables-only` for tables.

| Output | Published input |
| --- | --- |
| Figures 1-2 | Authored diagrams in `assets/calm-overview.png` and `assets/calm-framework.png`, copied unchanged |
| Figures 3-4; S19-S20; Tables 2 and S16.1 | `results/benchmark/` paired convergence records and task summaries |
| Figure 5; Table 1 | `results/ablations.json`, all 450 runs for the five module configurations |
| Figure 6 | `knowledge_ranks.csv`, all 90 final pre-query shortlist comparisons |
| Figure 7 | `mpea_case.json`, measured compositions, recorded model rationales and same-state ranks |
| Figure 8; S21 | `mpea_search.json`, fixed PCA coordinates and complete paired MPEA run-7 searches |
| Figure 9 | `ewrc_queries.csv`, all 218 eligible states, and `ewrc_selection.json`, three complete weight/search trajectories |
| Figure S18; Tables S15.1-S15.3 | `first_batch.json`, all 90 runs and the 83-run sensitivity analysis excluding initial optima |
| Figure S22 | `knowledge_reliability.json`, per-state, per-run and per-task reliability results |

Main Figures 3-4 compare nine strategies; supplementary Figures S19-S20 also include CALM-static. The benchmark and ablation files are the numerical source of their panels and tables, rather than another copy of precomputed plot arrays. Figure 9 recomputes selection changes from the actual and preceding-weight selected candidate sets.

The quantitative figures are portable Matplotlib redraws of the final manuscript revisions. They preserve the selected data, panels, comparisons and scientific content; the bundled DejaVu font, annotation placement and simplified Figure 7 cards can differ from the publication artwork. The PCA coordinates in Figures 8 and S21 are frozen from the original analysis, not refitted. The model rationale excerpts in Figure 7 are historical judgments; they are not measured outcomes or independently verified physical explanations.

`sources.json` records the original source and published extract SHA-256 hashes. Its `research/` and `manuscript_workspace/` paths identify the author archive and are not runtime dependencies. The extracts omit caches, complete prompt/evidence logs and local report paths. Dataset rights and attribution remain as described in [`DATA_SOURCES.md`](../../DATA_SOURCES.md); reproducing a numerical figure does not change the terms of its source data.
