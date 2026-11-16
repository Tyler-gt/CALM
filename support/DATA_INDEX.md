# Data and reproducibility index

The links below resolve within the CALM repository. Downloading the support ZIP alone supplies the documents and Supplementary Data S1; repository result links additionally require the full checkout or the [online repository](https://github.com/Tyler-gt/CALM).

## Supplementary Data S1

[Description](Supplementary_Data_S1/README.md), [source manifest](Supplementary_Data_S1/static_scores_manifest.json) and [ZIP](Supplementary_Data_S1.zip). The nine JSON datasets retain all 3,849 candidate records, original score precision, candidate identifiers and source hashes. Research paths embedded in the original manifests identify source records and are not current repository locations.

## Results available in this repository

| Analysis | Repository entry |
|---|---|
| Nine-task benchmark summary, actual seeds and run identities | [summary](../results/benchmark/summary.json) |
| All ten strategies, paired initial samples and complete query trajectories | [task archives](../results/benchmark/) containing each task's initial_sets.json and convergence.json |
| Five-configuration module ablation, 450 trajectories and associated statistics | [ablation results](../results/ablations.json) |
| Current 90-run shortlist comparisons in main Fig. 6 | [knowledge rank metrics](../results/figure_data/knowledge_ranks.csv) |
| MPEA candidate 131, recorded rationale excerpts and same-state ranks | [case extract](../results/figure_data/mpea_case.json) |
| MPEA run 7 spatial coordinates and complete paired trajectories | [search case](../results/figure_data/mpea_search.json) |
| S15 and Fig. S18: first-batch, cumulative five-batch and 83-run sensitivity statistics | [acquisition fractions](../results/figure_data/first_batch.json) |
| S17 and Fig. S22: per-state, per-run and per-task knowledge reliability | [reliability results](../results/figure_data/knowledge_reliability.json) |
| EWRC same-state selection changes across 218 eligible queries | [query comparisons](../results/figure_data/ewrc_queries.csv) |
| Complete weight and search trajectories for the displayed EWRC cases | [selection cases](../results/figure_data/ewrc_selection.json) |
| Figure-input hashes and result/manuscript identities | [figure sources](../results/figure_data/sources.json); [result manifest](../results/manifest.json) |
| Registered candidate inputs and attribution | [data sources](../DATA_SOURCES.md); [model-ready tables](../dabase/) |
| Offline reproduction of the bundled figures and tables | [figure guide](../results/figure_data/README.md); [plotting script](../reproduce/figures.py) |

The benchmark contains 900 strategy trajectories. Main comparison figures display nine strategies; the complete supplementary comparison also includes CALM-static. The separate five-configuration ablation contains 450 trajectories, including the reused Full CALM and FSR-disabled runs. Stored names Hybrid and Hybrid-static denote CALM and CALM-static.

## Research archives not included in this repository

The supplementary document also discusses analyses retained in the original research archive. This upload does not add the full archive: S5 model-level initial-CV and stability records; S10 full-remaining-pool, recall and score-tie sensitivity exports and historical diagnostics; S2 original exact-test and Holm summary exports; complete LLM prompt/evidence logs and recovery records; detailed Mg/Ti case audits and Mg YS conditional reconstruction; or the S12 static-reliability diagnostic export and its research scripts.

The benchmark trajectories and newly supplied static scores are available above, but they are not presented as substitutes for these omitted diagnostic exports or complete LLM evidence. In particular, knowledge_ranks.csv covers the current shortlist metrics, and mpea_case.json is a compact case extract. The previously recorded limitation on complete dynamic-evidence coverage for the replacement Al UTS run remains as stated in Section S4.
