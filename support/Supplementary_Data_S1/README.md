# Supplementary Data S1 Candidate level static knowledge scores

This dataset contains the archived static knowledge scores for all 3,849 candidates in the nine tasks analysed in the manuscript. Cu EC uses the final source package containing ten completed runs; the remaining tasks use their respective final sources. The export reorganizes the JSON structure without rescoring candidates, changing scores, generating rankings or invoking an LLM.

| Task | Candidates | File |
|---|---:|---|
| Cu EC | 1486 | Cu_EC_static_scores.json |
| HEA Invar TEC | 709 | HEA_Invar_TEC_static_scores.json |
| MPEA UTS | 362 | MPEA_UTS_static_scores.json |
| Mg UTS | 289 | Mg_UTS_static_scores.json |
| Ti YS | 248 | Ti_YS_static_scores.json |
| Ti UTS | 185 | Ti_UTS_static_scores.json |
| Mg YS | 298 | Mg_YS_static_scores.json |
| Al EL | 134 | Al_EL_static_scores.json |
| Al UTS | 138 | Al_UTS_static_scores.json |

Each task file records the task identifier, target column, optimization direction, composition and property units, candidate-pool size, source settings, and the repository-relative paths, byte counts and SHA-256 hashes of the archived scores, configuration and candidate pool. The original `static_scores_manifest.json` records the hashes of the exported files, the result snapshot, the export script and the numerical validation results. It is retained unchanged as a record of the original export; its source-index path identifies the original repository document. The English package manifest separately identifies this translated readme. Composition and property units are not score units: the scores are dimensionless knowledge signals.

The `records` array retains the original array order. The zero-based `candidate_index` equals the position in the source `_llm_scores.json` arrays and matches the corresponding field in the archived `candidate_pool.csv`. The `source_csv_row` field preserves the original row identifier in that candidate pool. It is distinct from `candidate_index`; the latter must not be used directly to index an original table that includes rows with missing targets.

| Field | Definition |
|---|---|
| `s_phys` | The archived static target-utility score used as a knowledge-ranking signal. It is neither a measured property nor a rank. |
| `s_target_raw` | The archived raw target-utility score. Higher scores indicate greater utility under the registered optimization direction. |
| `s_general_plausibility_raw` | The archived general-plausibility score, stored separately from the target-utility score and not used as its replacement. |
| `s_phys_raw` | A compatibility field retained in the source. Its values have been verified to match `s_target_raw` candidate by candidate in all nine tasks. |

The numerical scale is defined by the corresponding source field; the four fields are not interchangeable. Every JSON numeric token is preserved exactly, including its original decimal precision. Export applies no floating-point rounding, division by 100, renormalization or sorting. The scores are not calibrated probabilities or property predictions in physical units. Deterministic treatment of static-score ties remains defined by the source protocol; these files identify the rule but do not recompute rankings. The static-reliability analysis in Section S12 and the deterministic ranking diagnostics in main Fig. 6 address different quantities.

The export can be reproduced or checked without modifying files by running the following commands from the repository root:

```text
python paper/scripts/export_support_static_scores_20261116.py
python paper/scripts/export_support_static_scores_20261116.py --check
```

The script uses the Python standard library and archived files. Validation covers nine tasks, 3,849 candidate indices and the exact numeric text of 15,396 score values, together with the result snapshot, configurations, scoring metadata and candidate-pool identities. It does not rerun the search or extend the original LLM-evidence audit. The source-run composition and the evidence-coverage limitation for the replacement Al UTS run remain as reported in Sections S1 and S4.
