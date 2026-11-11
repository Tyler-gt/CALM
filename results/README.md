# Paper results

These files contain the numerical results used by the current manuscript. They support plotting and statistical reproduction without model training or API requests.

- `benchmark/summary.json` provides the nine tasks, actual seeds, run identifiers, metrics and ranks used in the manuscript.
- Each `benchmark/<dataset>/convergence.json` contains ten paired runs for all ten strategies, including per-round acquisitions and numerical diagnostics. `initial_sets.json` records the paired starting candidates.
- `ablations.json` contains 450 run summaries and their 7,150 per-round curve records for five module configurations, together with the reported task statistics and comparisons. It records the reuse of main full/static runs and the offline control with both FSR and EWRC disabled.
- `manifest.json` records file hashes and manuscript identity.

The stored strategy names `Hybrid` and `Hybrid-static` mean CALM and CALM-static. Main-paper comparisons use CALM and eight baselines; CALM-static remains available for supplementary analyses.

Candidate indices are zero-based positions after filtering the registered CSV to rows with a numeric target value, preserving CSV order. The data file hash in each task's summary binds that interpretation.

Plotting reads only files included in this repository. Caches, complete LLM evidence, model binaries, checkpoints and temporary files are not distributed here.
