# Data sources

The seven CSVs in `dabase/curated/` are model-ready derivatives of the sources below. The source terms listed here follow the project's existing dataset records. Cite the original datasets and associated publications when using these data.

| Included CSV files | Dataset and publication | Recorded source terms |
|---|---|---|
| `al_alloys.csv` | Author-owned laboratory measurements | Permission to release with the paper confirmed by the data owner |
| `Cu_alloys__EC.csv` | [Figshare collection](https://doi.org/10.6084/m9.figshare.c.6475600.v1); [publication](https://doi.org/10.1038/s41597-023-02411-9) | CC0-1.0 |
| `mg_alloy__UTS.csv`, `mg_alloy__YS.csv` | [Zenodo dataset](https://doi.org/10.5281/zenodo.17672235); [publication](https://doi.org/10.1016/j.matdes.2026.116060) | CC-BY-4.0 |
| `steel__UTS.csv` | [Materials Cloud dataset](https://doi.org/10.24435/materialscloud:ws-8j); [publication](https://doi.org/10.1038/s41597-024-04329-2) | CC-BY-4.0 |
| `Ti_alloy__UTS.csv`, `Ti_alloy__YS.csv` | [Zenodo dataset](https://doi.org/10.5281/zenodo.6089749); [publication](https://doi.org/10.1038/s41597-022-01283-9) | CC-BY-4.0 |

Source license texts: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code's [MIT License](LICENSE) does not replace these dataset terms.

## Data preparation

| Material | Composition unit | Included targets | Main transformations |
|---|---|---|---|
| Al | wt.% | UTS: 138; EL: 134 candidates | Group identical visible-input candidates and take the median of reported responses. |
| Cu | wt.% | EC: 1,486 candidates | Retain 27 element columns and reported process missingness; exclude 12 rows outside the 95–105 wt.% composition-sum range; deduplicate by composition/process and median-aggregate responses. |
| Mg | wt.% | UTS: 289; YS: 298 candidates | Use measured raw records; exclude five unspecified `RE` records and two records without structured composition; derive the Mg balance; deduplicate before target filtering. |
| Steel | wt.% | UTS: 272 candidates | Select reported UTS with Fe ≥50 wt.% plus the recorded SS316L/UNS S31035 exceptions; deduplicate using composition, processing and test conditions. |
| Ti | at.% | UTS: 185; YS: 248 candidates | Parse atomic compositions, scale atomic fractions to at.%, and deduplicate by composition and condition. |

The tables are already deduplicated. Runtime loading must preserve their composition units and stored target values. Unreported composition or process fields remain unknown rather than zero. Al UTS and EL share one table; two response targets do not constitute independent datasets.

HEA-Invar, MPEA, FeCoNi, metallic-glass and steel-fatigue inputs are not included and must be obtained separately with the appropriate source terms. Original workbooks, downloaded source files and full experiment-result archives are also not bundled.
