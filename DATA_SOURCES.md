# Data sources

The eight bundled CSVs support all nine tasks in the manuscript: six tables are in `dabase/curated/`, and the HEA-Invar and MPEA tables are in `dabase/`. Cite the original datasets and associated publications when using these model-ready derivatives. Dataset licenses are separate from the code license.

| Included CSV files | Dataset and publication | Recorded source terms |
|---|---|---|
| `al_alloys.csv` | Author-owned laboratory measurements | Permission to release with the paper confirmed by the data owner |
| `Cu_alloys__EC.csv` | [Figshare collection](https://doi.org/10.6084/m9.figshare.c.6475600.v1); [publication](https://doi.org/10.1038/s41597-023-02411-9) | CC0-1.0 |
| `mg_alloy__UTS.csv`, `mg_alloy__YS.csv` | [Zenodo dataset](https://doi.org/10.5281/zenodo.17672235); [publication](https://doi.org/10.1016/j.matdes.2026.116060) | CC-BY-4.0 |
| `Ti_alloy__UTS.csv`, `Ti_alloy__YS.csv` | [Zenodo dataset](https://doi.org/10.5281/zenodo.6089749); [publication](https://doi.org/10.1038/s41597-022-01283-9) | CC-BY-4.0 |
| `HEA_Invar__TEC.csv` | Ziyuan Rao et al., *Machine learning-enabled high-entropy alloy discovery*, [Zenodo v1](https://doi.org/10.5281/zenodo.7019194); [publication](https://doi.org/10.1126/science.abo4940) | CC-BY-4.0, verified from the Zenodo deposit |
| `MPEA__Mechanical.csv` | Christopher Borg et al., *Expanded dataset of mechanical properties and observed phases of multi-principal element alloys*, [Figshare v9](https://doi.org/10.6084/m9.figshare.12642953.v9); [publication](https://doi.org/10.1038/s41597-020-00768-9) | CC-BY-4.0, verified from the Figshare deposit |

Source license texts: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code's [MIT License](LICENSE) does not replace these dataset terms.

The two added datasets were checked against their primary deposits on November 9, 2026. The deposited source-file checksums match the originals used to prepare these tables. Full creator attribution, source versions, checksums and publication-only metadata changes are recorded in [`dabase/source_licenses.json`](dabase/source_licenses.json). For CC BY data, retain attribution and the license link, identify the preparation changes below, and indicate any further modifications.

## Data preparation

| Material | Composition unit | Included targets | Main transformations |
|---|---|---|---|
| Al | wt.% | UTS: 138; EL: 134 candidates | Group identical visible-input candidates and take the median of reported responses. |
| Cu | wt.% | EC: 1,486 candidates | Retain 27 element columns and reported process missingness; exclude 12 rows outside the 95–105 wt.% composition-sum range; deduplicate by composition/process and median-aggregate responses. |
| Mg | wt.% | UTS: 289; YS: 298 candidates | Use measured raw records; exclude five unspecified `RE` records and two records without structured composition; derive the Mg balance; deduplicate before target filtering. |
| Ti | at.% | UTS: 185; YS: 248 candidates | Parse atomic compositions, scale atomic fractions to at.%, and deduplicate by composition and condition. |
| HEA-Invar | at.% | TEC: 709 candidates | Select the six reported elements and thermal expansion target; scale atomic fractions to at.% without renormalization; merge identical compositions using median responses (717 source rows to 709 candidates). |
| MPEA | at.% | UTS: 362 target-valid candidates in a 1,154-row table | Parse formula coefficients into at.%; retain processing method, test type and temperature; median-aggregate repeated visible inputs (1,545 source rows to 1,154 candidates). Keep auxiliary properties excluded from model and LLM inputs. |

The tables are already deduplicated. Runtime loading must preserve their composition units and stored target values. Unreported composition or process fields remain unknown rather than zero. Al UTS and EL share one table; two response targets do not constitute independent datasets.

The bundled JSON records retain the preparation hashes and source paths. Some paths refer to original audit materials that are not included in this release; they document provenance and are not required runtime inputs.

The HEA-Invar deduplication record preserves the historical observation that the source GitHub repository did not declare a license. The author's Zenodo deposit supplies CC BY 4.0 for the byte-identical source workbook, as documented in `source_licenses.json`. The MPEA column record adds an audited-registry marker for the public loader; its column roles, source fingerprints and CSV values are unchanged.

The MPEA UTS task includes both tensile and compression records, distinguished by the stored test-type input. Target-valid counts are task-specific: Al UTS and EL share one table, and only 362 of the 1,154 MPEA rows have the registered UTS target. Original workbooks, downloaded source files, caches and temporary preparation files are not bundled.
