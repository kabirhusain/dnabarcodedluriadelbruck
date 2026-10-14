<p align="center"><img src="20261014_barfly_logo.png" alt="BarFly" width="400"></p>

# BarFly — analysis code

Code and processed data for *A DNA-barcoded Luria-Delbrück assay resolves mechanisms of
adaptation* (Husain, Flagg, Pincus & Murugan).

BarFly replaces the physically separate replicate cultures of a Luria-Delbrück fluctuation
assay with DNA-barcoded lineages grown in one flask, and colony counting with deep sequencing.
That gives >10⁵ statistical replicates in a single experiment, enough to resolve the shape of
the Luria-Delbrück distribution — and so to tell apart mechanisms of adaptation that summary
statistics cannot.

## Quick start

```bash
conda env create -f environment.yml && conda activate barfly
```

Run first:

```bash
python figures/20260907_expt3_fits.py --workers 18    # Expt 3 fit pipeline, ~8 min on 20 cores
```

Run order:

| order | notebook | |
|---|---|---|
| 1 | `figures/20260907_figS04_expt1_fit.ipynb` | Expt 1 fit, ~4 min — writes `cache/expt1_fit.npz` |
| 2 | `figures/20260907_fig02_expt1.ipynb` | needs 1 |
| 3 | `figures/20260907_fig05_dmus.ipynb` | needs 1 |
| 4 | `figures/20260907_figS01_expt1_qc.ipynb` | independent |
| 5 | `figures/20260907_figS05_expt2_fit.ipynb` | Expt 2 fit, ~6 min |
| 6 | `figures/20260907_fig03_expt2_collapse.ipynb` | independent |
| 7 | `figures/20260907_figS02_expt2_qc.ipynb` | independent |
| 8 | `figures/20260907_fig04_expt3_uv.ipynb` | needs the Expt 3 pipeline above |
| 9 | `figures/20260907_figS06_expt3_fit.ipynb` | needs the Expt 3 pipeline above |
| 10 | `figures/20260907_figS03_expt3_qc.ipynb` | independent |
| 11 | `figures/20260907_fig01_mechanisms.ipynb` | simulation only, no data — ~2 min |
| 12 | `figures/20260907_table01_bootstrap_cis.ipynb` | SI Table 1 |

Headless, in that order:

```bash
jupyter nbconvert --to notebook --execute --inplace figures/20260907_figS04_expt1_fit.ipynb
```

## Layout

```
barfly/            library
  ld.py            the Luria-Delbruck distribution: simulation, CCDF, Laplace transform, binning
  fitting.py       least-squares, Laplace-space and likelihood fits
  expt1/2/3.py     per-experiment definitions and loaders; expt3 also holds its batch fit
  io.py            count-table readers and the lineage filters
  plotting.py      shared style and palettes
figures/           one notebook per manuscript figure, plus the Expt 3 fit pipeline as a script
pipeline/          reads -> barcode counts (needs the raw FASTQ, not included)
data/              processed inputs, ~100 MB
cache/             intermediates written by the fit notebooks and script
output/            panel SVGs
```

## The three experiments

| shorthand | experiment | figures |
|---|---|---|
| **Expt 1** | single-strain BY4741 *msh2Δ*, canavanine selection | Fig. 2, Fig. 5, S1, S4 |
| **Expt 2** | wildtype / *msh6Δ* / *msh2Δ* | Fig. 3, S2, S5 |
| **Expt 3** | UV: no-UV control + 2 UV populations | Fig. 4, S3, S6 |

## Data

| file | what it is |
|---|---|
| `data/expt1/D709_D50{1..6}_errorCorrectedBcds.npz` | Expt 1, scipy sparse, one per timepoint. `D501` = t0, `D502` = a technical replicate of t0, `D503`–`D506` = t1–t4. |
| `data/expt2/20250725_merged_counts.pkl` | Expt 2, all six timepoints x three strains |
| `data/expt2/20260625_bestfit_params.pkl` | Expt 2 experimental constants and published best-fit values |
| `data/expt3/20260224_merged_counts.pkl` | Expt 3, four timepoints x three populations |
| `data/bootstrap/20260801_expt{1,2,3}_bootstrap.npz` | 10,000 bootstrap replicates each |

The bootstrap replicates were generated separately and are provided as-is; the table notebook
reads them rather than regenerating them.
