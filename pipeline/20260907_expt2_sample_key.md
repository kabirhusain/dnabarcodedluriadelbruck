# Expt 2 sample key

Eighteen libraries: three strains x six timepoints. The sequencing index pair is
the sample key throughout; downstream code appends `_f1r1`.

**Each strain was run with a different R1 UMI length.** This is not cosmetic — it
is why there are three extraction scripts, and it sets the barcode's offset
within R1 (UMI length + 16). The R2 UMI is 8 nt in all three.

| strain | genotype | index pairs | R1 UMI | extraction script |
|---|---|---|---|---|
| yKH88 | wild type | `D765-D566` … `D770-D571` | 8 nt | `20260908_expt2_00_extract_barcodes_8N.py` |
| yFR1 | *msh6Δ* | `D771-D572` … `D776-D577` | 9 nt | `20260908_expt2_00_extract_barcodes_9N.py` |
| yFR2 | *msh2Δ* | `D777-D578` … `D782-D583` | 10 nt | `20260908_expt2_00_extract_barcodes_10N.py` |

```py
# yKH88, 8N
dlist = ["D765-D566", "D766-D567", "D767-D568", "D768-D569", "D769-D570", "D770-D571"]
# yFR1, 9N
dlist = ["D771-D572", "D772-D573", "D773-D574", "D774-D575", "D775-D576", "D776-D577"]
# yFR2, 10N
dlist = ["D777-D578", "D778-D579", "D779-D580", "D781-D582", "D782-D583"]   # note: 5, not 6
# yFR2, 10N, the sixth
dlist = ["D780-D581"]
```

`D780-D581` is missing from the 10N script's `dlist` and was extracted separately
by `20260908_expt2_00_extract_barcodes_10N_lastone.py`, then merged into the
combined pickle by `20260908_expt2_01_merge_barcodes_lastone.py`. All 18 samples
are present in `data/expt2/20250725_merged_counts.pkl`; the split is a record of
how the run actually went, not a gap.

Timepoints, in `dlist` order within each strain:

| index | date |
|---|---|
| 0 | 5/1 |
| 1 | 5/5 |
| 2 | 5/6 |
| 3 | 5/7 |
| 4 | 5/8 |
| 5 | 5/11 |

So `D780-D581` is yFR2 at timepoint 3 (5/7).
