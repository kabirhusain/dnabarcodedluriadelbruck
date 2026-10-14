"""Expt 2: the three-strain experiment on wildtype, msh6D and msh2D.

Strain codes are `yKH88` (wildtype BY4741), `yFR1` (msh6D) and `yFR2` (msh2D).  Each
was barcoded separately and taken through the same canavanine selection, so the three
datasets are independent Luria-Delbruck experiments differing only in mismatch-repair
genotype.

Samples are keyed `<Dxxx-Dyyy>_f1r1` by sequencing index pair; the mapping to strain
and timepoint is fixed below.
"""

from pathlib import Path

import numpy as np

from .io import load_merged_counts, umi_counts
from .ld import bin_barcodes

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "expt2"

T0_SAMPLE = {"yKH88": "D765-D566", "yFR1": "D771-D572", "yFR2": "D777-D578"}

# six sequenced timepoints per strain, in order
TIMECOURSE = {
    "yKH88": [f"D7{i}-D5{i + 1}" for i in range(65, 71)],
    "yFR1": [f"D7{i}-D5{i + 1}" for i in range(71, 77)],
    "yFR2": [f"D7{i}-D5{i + 1}" for i in range(77, 83)],
}


def load_params():
    """Experimental constants and published best-fit values, per strain.

    Holds `strains`, the plated mutant counts `CFU`, the pre-selection population
    `totalN`, the bin counts `kvec`, the read cutoffs, and the fitted `best_mu` /
    `best_alpha` (least squares) and `lap_best_mu` (Laplace).
    """
    import pickle
    with open(DATA_DIR / "20260625_bestfit_params.pkl", "rb") as f:
        return pickle.load(f)


def load_counts(params=None):
    """Per-strain counts, filtered to genuine lineages.

    Applies the same two cutoffs as Expt 1 (see `barfly.io.load_expt1`), with the
    upper cutoff set per strain because the mutant load differs by two orders of
    magnitude between wildtype and msh2D.

    Returns (counts_pre, counts_post, reads_post, merged_counts).  As in Expt 1,
    `reads_post` deliberately spans every barcode passing the lower cutoff only, since
    it is the denominator calibrating reads against the plated mutant count.
    """
    params = params or load_params()
    merged_counts = load_merged_counts(DATA_DIR / "20250725_merged_counts.pkl")

    counts_pre, counts_post, reads_post = {}, {}, {}
    for strain in params["strains"]:
        at_t0 = merged_counts[f"{T0_SAMPLE[strain]}_f1r1"]["umiCounts"]
        barcodes = np.array([b for b in at_t0 if at_t0[b] > params["minreads"]])
        pre = np.array([at_t0[b] for b in barcodes])

        post = umi_counts(merged_counts, f"{TIMECOURSE[strain][-1]}_f1r1", barcodes)

        reads_post[strain] = np.sum(post[pre > params["minreads"]])
        keep = (pre > params["minreads"]) & (post < params["uppercutoff"][strain])
        counts_pre[strain] = pre[keep]
        counts_post[strain] = post[keep]
    return counts_pre, counts_post, reads_post, merged_counts
