"""Loading barcode count tables, and the filters that define a genuine lineage.

Two on-disk formats:

  scipy sparse .npz, one file per timepoint (Expt 1).  Row index is a barcode id and
  column 0 the UMI count.

  a pickled dict of dicts (Expts 2 and 3), keyed by sample:
  counts[sample] = {'umiCounts': {barcode: n}, 'totalReads': {barcode: n}}

Throughout, "reads" means unique UMIs per barcode, as in the paper.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_expt1(t0_sample="D709_D501", tf_sample="D709_D506", min_reads_t0=10,
               upper_cutoff_tf=1e5, data_dir=None):
    """Expt 1 counts, filtered to genuine lineages (SI section 5.1).

    Two cutoffs, both visible as gaps in the count histograms of SI Fig. S1:

      `min_reads_t0` separates real lineages from the low-count mode produced by
      residual PCR and sequencing error, which peaks near singletons.

      `upper_cutoff_tf` removes a small outlier mode at the final timepoint, which is
      attributed to lineages already carrying a canavanine-resistant mutation before
      the experiment began rather than one arising during growth.  It excludes 52 of
      119,724 barcodes here.

    Returns (barcodes, counts_pre, counts_post, reads_post), where `reads_post` is the
    total final-time reads over every barcode passing the *lower* cutoff only.  That
    is the denominator converting read fraction to mutant number, so it must cover the
    same population the plated mutant count M_cfu was drawn from -- including the
    pre-existing mutants excluded from the fit.
    """
    directory = Path(data_dir) if data_dir else DATA_DIR / "expt1"
    counts_t0 = load_npz(directory / f"{t0_sample}_errorCorrectedBcds.npz")
    counts_tf = load_npz(directory / f"{tf_sample}_errorCorrectedBcds.npz")

    # Pull both columns for every populated row in one indexing call each; element
    # -by-element access on a sparse matrix costs a Python call per barcode, and there
    # are hundreds of thousands of them.
    present = counts_t0.nonzero()[0]
    pre_present = np.asarray(counts_t0[present, 0].todense()).ravel()
    post_present = np.asarray(counts_tf[present, 0].todense()).ravel()

    above_lower = pre_present > min_reads_t0
    keep = above_lower & (post_present < upper_cutoff_tf)

    barcodes = present[keep]
    counts_pre = pre_present[keep]
    counts_post = post_present[keep]
    reads_post = np.sum(post_present[above_lower])
    return barcodes, counts_pre, counts_post, reads_post


def load_merged_counts(path):
    """Read a merged-counts pickle produced by the barcode merging step."""
    with open(path, "rb") as f:
        return pickle.load(f)


def umi_counts(merged_counts, sample, barcodes):
    """UMI counts for `barcodes` in one sample, as a dense array (0 where absent)."""
    counts = merged_counts[sample]["umiCounts"]
    return np.array([counts.get(b, 0) for b in barcodes])
