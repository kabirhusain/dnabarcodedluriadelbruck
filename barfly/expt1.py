"""Expt 1: the single-strain BY4741 msh2D canavanine experiment.

One barcoded library, five sequenced timepoints, and an independent technical
replicate of the first.  Sample `D709_D501` is t0, `D502` its replicate, and
`D503`-`D506` are t1-t4.

This is the experiment Figs. 2 and 5 and SI Figs. S1 and S4 all draw on.  The
plating measurements live here; the calibration that uses them is written out in each
notebook, since it is part of the analysis a reader should see.
"""

from .io import load_expt1

T0_SAMPLE = "D709_D501"
REPLICATE_SAMPLE = "D709_D502"
TIMECOURSE = ["D709_D501", "D709_D503", "D709_D504", "D709_D505", "D709_D506"]

MIN_READS_T0 = 10
UPPER_CUTOFF_TF = 1e5
TOTAL_CELLS = 1e10          # cells in the flask before selection
PLATED_CFU = 3e5            # mutant colonies counted by plating -> M_cfu


def load(min_reads_t0=MIN_READS_T0, upper_cutoff_tf=UPPER_CUTOFF_TF):
    """Counts filtered to genuine lineages; see `barfly.io.load_expt1`."""
    return load_expt1(t0_sample=T0_SAMPLE, tf_sample=TIMECOURSE[-1],
                      min_reads_t0=min_reads_t0, upper_cutoff_tf=upper_cutoff_tf)
