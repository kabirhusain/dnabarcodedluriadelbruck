"""Expt 3: the UV experiment, separating spontaneous from induced mutagenesis.

Three populations of barcoded BY4741 were grown to saturation.  Two were exposed to
302 nm UV for 30 minutes immediately before canavanine selection; the third was not.
UV therefore mimics a drug that both induces resistance and selects for it, adding
induced mutants on top of the spontaneous background.

Induced mutations are distinguishable in the mutant-number distribution because they
arise at the very end of growth: the cells carrying them have no time to divide, so
each induced mutation contributes exactly one mutant instead of a clone.  Writing mu
for the spontaneous rate and mu_ind for the induced rate per cell (SI Eq. 11),

    m = m_LD + m_ind ,     m_ind ~ Poisson(mu_ind * N)

so induced mutagenesis shifts the body of the distribution without adding jackpots.
Setting mu_ind = 0 recovers the spontaneous-only model, making the two nested and
directly comparable.

The fit proceeds in three steps:

  1. alpha for the whole experiment, from a least-squares CCDF fit of the *unexposed*
     population under mu_ind = 0.  Fixing the read-to-mutant calibration on a
     population known to have no induced component is what makes mu_ind identifiable
     in the other two.
  2. a least-squares fit of (mu, mu_ind) at that fixed alpha, for all three
     populations and every bin count.
  3. a maximum-likelihood fit of the same, over the same grid.

Both estimators share one simulated distribution library, since alpha rescales only
the data and the Poisson component can be added to an existing LD sample.
"""

import heapq
import pickle
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import load_merged_counts, umi_counts
from .ld import seed_simulator, simulate_ld

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "expt3"

# Sequencing index pairs per population, in timepoint order, with the t0 read cutoff
# and the plating measurements.  totalN is the pre-selection population; totalMutants
# is the plated mutant count M_cfu.  Cutoffs differ because library depth did.
POPULATIONS = {
    "20260224_yKH88": dict(
        label="BY4741",
        samples=["D756-D560", "D757-D561", "D758-D562", "D759-D563"],
        min_reads_t0=100.0, total_n=1.2e10, total_mutants=20520.0),
    "20260224_yKH88UV_A": dict(
        label="BY4741 + UV repl. 1",
        samples=["D761-D565", "D762-D566", "D763-D567", "D764-D568"],
        min_reads_t0=170.0, total_n=1.2e10, total_mutants=87360.0),
    "20260224_yKH88UV_B": dict(
        label="BY4741 + UV repl. 2",
        samples=["D707-D506", "D767-D570", "D768-D571", "D769-D572"],
        min_reads_t0=30.0, total_n=1.2e10, total_mutants=26280.0),
}
KEYS = list(POPULATIONS)
KEY_LABELS = {key: spec["label"] for key, spec in POPULATIONS.items()}

# Parameter grids.  mu_ind includes an exact zero so the spontaneous-only model is a
# point on the same grid rather than a separate fit.
MU_GRID = np.logspace(-8, -5, 50)
MU_IND_GRID = np.concatenate([[0.0], np.logspace(-10, -5, 50)])
ALPHA_GRID = np.logspace(np.log10(0.1), np.log10(10.0), 100)
BIN_COUNTS = [1000, 2000, 4000, 6000, 8000, 12000, 16000, 25000, 32000]

N_REPLICATES = 3_000_000
N_CCDF_POINTS = 40
N_LIKELIHOOD_BINS = 50
LOW_CUTOFF = 1
MIN_PROBABILITY = 1e-12

# Shared fine histogram edges for every simulated distribution.  The leading 0 -> 0.5
# bin captures lineages with no mutants; the log-spaced remainder runs to 1e12, above
# any attainable mutant number, so one set of edges serves every (mu, mu_ind, N).
FINE_EDGES = np.concatenate([[0.0], np.logspace(np.log10(0.5), np.log10(1e12), 300)])


@dataclass
class Population:
    """One UV or control population, filtered to genuine lineages."""
    key: str
    counts_t0: np.ndarray
    counts_tf: np.ndarray
    total_n: float
    total_mutants: float
    n_lineages: int
    reads_tf_total: float


def _timecourse_counts(merged_counts, spec):
    """Counts over the timecourse for one population, and its t0 filter mask.

    Barcodes are those present in the t0 sample; later timepoints contribute zero for
    barcodes they never saw.
    """
    samples = [f"{sample}_f1r1" for sample in spec["samples"]]
    barcodes = sorted(merged_counts[samples[0]]["umiCounts"])
    counts = np.array([umi_counts(merged_counts, sample, barcodes)
                       for sample in samples], dtype=np.float64)
    return counts, counts[0] > spec["min_reads_t0"]


def load_populations(merged_counts_path=None):
    """Rebuild the three filtered populations from the merged counts.

    The barcode set is taken from the t0 sample, and a lineage is kept if its t0 count
    exceeds the population's cutoff.  No upper cutoff is applied: unlike the
    mismatch-repair experiments, these libraries show no separated outlier mode of
    pre-existing mutants.
    """
    merged_counts = load_merged_counts(
        merged_counts_path or DATA_DIR / "20260224_merged_counts.pkl")

    populations = {}
    for key, spec in POPULATIONS.items():
        counts, keep = _timecourse_counts(merged_counts, spec)
        populations[key] = Population(
            key=key, counts_t0=counts[0][keep], counts_tf=counts[-1][keep],
            total_n=spec["total_n"], total_mutants=spec["total_mutants"],
            n_lineages=int(keep.sum()), reads_tf_total=float(counts[-1][keep].sum()))
    return populations


def load_timecourse(merged_counts_path=None):
    """Unfiltered counts and the filter mask per population, for the QC trajectories.

    Returns {key: (counts, mask)} with counts of shape (timepoints, barcodes).
    """
    merged_counts = load_merged_counts(
        merged_counts_path or DATA_DIR / "20260224_merged_counts.pkl")
    timecourse = {}
    for key, spec in POPULATIONS.items():
        timecourse[key] = _timecourse_counts(merged_counts, spec)
    return timecourse


def bin_barcodes(counts_t0, n_bins, rng=None):
    """Group barcodes into `n_bins` virtual lineages balanced on t0 count.

    Same greedy least-loaded scheduling as `barfly.ld.bin_barcodes`, but with a heap
    for the bin totals and a tiny jitter on the sort key.  These libraries have deep
    ties in the t0 counts, and the jitter breaks them reproducibly instead of leaving
    the order to the sort's internal behaviour.

    Returns a list of barcode-index lists, one per bin.
    """
    n_barcodes = counts_t0.size
    if n_bins >= n_barcodes:
        return [[int(b)] for b in np.argsort(counts_t0, kind="stable")]

    rng = rng if rng is not None else np.random.default_rng(0)
    jitter = rng.uniform(0.0, 1e-9, size=n_barcodes)
    order = np.argsort(counts_t0 + jitter, kind="stable")

    totals = [(0.0, b, b) for b in range(n_bins)]
    heapq.heapify(totals)
    bins = [[] for _ in range(n_bins)]
    for barcode in order:
        total, _, bin_id = heapq.heappop(totals)
        bins[bin_id].append(int(barcode))
        heapq.heappush(totals, (total + float(counts_t0[barcode]), bin_id, bin_id))
    return bins


def mutants_per_bin(bins, counts_tf, total_mutants, reads_tf_total):
    """Mutant numbers per virtual lineage at alpha = 1."""
    per_bin = np.zeros(len(bins))
    for i, barcodes in enumerate(bins):
        per_bin[i] = counts_tf[np.asarray(barcodes, dtype=np.int64)].sum()
    return total_mutants * per_bin / reads_tf_total


def ccdf_thresholds(mutants_per_bin_at_alpha1):
    """Mutant numbers at which data and theory CCDFs are compared.

    An anchor at 0.5 plus log-spaced points up to the eleventh largest mutant number.
    Fixed in absolute mutant number rather than scaled with alpha, so that rescaling
    the data genuinely constrains mu*N instead of sliding the whole comparison.
    """
    sorted_mutants = np.sort(mutants_per_bin_at_alpha1)
    upper = (float(sorted_mutants[-11]) if sorted_mutants.size > 10
             else float(sorted_mutants.max()))
    return np.concatenate([[0.5],
        np.logspace(np.log10(1.001), np.log10(max(upper, 1.01)), N_CCDF_POINTS - 1)])


def ccdf_of_samples(samples, thresholds):
    """Empirical P(m > x) of a sample at each threshold."""
    ordered = np.sort(samples)
    return 1.0 - np.searchsorted(ordered, thresholds, side="right") / ordered.size


def ccdf_of_histogram(fine_pmf, fine_edges, thresholds):
    """Theory P(m > x) at each threshold, from a simulated fine histogram."""
    cdf = np.concatenate([[0.0], np.cumsum(fine_pmf)])
    return 1.0 - np.interp(thresholds, fine_edges, cdf)


def rebin_histogram(fine_pmf, fine_edges, coarse_edges):
    """Re-express a fine simulated histogram on arbitrary coarser bins."""
    cdf = np.concatenate([[0.0], np.cumsum(fine_pmf)])
    return np.diff(np.interp(coarse_edges, fine_edges, cdf))


def ccdf_squared_error(observed_ccdf, theory_ccdf, n_replicates=N_REPLICATES):
    """Squared error between log10 CCDFs, ignoring theory below the noise floor.

    A simulated CCDF below ~5/n_replicates is indistinguishable from zero, and taking
    its logarithm would let simulation noise dominate the loss.
    """
    noise_floor = 5.0 / n_replicates
    scored = (observed_ccdf > 0) & (theory_ccdf > noise_floor)
    if scored.sum() == 0:
        return np.inf
    residual = np.log10(observed_ccdf[scored]) - np.log10(theory_ccdf[scored])
    return float(np.sum(residual ** 2))


def likelihood_bin_edges(mutant_numbers):
    """Coarse bins for the likelihood: one below `LOW_CUTOFF`, log-spaced, then one above.

    The likelihood needs a probability per observation, so the mutant numbers are
    discretised.  The overflow bins keep every observation inside the support.
    """
    largest = max(float(mutant_numbers.max()), LOW_CUTOFF * 2.0)
    log_bins = np.logspace(np.log10(LOW_CUTOFF), np.log10(largest * 2.0),
                           N_LIKELIHOOD_BINS)
    return np.concatenate(([0.0], log_bins, [max(largest * 1000.0, 1e12)]))


def block_seed(key, n_bins, i_mu):
    """Deterministic per-task seed.

    Python's `hash()` of a string is salted per interpreter, so seeding from it makes
    the library -- and every fit downstream of it -- irreproducible between runs.  A
    checksum of the same identifying fields is stable across processes and machines.
    """
    return zlib.crc32(f"{key}|{n_bins}|{i_mu}".encode()) % (2 ** 31)


def _simulate_block(task):
    """Simulate one (population, bin count, mu) and histogram it for every mu_ind.

    One LD sample is drawn and reused across the mu_ind grid, since the induced
    component is an independent Poisson draw added to the same spontaneous mutants.
    Module-level for pickling into worker processes.
    """
    key, n_bins, i_mu, mu, population_per_bin, n_replicates, seed = task
    rng = np.random.default_rng(seed)
    seed_simulator(seed)
    spontaneous = simulate_ld(float(mu), 1.0, n_replicates, float(population_per_bin))

    histograms = np.zeros((len(MU_IND_GRID), len(FINE_EDGES) - 1), dtype=np.float32)
    for i_mu_ind, mu_ind in enumerate(MU_IND_GRID):
        if mu_ind == 0.0:
            combined = spontaneous
        else:
            induced = rng.poisson(float(mu_ind) * float(population_per_bin),
                                  n_replicates)
            combined = spontaneous + induced.astype(np.float64)
        counts, _ = np.histogram(combined, bins=FINE_EDGES)
        histograms[i_mu_ind] = (counts / n_replicates).astype(np.float32)
    return key, n_bins, i_mu, histograms


def build_distribution_library(populations, n_workers=None, progress=True):
    """Simulate P(m) over (population, bin count, mu, mu_ind).

    Returns {(key, n_bins): {mutants_at_alpha1, population_per_bin, n_bins,
    thresholds, fine_pmf}} where `fine_pmf` is indexed (mu, mu_ind, fine bin).
    About 82 MB, and the dominant cost of the whole Expt 3 analysis.
    """
    library, tasks = {}, []
    for key in KEYS:
        population = populations[key]
        for n_bins in BIN_COUNTS:
            if n_bins > population.n_lineages:
                continue
            bins = bin_barcodes(population.counts_t0, n_bins)
            mutants = mutants_per_bin(bins, population.counts_tf,
                                      population.total_mutants,
                                      population.reads_tf_total)
            population_per_bin = population.total_n / n_bins
            library[(key, n_bins)] = {
                "mutants_at_alpha1": mutants,
                "population_per_bin": float(population_per_bin),
                "n_bins": n_bins,
                "thresholds": ccdf_thresholds(mutants),
                "fine_pmf": np.zeros((len(MU_GRID), len(MU_IND_GRID),
                                      len(FINE_EDGES) - 1), dtype=np.float32),
            }
            for i_mu, mu in enumerate(MU_GRID):
                tasks.append((key, n_bins, i_mu, float(mu), float(population_per_bin),
                              N_REPLICATES, block_seed(key, n_bins, i_mu)))

    if progress:
        print(f"{len(tasks)} simulations over {len(library)} (population, bin count) blocks")
    started, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_simulate_block, task) for task in tasks]
        for future in as_completed(futures):
            key, n_bins, i_mu, histograms = future.result()
            library[(key, n_bins)]["fine_pmf"][i_mu] = histograms
            done += 1
            if progress and (done % 25 == 0 or done == len(tasks)):
                elapsed = time.time() - started
                remaining = elapsed / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)}  elapsed={elapsed:.0f}s  "
                      f"ETA={remaining:.0f}s", flush=True)
    return library


def ccdf_loss_surface(library, key, n_bins, alpha):
    """Least-squares loss over the (mu, mu_ind) grid for one block."""
    block = library[(key, n_bins)]
    observed = ccdf_of_samples(alpha * block["mutants_at_alpha1"], block["thresholds"])
    loss = np.full((len(MU_GRID), len(MU_IND_GRID)), np.inf)
    for i_mu in range(len(MU_GRID)):
        for i_mu_ind in range(len(MU_IND_GRID)):
            theory = ccdf_of_histogram(block["fine_pmf"][i_mu, i_mu_ind],
                                       FINE_EDGES, block["thresholds"])
            loss[i_mu, i_mu_ind] = ccdf_squared_error(observed, theory)
    return loss


def log_likelihood_surface(library, key, n_bins, alpha):
    """Log-likelihood over the (mu, mu_ind) grid for one block (SI Eq. 12)."""
    block = library[(key, n_bins)]
    observed = alpha * block["mutants_at_alpha1"]
    edges = likelihood_bin_edges(observed)
    bin_of_lineage = np.clip(np.digitize(observed, edges) - 1, 0, len(edges) - 2)

    log_likelihood = np.zeros((len(MU_GRID), len(MU_IND_GRID)))
    for i_mu in range(len(MU_GRID)):
        for i_mu_ind in range(len(MU_IND_GRID)):
            pmf = rebin_histogram(block["fine_pmf"][i_mu, i_mu_ind], FINE_EDGES, edges)
            pmf = np.clip(pmf, MIN_PROBABILITY, None)
            pmf /= pmf.sum()
            log_likelihood[i_mu, i_mu_ind] = np.log(pmf)[bin_of_lineage].sum()
    return log_likelihood


def fit_alpha(library, control_key="20260224_yKH88"):
    """Calibrate alpha on the unexposed control, under the spontaneous-only model.

    Fitted separately at each bin count, then summarised by the median.  Returns
    (alpha, per-bin-count rows).
    """
    rows, per_bin_count = [], []
    for n_bins in BIN_COUNTS:
        if (control_key, n_bins) not in library:
            continue
        block = library[(control_key, n_bins)]
        thresholds = block["thresholds"]
        spontaneous_only = [
            ccdf_of_histogram(block["fine_pmf"][i_mu, 0], FINE_EDGES, thresholds)
            for i_mu in range(len(MU_GRID))]

        best_loss, best_mu, best_alpha = np.inf, np.nan, np.nan
        for alpha in ALPHA_GRID:
            observed = ccdf_of_samples(alpha * block["mutants_at_alpha1"], thresholds)
            for i_mu in range(len(MU_GRID)):
                loss = ccdf_squared_error(observed, spontaneous_only[i_mu])
                if loss < best_loss:
                    best_loss, best_mu, best_alpha = loss, MU_GRID[i_mu], alpha

        per_bin_count.append(best_alpha)
        rows.append(dict(key=control_key, k=n_bins, mu=best_mu, alpha=best_alpha,
                         ss=best_loss))
    return float(np.median(per_bin_count)), rows


def _best_fit_rows(library, alpha, surface, take_best):
    """Tabulate the best (mu, mu_ind) per population and bin count for one estimator.

    `mu_best_muind0` is the best mu on the mu_ind = 0 slice, i.e. the spontaneous-only
    model, which is what Fig. 4c and SI Fig. S6c show.
    """
    rows = []
    for key in KEYS:
        for n_bins in BIN_COUNTS:
            if (key, n_bins) not in library:
                continue
            values = surface(library, key, n_bins, alpha)
            i_mu, i_mu_ind, i_mu_spontaneous_only, extreme = take_best(values)
            rows.append(dict(key=key, date=key.split("_")[0], k=n_bins, alpha=alpha,
                             mu_best=MU_GRID[i_mu],
                             mu_ind_best=MU_IND_GRID[i_mu_ind],
                             mu_best_muind0=MU_GRID[i_mu_spontaneous_only],
                             **extreme))
    return rows


def fit_least_squares(library, alpha):
    """Least-squares fit of (mu, mu_ind) at fixed alpha, per population and bin count."""
    def take_best(loss):
        i_mu, i_mu_ind = np.unravel_index(np.nanargmin(loss), loss.shape)
        return (i_mu, i_mu_ind, int(np.nanargmin(loss[:, 0])),
                dict(ss_min=loss[i_mu, i_mu_ind], ss_muind0=loss[:, 0].min()))
    return _best_fit_rows(library, alpha, ccdf_loss_surface, take_best)


def fit_maximum_likelihood(library, alpha):
    """Maximum-likelihood fit of (mu, mu_ind) at fixed alpha."""
    def take_best(log_likelihood):
        i_mu, i_mu_ind = np.unravel_index(np.argmax(log_likelihood),
                                          log_likelihood.shape)
        return (i_mu, i_mu_ind, int(np.argmax(log_likelihood[:, 0])),
                dict(nll_min=-log_likelihood[i_mu, i_mu_ind]))
    return _best_fit_rows(library, alpha, log_likelihood_surface, take_best)
