"""The Luria-Delbruck distribution: simulation, CCDF, Laplace transform, binning."""

import numba
import numpy as np
import mpmath as mp


@numba.njit(cache=True)
def seed_simulator(seed):
    """Seed the random stream `simulate_ld` draws from.

    Numba-compiled code keeps its own generator state, separate from the one
    `numpy.random.seed` controls, so seeding a run has to happen from inside compiled
    code.  Without this a fit is not reproducible between runs, which matters wherever
    the loss surface is shallow enough for Monte-Carlo noise to move the optimum
    between adjacent grid points.
    """
    np.random.seed(seed)


@numba.njit(cache=True)
def simulate_ld(mu, sigma, n_replicates, population_size):
    """Sample mutant numbers from the Luria-Delbruck distribution.

    Follows the algorithm of SI section 1.5.  A culture grows from one wildtype cell
    to `population_size`; mutations occur at rate `mu` per cell per generation and
    each founds a lineage that grows with rate ratio `sigma` relative to wildtype:

      1. the number of mutations is Poisson with mean mu * N;
      2. each mutation time t is drawn from P(t) proportional to exp(lambda0 t) by
         inverse transform, t = ln(1 + (N - 1) u) for u uniform on (0, 1);
      3. the mutant number is the sum of exp(sigma * (T - t)) over mutations.

    Time is measured in wildtype generations (lambda0 = 1), so the final time is
    T = ln(N).  Early mutations therefore dominate: they contribute the jackpots that
    give the distribution its power-law tail.

    sigma = 1 for neutral mutants, which is the case throughout except in Fig. 1d,
    where reversible phenotypic switching is folded into an effective sigma.
    """
    final_time = np.log(population_size)
    mutant_numbers = np.zeros(n_replicates)
    n_mutations = np.random.poisson(mu * population_size, n_replicates)
    for replicate in range(n_replicates):
        uniform = np.random.random(n_mutations[replicate])
        mutation_times = np.log(1 + (population_size - 1) * uniform)
        mutant_numbers[replicate] = np.sum(
            np.exp(sigma * (final_time - mutation_times)))
    return mutant_numbers


def simulate_ld_extended(mu_per_class, sigma_per_class, mu_induced=0.0,
                         phenotypic_delay=0.0, n_replicates=10000,
                         population_size=1e6, rng=None):
    """Luria-Delbruck sampling with induced mutations, fitness and a phenotypic delay.

    The three departures from the canonical model, all used in Fig. 1 (SI section 1.6):

    induced mutations (`mu_induced`)
        Mutations caused by the selective environment itself.  They arise at the very
        end of growth, so each yields exactly one mutant rather than a clone, and the
        contribution is Poisson with mean mu_induced * N -- extra mutants, but no
        extra jackpots.

    mutant fitness (`sigma_per_class`)
        The ratio of mutant to wildtype growth rate.  sigma < 1 steepens the tail to
        m^(-1/sigma).  It also absorbs reversible phenotypic switching: a state
        inherited with reversion rate nu grows at lambda - nu, so for lineage-size
        purposes sigma_eff = (lambda - nu) / lambda0, and sigma = 0.7 corresponds to
        nu = 0.3 at lambda = lambda0, roughly a three-generation memory.

    phenotypic delay (`phenotypic_delay`)
        Mutants express resistance only after a lag, so only mutations arising before
        T - t_delay are counted.  Implemented by shrinking the mutation window to
        exp(T - t_delay), which removes the most recent (and most numerous) mutations
        while leaving the jackpots intact.

    `mu_per_class` and `sigma_per_class` are parallel sequences, allowing several
    classes of spontaneous mutation with different fitnesses to coexist.
    """
    rng = rng if rng is not None else np.random.default_rng()
    final_time = np.log(population_size)

    # only mutations inside this window can express the phenotype by the final time
    window_population = np.exp(final_time - phenotypic_delay)

    n_classes = len(mu_per_class)
    n_mutations = np.zeros([n_classes, n_replicates])
    for cls, mu in enumerate(mu_per_class):
        n_mutations[cls, :] = rng.poisson(mu * window_population, size=n_replicates)

    mutation_times = {}
    for cls in range(n_classes):
        most_mutations = int(np.max(n_mutations[cls, :]))
        uniform = rng.random([n_replicates, max(most_mutations, 1)])
        mutation_times[cls] = np.log(1 + (window_population - 1) * uniform)

    mutant_numbers = np.zeros(n_replicates)
    for replicate in range(n_replicates):
        for cls in range(n_classes):
            times = mutation_times[cls][replicate,
                                        0:int(n_mutations[cls, replicate])]
            mutant_numbers[replicate] += np.sum(
                np.exp(sigma_per_class[cls] * (final_time - times)))

    return mutant_numbers + rng.poisson(mu_induced * population_size, n_replicates)


def ccdf_of_sorted(ordered, thresholds):
    """Complementary CDF P(x_i > x) of an already-sorted sample.

    Counting with `side="right"` counts samples <= x, so subtracting from n gives the
    strict inequality.  The count stays an integer until the single division, which
    makes this bit-identical to summing the boolean mask directly.
    """
    n = len(ordered)
    return (n - np.searchsorted(ordered, thresholds, side="right")) / n


def ccdf(samples, thresholds):
    """Complementary CDF P(samples > x) at each x in `thresholds`.

    Sorting once and binary-searching beats one pass per threshold, which matters
    because this runs on simulated samples of several million for every point of a
    parameter grid.  Callers holding an already-sorted sample should use
    `ccdf_of_sorted` instead of paying for the sort twice.
    """
    return ccdf_of_sorted(np.sort(samples), thresholds)


def log_laplace_transform(mu_times_n, s):
    """ln of the Laplace transform of the LD distribution at sigma = 1.

    SI Eq. 8 with sigma = 1: ln P~(s) = -mu N (1 - E_2(s)), where E_2 is the
    generalised exponential integral.  mu and N enter only through their product, and
    only as an overall scale -- so the transform is linear in mu*N, which is what
    makes the fit in Laplace space a one-parameter problem.
    """
    return -mu_times_n + mu_times_n * mp.expint(2, s)


def unit_laplace_transform(s_values):
    """`log_laplace_transform` at mu*N = 1, as float64.

    The prediction for any mu*N is this curve scaled by mu*N.
    """
    return np.array([float(log_laplace_transform(1, s)) for s in s_values])


def empirical_laplace_transform(mutant_numbers, s_values):
    """ln <exp(-s m)> over the observed mutant numbers, at each s."""
    return np.log(np.mean(
        np.exp(-s_values[:, np.newaxis] * mutant_numbers[np.newaxis, :]), axis=1))


def bin_barcodes(counts_pre, counts_post, n_bins):
    """Group barcodes into `n_bins` virtual lineages of near-equal founding size.

    Barcode abundances span more than an order of magnitude, so the population size N
    per lineage is far from uniform -- while the theory assumes a common N.  The
    Luria-Delbruck distribution is approximately stable under addition (SI section
    1.4): summing lineages of sizes N_1, N_2, ... gives the distribution for a single
    lineage of size N_1 + N_2 + ...  Combining barcodes into bins balanced on their
    t0 counts therefore yields virtual lineages that share a common N and can be
    compared against theory directly.

    Greedy list scheduling: barcodes are taken in increasing order of t0 count and
    each is added to the currently smallest bin.  Returns
    (bin_members, binned_pre, binned_post).
    """
    counts_pre = np.asarray(counts_pre)
    counts_post = np.asarray(counts_post)

    binned_pre = np.zeros(n_bins)
    bin_members = {b: [] for b in np.arange(n_bins)}
    for barcode in np.argsort(counts_pre):
        smallest = np.argmin(binned_pre)
        binned_pre[smallest] += counts_pre[barcode]
        bin_members[smallest].append(barcode)

    binned_post = np.zeros(n_bins)
    for b in np.arange(n_bins):
        binned_post[b] = np.sum(counts_post[bin_members[b]])

    return bin_members, binned_pre, binned_post
