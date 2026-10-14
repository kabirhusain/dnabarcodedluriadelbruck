"""Fitting the Luria-Delbruck distribution: least squares, Laplace space, likelihood.

Two parameters are fitted throughout:

  mu     the rate at which adaptive mutations arise, per cell per generation
  alpha  the calibration from read fraction to absolute mutant number,
         m_i = alpha * (r_i / sum_j r_j) * M_cfu   (SI Eq. 10)

M_cfu is the total mutant count from plating.  Some viable cells fail to form
colonies, so M_cfu underestimates the true total and alpha is expected above 1; it is
fitted over [0.1, 10] and comes out between 1 and 5 for every dataset here.

A third parameter appears only for the UV experiment:

  mu_ind  the probability that the selective environment induces resistance in a
          previously wildtype cell (see `barfly.expt3`)
"""

import zlib

import numpy as np

from .ld import (ccdf, empirical_laplace_transform, seed_simulator,
                 simulate_ld, unit_laplace_transform)

# CCDF values below this are treated as empty rather than scored in log space
MIN_CCDF = 1e-6


def ccdf_thresholds(mutants_by_bin_count, n_log_points=30):
    """Mutant numbers at which data and theory CCDFs are compared.

    One point at 0.5, so that the fraction of lineages with no mutants at all
    constrains alpha, then log-spaced points up to the tenth largest observed mutant
    number.  Truncating the last few jackpots keeps the noisiest points out of the
    loss without discarding the tail that carries the signal.

    Returns (thresholds, largest_scored_mutant_number).
    """
    all_mutants = np.concatenate(
        [mutants_by_bin_count[k] for k in sorted(mutants_by_bin_count)])
    upper = np.sort(all_mutants)[::-1][9]
    thresholds = np.concatenate([[0.5], np.logspace(0, np.log10(upper), n_log_points)])
    return thresholds, upper


def simulate_over_mu(mu_values, bin_counts, total_population, n_replicates=1_000_000,
                     sigma=1, label=""):
    """Simulate the LD distribution for every (mu, bin count), at N = total / k.

    Independent of alpha, which rescales only the data, so this is computed once and
    reused across the whole alpha grid.  Returns simulations[i_mu][i_k].

    Each (mu, bin count) is seeded from a checksum of its own indices and `label`, so
    a run is reproducible and does not depend on how much simulation preceded it.
    That matters because the least-squares fit can sit in a shallow valley where
    Monte-Carlo noise moves the optimum between adjacent grid points.
    """
    simulations = {}
    for i_mu, mu in enumerate(mu_values):
        simulations[i_mu] = {}
        for i_k, n_bins in enumerate(bin_counts):
            seed_simulator(
                zlib.crc32(f"{label}|{i_mu}|{n_bins}".encode()) % (2 ** 31))
            simulations[i_mu][i_k] = simulate_ld(
                mu, sigma, n_replicates, total_population / n_bins)
    return simulations


def ccdf_loss(mutants_by_bin_count, simulations, mu_values, alpha_values, thresholds):
    """Squared error between log10 data and theory CCDFs, per bin count.

    Returns an array indexed (bin count, mu, alpha); the global loss is its sum over
    bin counts.  Only thresholds where the data CCDF is non-empty contribute, so
    exhausted tail bins are not scored as disagreement.

    The data CCDF depends on alpha and the bin count but not on mu, and the theory
    CCDF on mu and the bin count but not alpha, so both are evaluated once and
    combined.
    """
    n_k = len(mutants_by_bin_count)
    n_mu, n_alpha = len(mu_values), len(alpha_values)

    theory_ccdf = {i_mu: {i_k: ccdf(simulations[i_mu][i_k], thresholds)
                          for i_k in range(n_k)}
                   for i_mu in range(n_mu)}
    data_ccdf = {(i_alpha, i_k): ccdf(alpha * mutants_by_bin_count[i_k], thresholds)
                 for i_alpha, alpha in enumerate(alpha_values)
                 for i_k in range(n_k)}

    loss = np.zeros((n_k, n_mu, n_alpha))
    for i_alpha in range(n_alpha):
        for i_k in range(n_k):
            observed = data_ccdf[(i_alpha, i_k)]
            scored = observed > MIN_CCDF
            if not scored.any():
                continue
            log_observed = np.log10(np.maximum(observed[scored], MIN_CCDF))
            for i_mu in range(n_mu):
                log_theory = np.log10(
                    np.maximum(theory_ccdf[i_mu][i_k][scored], MIN_CCDF))
                loss[i_k, i_mu, i_alpha] = np.sum((log_observed - log_theory) ** 2)
    return loss


def laplace_loss(mutants_by_bin_count, bin_counts, mu_values, alpha_values,
                 total_population, s_values=None):
    """Squared error between log(-ln P~(s)) of data and theory, per bin count.

    An independent estimator to the CCDF fit, exploiting a different feature of the
    data: the Laplace transform has the closed form of SI Eq. 8, so no simulation is
    needed and the theory is exact.  It also never uses the zero-mutant fraction to
    pin alpha, which makes it the more stable of the two when mu*N is large.

    Returns an array indexed (bin count, mu, alpha).
    """
    if s_values is None:
        s_values = np.logspace(-3, -1, 30)
    unit_theory = unit_laplace_transform(s_values)

    loss = np.zeros((len(bin_counts), len(mu_values), len(alpha_values)))
    for i_k, n_bins in enumerate(bin_counts):
        population_per_bin = total_population / n_bins
        # The theory depends on mu and N, the data on alpha; neither depends on the
        # other's index, so both sides are evaluated once per bin count.
        log_theory = [np.log(-mu * population_per_bin * unit_theory)
                      for mu in mu_values]
        for i_alpha, alpha in enumerate(alpha_values):
            log_observed = np.log(-empirical_laplace_transform(
                alpha * mutants_by_bin_count[i_k], s_values))
            for i_mu in range(len(mu_values)):
                loss[i_k, i_mu, i_alpha] = np.sum(
                    (log_observed - log_theory[i_mu]) ** 2)
    return loss


def best_fit(loss_surface, mu_values, alpha_values):
    """The (mu, alpha) minimising a loss surface indexed (mu, alpha)."""
    i_mu, i_alpha = np.unravel_index(np.argmin(loss_surface), loss_surface.shape)
    return mu_values[i_mu], alpha_values[i_alpha]


def best_fit_per_bin_count(loss, mu_values, alpha_values):
    """Best fit for each bin count, with the global (summed-loss) fit appended last.

    Agreement across bin counts is the internal consistency check on the fit: the
    theory should describe the data at every N, not just one.
    """
    n_k = loss.shape[0]
    mu_best = np.zeros(n_k + 1)
    alpha_best = np.zeros(n_k + 1)
    for i_k in range(n_k):
        mu_best[i_k], alpha_best[i_k] = best_fit(loss[i_k], mu_values, alpha_values)
    mu_best[n_k], alpha_best[n_k] = best_fit(loss.sum(axis=0), mu_values, alpha_values)
    return mu_best, alpha_best


def percentile_interval(point, replicates, percentiles=(2.5, 97.5)):
    """Bootstrap confidence interval as distances below and above `point`.

    Returns the 2-row array `matplotlib.errorbar` expects for `yerr`.  Clipped at
    zero: the point estimate can sit marginally outside its own percentile range
    when the parameter grid is coarse, and a negative error bar is meaningless.
    """
    lower, upper = np.percentile(replicates, list(percentiles), axis=0)
    return np.maximum(np.vstack([point - lower, upper - point]), 0)


def p0_mle(produced_mutants, population_size, mu_grid):
    """Mutation rate from the presence or absence of mutants alone.

    The maximum-likelihood form of the classical P0 method: since no mutants means no
    mutations occurred, and mutations are Poisson with mean mu*N,

        P(m = 0) = exp(-mu N)                                     (SI Eq. 3)

    so the likelihood over lineages depends only on the indicator s_i and N_i,

        L(mu) = prod_i [ exp(-mu N_i) (1 - s_i) + (1 - exp(-mu N_i)) s_i ]

    Discarding the mutant counts costs precision but makes the estimator cheap and
    robust to the calibration of large counts.  `population_size` is a scalar or one
    value per lineage.  Returns (best mu, log-likelihood over the grid).
    """
    produced_mutants = np.asarray(produced_mutants, dtype=float)
    population_size = np.broadcast_to(np.asarray(population_size, dtype=float),
                                      produced_mutants.shape)

    log_likelihood = np.empty(len(mu_grid))
    for i, mu in enumerate(mu_grid):
        p_no_mutants = np.exp(-mu * population_size)
        probability = np.where(produced_mutants > 0, 1.0 - p_no_mutants, p_no_mutants)
        log_likelihood[i] = np.sum(np.log(np.maximum(probability, 1e-300)))
    return mu_grid[np.argmax(log_likelihood)], log_likelihood
