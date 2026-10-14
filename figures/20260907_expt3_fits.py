"""Fit Expt 3, separating spontaneous from induced mutagenesis.

Run before Fig. 4 and SI Fig. S6; both read the results from cache/.

Three steps, described in `barfly.expt3`:

  1. alpha for the whole experiment, from the unexposed control under the
     spontaneous-only model.  Calibrating on a population with no induced component
     is what makes mu_ind identifiable in the other two.
  2. a least-squares fit of (mu, mu_ind) at that fixed alpha.
  3. a maximum-likelihood fit of the same.

Step 1 fits alpha per bin count and takes the median.  Each of those fits is
Monte-Carlo noisy and the alpha grid steps by 4.8%, so the median is only stable to
about one grid step -- and because the data are scaled by alpha, a one-step move
propagates into mu.  Steps 2 and 3 therefore use the alpha carried in the bootstrap
results, which was fitted on a grid roughly four times finer with more replicates and
is the value the paper quotes.  The freshly fitted alpha is printed alongside it as a
consistency check, not used for the figures.

The simulated distribution library is shared by both estimators and is the dominant
cost.  It is cached, so a second run skips straight to the fits; pass --force to
rebuild it.

Runtime: ~8 min on 20 cores.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from barfly import expt3                                    # noqa: E402

CACHE_DIR = ROOT / "cache"
LIBRARY_PATH = CACHE_DIR / "expt3_precompute.pkl"
BOOTSTRAP_PATH = ROOT / "data" / "bootstrap" / "20260801_expt3_bootstrap.npz"
BIN_COUNT_FOR_FIGURES = 4000


def load_or_build_library(populations, workers, force):
    """Read the cached distribution library, or simulate and cache it."""
    if LIBRARY_PATH.exists() and not force:
        print(f"loading the distribution library from {LIBRARY_PATH.name}")
        with open(LIBRARY_PATH, "rb") as f:
            return pickle.load(f)["blocks"]

    print("simulating the distribution library")
    started = time.time()
    library = expt3.build_distribution_library(populations, n_workers=workers)
    with open(LIBRARY_PATH, "wb") as f:
        pickle.dump({"FINE_EDGES": expt3.FINE_EDGES, "mu_grid": expt3.MU_GRID,
                     "mu_ind_full": expt3.MU_IND_GRID, "blocks": library}, f)
    print(f"cached to {LIBRARY_PATH.name} "
          f"({LIBRARY_PATH.stat().st_size / 1e6:.0f} MB, {time.time() - started:.0f}s)")
    return library


def main(workers=None, force=False):
    started = time.time()
    CACHE_DIR.mkdir(exist_ok=True)

    populations = expt3.load_populations()
    for key, population in populations.items():
        print(f"{key:24s} lineages={population.n_lineages:6d} "
              f"N={population.total_n:.2e} mutants={population.total_mutants:.2e}")

    library = load_or_build_library(populations, workers, force)
    print(f"{len(library)} (population, bin count) blocks available")

    step = time.time()
    fitted_alpha, alpha_rows = expt3.fit_alpha(library)
    per_bin_count = [row["alpha"] for row in alpha_rows]
    alpha = float(np.load(BOOTSTRAP_PATH)["alpha"])
    print(f"\nstep 1: alpha = {alpha:.4f} (from the bootstrap fit; used below)")
    print(f"        refitted here: {fitted_alpha:.4f} "
          f"(median of {min(per_bin_count):.2f}-{max(per_bin_count):.2f} over "
          f"bin counts)   [{time.time() - step:.0f}s]")
    pd.DataFrame(alpha_rows).to_csv(CACHE_DIR / "expt3_alpha_summary.csv", index=False)

    step = time.time()
    least_squares = pd.DataFrame(expt3.fit_least_squares(library, alpha))
    least_squares.to_csv(CACHE_DIR / "expt3_ls_summary.csv", index=False)
    print(f"step 2: least-squares fit   [{time.time() - step:.0f}s]")

    step = time.time()
    maximum_likelihood = pd.DataFrame(expt3.fit_maximum_likelihood(library, alpha))
    maximum_likelihood.to_csv(CACHE_DIR / "expt3_mle_summary.csv", index=False)
    print(f"step 3: maximum-likelihood fit   [{time.time() - step:.0f}s]")

    print(f"\nbest fits at k = {BIN_COUNT_FOR_FIGURES}, the bin count used for every "
          f"Fig. 4 and S6 panel:")
    for name, table in (("MLE", maximum_likelihood), ("LS ", least_squares)):
        for key in expt3.KEYS:
            row = table.loc[(table.key == key)
                            & (table.k == BIN_COUNT_FOR_FIGURES)].iloc[0]
            print(f"  {name} {key:22s} mu = {row.mu_best:.2e}  "
                  f"mu_ind = {row.mu_ind_best:.2e}  "
                  f"(spontaneous-only model: mu = {row.mu_best_muind0:.2e})")

    print(f"\nDone in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel simulation workers (default: all cores)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the distribution library even if cached")
    arguments = parser.parse_args()
    main(workers=arguments.workers, force=arguments.force)
