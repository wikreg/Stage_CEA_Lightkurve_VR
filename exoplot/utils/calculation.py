import numpy as np
import emcee
import sys
from scipy.optimize import minimize
from functools import partial
from .mcmc import log_likelihood
from .constants import *
import multiprocessing


# ===========================================================
# Helpers
# ===========================================================

def neg_loglike(params, loglike):
    """
    Negative log-likelihood for minimization.
    Returns a large number if likelihood is not finite.
    """
    val = -loglike(params)
    if not np.isfinite(val):
        return 1e10
    return val


def optimize_initial_guess(loglike):
    """
    Perform initial optimization to find starting point.
    """
    result = minimize(
        neg_loglike, X0, bounds=BOUNDS, method="Powell", args=(loglike,)
    )
    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)
    return result.x


def compute_summary(flat_samples):
    """
    Compute median and uncertainties for each parameter.
    """
    results = {}
    ndim = flat_samples.shape[1]
    for i in range(ndim):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        results[LABELS[i]] = (q50, q84 - q50, q50 - q16)
    return results


# ===========================================================
# Main MCMC Function
# ===========================================================

def terminal_progress_callback(step, total):
    """
    Prints progress to the terminal.
    """
    percent = 100 * step / total
    bar = "#" * int(percent // 2)  # 50 chars wide
    sys.stdout.write(f"\r[{bar:<50}] {percent:.1f}% ({step}/{total})")
    sys.stdout.flush()
    if step == total:
        print()  # move to next line when done


def combine_callbacks(*callbacks):
    """
    Returns a function that calls all given callbacks.
    """
    def combined(step, total):
        for cb in callbacks:
            if cb is not None:
                cb(step, total)
    return combined


def run_mcmc(time, flux, flux_err, period, ndim=4, nwalkers=32, nsteps=5000, progress_callback=None):

    loglike = partial(log_likelihood, time=time, flux=flux, flux_err=flux_err, period=period)
    best_fit = optimize_initial_guess(loglike)
    p0 = best_fit + 1e-3 * np.random.randn(nwalkers, ndim)

    # Use ~90% of your CPU cores
    num_cores = max(1, int(multiprocessing.cpu_count() * 0.5))

    with multiprocessing.Pool(num_cores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, pool=pool)

        combined_callback = combine_callbacks(progress_callback, terminal_progress_callback)
        for step, _ in enumerate(sampler.sample(p0, iterations=nsteps, progress=False)):
            if step % 10 == 0:
                combined_callback(step, nsteps)

        combined_callback(nsteps, nsteps)
        flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)

    results = compute_summary(flat_samples)

    return {
        'results': results,
        'sampler': sampler,
        'flat_samples': flat_samples,
        'labels': LABELS,
        'best_fit': best_fit
    }

