import numpy as np
import batman

from .constants import BOUNDS

# ===========================================================
# Constants
# ===========================================================

LIMB_DARKENING_COEFFS = [0.1, 0.3]
LIMB_DARKENING_MODEL = "quadratic"
ECCENTRICITY = 0.0
ARG_PERI = 90.0  # argument of periastron


# ===========================================================
# Helpers
# ===========================================================

def check_bounds(params):
    """
    Check if params are within physical and prior bounds.
    """
    for p, (low, high) in zip(params, BOUNDS):
        if not (low <= p <= high):
            return False

    rp_rs, inc_deg, a_rs, _ = params
    b = a_rs * np.cos(np.radians(inc_deg))
    if b > 1 + rp_rs:
        return False

    return True


# ===========================================================
# Transit Model
# ===========================================================

def make_transit_model(t, params, period):
    """
    Compute transit model at times `t` given parameters:
        params: [rp/rs, inclination (deg), a/rs, t0]
        period: orbital period [days]
    Returns flux array.
    """
    rp_rs, inc_deg, a_rs, t0 = params

    transit_params = batman.TransitParams()
    transit_params.t0 = t0
    transit_params.per = period
    transit_params.rp = rp_rs
    transit_params.a = a_rs
    transit_params.inc = inc_deg
    transit_params.ecc = ECCENTRICITY
    transit_params.w = ARG_PERI
    transit_params.u = LIMB_DARKENING_COEFFS
    transit_params.limb_dark = LIMB_DARKENING_MODEL

    model = batman.TransitModel(transit_params, t)
    return model.light_curve(transit_params)


# ===========================================================
# Log-Likelihood
# ===========================================================

def log_likelihood(params, time, flux, flux_err, period):
    """
    Compute log-likelihood of transit model given observed data.
    """
    if not check_bounds(params):
        return -np.inf

    try:
        model_flux = make_transit_model(time, params, period)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(model_flux)):
        return -np.inf

    chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)

    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2
