"""
Heston stochastic volatility model implementation.

Provides functions for simulating variance processes, computing characteristic
functions and implied volatilities using Andersen's QE scheme.

Reference: Heston (1993), Andersen (2008)
"""

import numpy as np

from utils import black_impvol, lewis_formula_otm_price


def psi_minus(psi, ev, z):
    """
    Compute algorithm psi- for QE scheme (Andersen's Quadratic Exponential scheme).

    Parameters
    ----------
    psi : float
        Ratio of variance to squared mean (var/mean^2)
    ev : float
        Expected value of the variance process
    z : ndarray
        Array of standard normal random variables

    Returns
    -------
    ndarray
        Simulated variance values following a quadratic distribution

    Notes
    -----
    Implementation of Andersen's QE scheme for psi <= 2.
    Uses the quadratic form V = alpha * (beta + Z)^2 where Z is standard normal.
    """
    beta2 = 2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(2 / psi - 1)
    beta = np.sqrt(beta2)
    alpha = ev / (1 + beta2)
    return alpha * (beta + z) ** 2


def psi_plus(psi, ev, u):
    """
    Compute algorithm psi+ for QE scheme (Andersen's Quadratic Exponential scheme).

    Parameters
    ----------
    psi : float
        Ratio of variance to squared mean (var/mean^2)
    ev : float
        Expected value of the variance process
    u : ndarray
        Array of uniform random variables on [0,1]

    Returns
    -------
    ndarray
        Simulated variance values following an exponential distribution

    Notes
    -----
    This implements Algorithm psi+ from Andersen's QE scheme for
    simulating the variance process when psi >= 1.5
    """
    p = 2 / (1 + psi)
    gamma = ev / 2 * (1 + psi)

    # Compute exponential variates using inverse transform
    vf = -1.0 * (u < p) * gamma * np.log(u / p)
    return vf


def psi(v, params, dt):
    """
    Compute psi ratio for QE variance simulation scheme.

    Calculates ratio to determine which QE scheme branch to use for variance paths.

    Parameters
    ----------
    v : float
        Current variance value
    params : dict
        Model parameters (nu, lbd, vbar)
    dt : float
        Time step size

    Returns
    -------
    float
        Ratio of variance to squared mean
    """
    nu = params["nu"]
    lbd = params["lbd"]
    vbar = params["vbar"]

    eldt = np.exp(-lbd * dt)

    ev = (v - vbar) * eldt + vbar
    varv = (nu**2 / lbd) * (eldt * (1 - eldt) * (v - vbar) + vbar / 2 * (1 - eldt**2))

    return varv / (ev**2)


def phi_heston(u, tau, params):
    """
    Compute the characteristic function of the Heston model E[exp(i * u * X_tau)]
    where X_tau = log(S_tau/S_0) at u for time tau.

    Parameters
    ----------
    u : float or array_like
        Points at which to evaluate the characteristic function
    tau : float
        Time to maturity
    params : dict
        Dictionary containing model parameters:
        - lbd: mean reversion rate
        - rho: correlation between asset and variance
        - nu: volatility of variance
        - vbar: long-term variance
        - v: initial variance

    Returns
    -------
    complex
        Value of the characteristic function
    """
    lbd = params["lbd"]
    rho = params["rho"]
    nu = params["nu"]
    vbar = params["vbar"]
    v = params["v"]

    al = -u * u / 2 - 1j * u / 2
    bet = lbd - rho * nu * 1j * u
    gam = nu**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    D = rm * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    C = lbd * (rm * tau - 2 / nu**2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    return np.exp(C * vbar + D * v)


def impvol_heston_charfunc(k, tau, params):
    """
    Calculate implied volatility in the Heston model using the characteristic function.

    Parameters
    ----------
    k : array_like
        Log strike k = log(K/F)
    tau : float
        Time to maturity
    params : dict
        Model parameters containing lbd, rho, nu, vbar, v

    Returns
    -------
    ndarray
        Black implied volatility values
    """
    k = np.atleast_1d(np.asarray(k))
    otm_price = lewis_formula_otm_price(
        lambda u, tau: phi_heston(u=u, tau=tau, params=params),
        k=k,
        tau=tau,
    )
    opttype = 2 * (k > 0) - 1  # otm options
    impvol = black_impvol(K=np.exp(k), T=tau, F=1, value=otm_price, opttype=opttype)
    return impvol
