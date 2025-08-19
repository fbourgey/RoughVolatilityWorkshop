import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
import pandas as pd

from scipy import optimize, stats
import seaborn as sns

# Module-level constants for magic numbers
IMPVOL_MIN = 1e-10
IMPVOL_MAX = 5.0


def m_q_moment_diff_logvol(sig, q, lag):
    """
    Compute the mean of the q-th power of absolute log-vol differences at given lag.

    For a given lag delta, computes E[|log(sig_t) - log(sig_{t-delta})|^q].

    Parameters
    ----------
    sig : np.ndarray
        Time series of volatility data as a 1-D numpy array
    q : float
        Power for the moment calculation
    lag : int or array-like
        Time lag(s) delta at which to compute the differences. Can be a single integer
        or an array of integers.

    Returns
    -------
    np.ndarray
        Array of q-th power mean absolute differences for each lag.
        If lag is a single integer, returns an array with one element.
        If lag is array-like, returns an array with the same length as lag.
    """
    lag_vec = np.atleast_1d(np.asarray(lag, dtype=int))
    log_sig = np.log(sig)
    return np.array(
        [np.mean(np.abs(log_sig[lag:] - log_sig[:-lag]) ** q) for lag in lag_vec]
    )


def linreg(x, y, intercept=True):
    """
    Perform linear regression on two variables: y = alpha + beta * x

    Parameters
    ----------
    x : np.ndarray
        The independent variable (predictor). Should be a 1-dimensional numpy array.
    y : np.ndarray
        The dependent variable (response). Should be a 1-dimensional numpy array of the
        same length as x.
    intercept : bool, optional
        If True, fits a model with an intercept (alpha). If False, forces the regression
        line through the origin. Default is True.

    Returns
    -------
    If intercept=True:
        Tuple[float, float]
            A tuple containing (alpha, beta) where:
            - alpha: float, the intercept of the regression line
            - beta: float, the slope of the regression line
    If intercept=False:
        float
            beta: the slope of the regression line (forced through origin)
    """
    if intercept:
        c = np.cov(x, y)
        beta = c[0, 1] / c[0, 0]
        alpha = np.mean(y) - beta * np.mean(x)
        return alpha, beta
    else:
        beta = np.mean(x * y) / np.mean(x**2)
        return beta


def plot_ivols_mc(ivol_data, slices=None, mc_matrix=None, plot=True, colnum=None):
    """Plot implied vols."""

    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.unique(ivol_data["Texp"])
    n_slices = len(exp_dates)

    if slices is not None:
        n_slices = len(slices)
    else:
        slices = range(n_slices)

    if colnum is None:
        colnum = int(np.sqrt(n_slices * 2))

    rows = int(np.round(colnum / 2))
    columns = int(np.round(colnum))

    while rows * columns < n_slices:
        rows += 1

    atm_vols = np.zeros(n_slices)
    atm_skew = np.zeros(n_slices)
    atm_curv = np.zeros(n_slices)
    atm_vols_mc = np.zeros(n_slices)
    atm_skew_mc = np.zeros(n_slices)
    atm_curv_mc = np.zeros(n_slices)
    atm_err_mc = np.zeros(n_slices)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 10))
    axes = axes.flatten()

    for i, slice in enumerate(slices):
        t = exp_dates[slice]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t]
        ask_vol = ask_vols[texp == t]
        mid_vol = (bid_vol + ask_vol) / 2
        f = ivol_data["Fwd"][texp == t].iloc[0]
        k = np.log(ivol_data["Strike"][texp == t] / f)
        include = ~np.isnan(bid_vol) & (bid_vol > 0)
        kmin = 0.9 * np.min(k[include])
        kmax = 1.1 * np.max(k[include])
        ybottom = 0.6 * np.min(bid_vol[include])
        ytop = 1.2 * np.max(ask_vol[include])
        xrange = [kmin, kmax]
        yrange = [ybottom, ytop]

        if plot:
            ax = axes[i]
            ax.plot(k, bid_vol, "ro", markersize=3, label="Bid Vol")
            ax.plot(k, ask_vol, "bo", markersize=3, label="Ask Vol")
            ax.axvline(0, linestyle="--", color="gray")
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.set_title(f"T = {t:.5f}")
            ax.set_xlabel("Log-Strike")
            ax.set_ylabel("Implied Vol.")

        if mc_matrix is not None:

            def vol_mc(k):
                return black_otm_impvol_mc(S=mc_matrix[slice, :], k=k, T=t)

            atm_vols_mc[slice] = vol_mc(0.0)
            sig_mc = atm_vols_mc[slice] * np.sqrt(t)
            sig_mc_right = vol_mc(sig_mc / 10.0)
            sig_mc_left = vol_mc(-sig_mc / 10.0)
            atm_skew_mc[slice] = (sig_mc_right - sig_mc_left) / (2.0 * sig_mc / 10.0)
            atm_curv_mc[slice] = (
                sig_mc_right + sig_mc_left - 2.0 * atm_vols_mc[slice]
            ) / (2.0 * (sig_mc / 10.0) ** 2.0)

            def err_mc(k):
                return black_otm_impvol_mc(
                    S=mc_matrix[slice, :], k=k, T=t, mc_error=True
                )["error_95"]

            atm_err_mc[slice] = err_mc(0)
            if plot:
                k_vals = np.linspace(kmin, kmax, 100)
                vol_mc_vals = np.array([vol_mc(k) for k in k_vals])
                ax.plot(k_vals, vol_mc_vals, "orange", linewidth=2, label="MC")

        k_in = k[~np.isnan(mid_vol)]
        vol_in = mid_vol[~np.isnan(mid_vol)]
        vol_interp = PchipInterpolator(k_in, vol_in)
        atm_vols[slice] = vol_interp(0)
        sig = atm_vols[slice] * np.sqrt(t)
        atm_skew[slice] = (vol_interp(sig / 10) - vol_interp(-sig / 10)) / (
            2 * sig / 10
        )
        atm_curv[slice] = (
            vol_interp(sig / 10) + vol_interp(-sig / 10) - 2 * atm_vols[slice]
        ) / (2 * (sig / 10) ** 2)

    if plot:
        for i in range(n_slices, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.show()

    return {
        "expiries": exp_dates,
        "atm_vols": atm_vols,
        "atm_skew": atm_skew,
        "atm_curv": atm_curv,
        "atm_vols_mc": atm_vols_mc,
        "atm_skew_mc": atm_skew_mc,
        "atm_err_mc": atm_err_mc,
    }


def var_swap_robust(ivol_data: pd.DataFrame, slices=None):
    """
    Robust estimation of variance swap quotes using implied volatility data.

    This function calculates variance swap rates from a strip of option prices using
    a robust estimation method. It handles bid-ask spreads and computes mid, bid, and
    ask variance swap rates for different time slices.

    Parameters
    ----------
    ivol_data : pandas.DataFrame
        DataFrame containing implied volatility data with columns:
        - 'Bid' : float
            Bid implied volatilities
        - 'Ask' : float
            Ask implied volatilities
        - 'Texp' : float
            Time to expiry for each option
        - 'Strike' : float
            Strike prices
        - 'Fwd' : float
            Forward prices
    slices : array-like, optional
        Specific time slices to compute variance swaps for. If None, uses all
        available expiry dates. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'expiries' : ndarray
            Array of expiry dates
        - 'vs_mid' : ndarray
            Array of mid variance swap rates
        - 'vs_bid' : ndarray
            Array of bid variance swap rates
        - 'vs_ask' : ndarray
            Array of ask variance swap rates
    """
    ivol_data = ivol_data.dropna()
    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.sort(np.unique(ivol_data["Texp"]))
    n_slices = len(exp_dates)

    if slices is not None:
        n_slices = len(slices)
    else:
        slices = range(n_slices)

    vs_mid = np.zeros(n_slices)
    vs_bid = np.zeros(n_slices)
    vs_ask = np.zeros(n_slices)

    def varswap(k_in, vol_series, slice_idx):
        t = exp_dates[slice_idx]
        sig_in = vol_series * np.sqrt(t)
        zm_in = -k_in / sig_in - sig_in / 2
        y_in = norm.cdf(zm_in)
        ord_y_in = np.argsort(y_in)
        sig_in_y = sig_in[ord_y_in]
        y_min = np.min(y_in)
        y_max = np.max(y_in)
        sig_in_0 = sig_in_y[0]
        sig_in_1 = sig_in_y[-1]

        wbar_flat = quad(PchipInterpolator(np.sort(y_in), sig_in_y**2), y_min, y_max)[0]
        res_mid = wbar_flat
        z_minus = zm_in[ord_y_in][0]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zm_in[ord_y_in][-1]
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_vs = res_mid + res_lh + res_rh
        return res_vs

    for slice_idx in slices:
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t].to_numpy()
        ask_vol = ask_vols[texp == t].to_numpy()
        mid_vol = (bid_vol + ask_vol) / 2
        F = ivol_data["Fwd"][texp == t].iloc[0]  # forward price
        k = np.log(ivol_data["Strike"][texp == t].to_numpy() / F)  # log-fwd moneyness
        vs_mid[slice_idx] = varswap(k, mid_vol, slice_idx) / t
        vs_bid[slice_idx] = varswap(k, bid_vol, slice_idx) / t
        vs_ask[slice_idx] = varswap(k, ask_vol, slice_idx) / t

    return {
        "expiries": exp_dates,
        "vs_mid": vs_mid,
        "vs_bid": vs_bid,
        "vs_ask": vs_ask,
    }


def gamma_swap_robust(ivol_data: pd.DataFrame, slices=None):
    """
    Robust estimation of gamma swap quotes.

    Parameters
    ----------
    ivol_data : pandas.DataFrame
        DataFrame containing implied volatility data with columns:
        - 'Bid' : float
            Bid implied volatilities
        - 'Ask' : float
            Ask implied volatilities
        - 'Texp' : float
            Time to expiry for each option
        - 'Strike' : float
            Strike prices
        - 'Fwd' : float
            Forward prices
    slices : array-like, optional
        Specific time slices to compute gamma swaps for. If None, uses all slices.

    Returns
    -------
    dict
        Dictionary containing:
        - expiries: array of expiry dates
        - gs_mid: array of mid gamma swap rates
        - gs_bid: array of bid gamma swap rates
        - gs_ask: array of ask gamma swap rates
    """
    ivol_data = ivol_data.dropna()
    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.sort(np.unique(ivol_data["Texp"]))
    n_slices = len(exp_dates)

    if slices is not None:
        n_slices = len(slices)
    else:
        slices = range(n_slices)

    gs_mid = np.zeros(n_slices)
    gs_bid = np.zeros(n_slices)
    gs_ask = np.zeros(n_slices)

    def gammaswap(k_in, vol_series, slice_idx):
        t = exp_dates[slice_idx]
        sig_in = vol_series * np.sqrt(t)
        zm_in = -k_in / sig_in - sig_in / 2
        zp_in = zm_in + sig_in  # Note: using zp for gamma swaps!
        y_in = norm.cdf(zp_in)
        ord_y_in = np.argsort(y_in)
        sig_in_y = sig_in[ord_y_in]
        y_min = np.min(y_in)
        y_max = np.max(y_in)
        sig_in_0 = sig_in_y[0]
        sig_in_1 = sig_in_y[-1]

        wbar_flat = quad(PchipInterpolator(np.sort(y_in), sig_in_y**2), y_min, y_max)[0]
        res_mid = wbar_flat
        z_minus = zp_in[ord_y_in][0]  # Note: using zp_in instead of zm_in
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zp_in[ord_y_in][-1]  # Note: using zp_in instead of zm_in
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_gs = res_mid + res_lh + res_rh
        return res_gs

    for slice_idx in slices:
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t].to_numpy()
        ask_vol = ask_vols[texp == t].to_numpy()
        mid_vol = (bid_vol + ask_vol) / 2
        F = ivol_data["Fwd"][texp == t].iloc[0]  # forward price
        k = np.log(ivol_data["Strike"][texp == t].to_numpy() / F)  # log-fwd moneyness
        gs_mid[slice_idx] = gammaswap(k, mid_vol, slice_idx) / t
        gs_bid[slice_idx] = gammaswap(k, bid_vol, slice_idx) / t
        gs_ask[slice_idx] = gammaswap(k, ask_vol, slice_idx) / t

    return {
        "expiries": exp_dates,
        "gs_mid": gs_mid,
        "gs_bid": gs_bid,
        "gs_ask": gs_ask,
    }


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    knots, weights = np.polynomial.legendre.leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Hermite quadrature points and weights.

    Integration is with respect to the Gaussian density. It corresponds to the
    probabilist's Hermite polynomials.

    Parameters
    ----------
    n: int
        Number of quadrature points.

    Returns
    -------
    knots: array-like
        Gauss-Hermite knots.
    weight: array-like
        Gauss-Hermite weights.
    """
    knots, weights = np.polynomial.hermite.hermgauss(n)
    knots *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    return knots, weights


def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a matrix using SVD and QR.

    This function works with positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Cholesky decomposition of the input matrix.
    """
    u, s, _ = np.linalg.svd(a)
    b = np.diag(np.sqrt(s)) @ u.T
    _, r = np.linalg.qr(b)
    return r.T


def black_price(K, T, F, vol, opttype: float | np.ndarray = 1.0):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : float or np.ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black price of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    price = opttype * (
        F * stats.norm.cdf(opttype * d1) - K * stats.norm.cdf(opttype * d2)
    )
    return price


def black_delta(K, T, F, vol, opttype=1):
    """
    Calculate the Black delta of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black delta of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return opttype * stats.norm.cdf(opttype * d1)


def black_gamma(K, T, F, vol):
    """
    Calculate the Black gamma of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black gamma of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return stats.norm.pdf(d1) / (F * s)


def black_speed(K, T, F, vol):
    """
    Calculate the Black speed of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black speed of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return -(d1 / s + 1.0) * stats.norm.pdf(d1) / (F**2 * s)


def black_vega(K, T, F, vol):
    """
    Calculate the Black vega of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black vega of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return F * stats.norm.pdf(d1) * np.sqrt(T)


@np.vectorize
def black_impvol_brentq(K, T, F, value, opttype=1):
    """
    Calculate the Black implied volatility using the Brent's method.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    value : float
        Observed market price of the option.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        Implied volatility corresponding to the input option price. Returns NaN
        if the implied volatility does not converge or if invalid inputs are provided.
    """
    if (K <= 0) or (T <= 0) or (F <= 0) or (value <= 0):
        return np.nan

    try:
        result = optimize.root_scalar(
            f=lambda vol: black_price(K, T, F, vol, opttype) - value,
            bracket=[IMPVOL_MIN, IMPVOL_MAX],
            method="brentq",
        )
        return result.root if result.converged else np.nan
    except ValueError:
        return np.nan


def black_impvol(
    K, T, F, value, opttype: int | np.ndarray = 1, TOL=1e-5, MAX_ITER=1000
):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-6.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
        If the implied volatility does not converge within `MAX_ITER` iterations.
    """
    K = np.atleast_1d(K)
    value = np.atleast_1d(value)
    opttype = np.full_like(K, opttype)

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    # Fix: check all opttype values
    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan)

    low = IMPVOL_MIN * np.ones_like(K)
    high = IMPVOL_MAX * np.ones_like(K)
    mid = 0.5 * (low + high)
    for _ in range(MAX_ITER):
        price = black_price(K, T, F, mid, opttype)
        diff = (price - value) / value

        if np.all(np.abs(diff) < TOL):
            return mid

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)

    # raise ValueError("Implied volatility did not converge.")
    print("Implied volatility did not converge for all log(K/F) values.")

    return mid


def black_otm_impvol_mc(
    S: np.ndarray, k: float | np.ndarray, T: float, mc_error: bool = False
) -> dict | np.ndarray:
    """
    Calculate Black implied volatility using Monte Carlo simulated stock prices and
    out-of-the-money (OTM) prices.

    Parameters
    ----------
    S : ndarray
        Array of Monte Carlo simulated stock prices.
    k : float or ndarray
        Log-Forward Moneyness `k=log(K/F)` for which the implied volatility is
        calculated.
    T : float
        Time to maturity of the option.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.

    Returns
    -------
    dict or ndarray
        If `mc_error` is False, returns an ndarray of OTM implied volatilities.
        If `mc_error` is True, returns a dictionary with the following keys:
        - 'otm_impvol': ndarray of OTM implied volatilities.
        - 'otm_impvol_high': ndarray of upper bounds of the 95% confidence interval.
        - 'otm_impvol_low': ndarray of lower bounds of the 95% confidence interval.
        - 'error_95': ndarray of the 95% confidence interval errors for the option
                      prices.
        - 'otm_price': ndarray of the calculated OTM option prices.
    """
    k = np.atleast_1d(k)
    F = np.mean(S)
    K = F * np.exp(k)
    # opttype: 1 for call, -1 for put, depending on moneyness
    opttype = 2 * (K >= F) - 1  # 1 if K >= F (call), -1 if K < F (put)
    payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0)
    otm_impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / S.shape[0] ** 0.5
        otm_impvol_high = black_impvol(
            K=K, T=T, F=F, value=otm_price + error_95, opttype=opttype
        )
        otm_impvol_low = black_impvol(
            K=K, T=T, F=F, value=otm_price - error_95, opttype=opttype
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol


def fourier(n: int, t: float) -> np.ndarray:
    """
    Compute the first n Fourier basis functions evaluated at point t.

    The basis consists of:
        - 1 (constant term)
        - sqrt(2) * sin(2 * pi * k * t), sqrt(2) * cos(2 * pi * k * t) for k = 1, 2, ...

    Parameters
    ----------
    n : int
        Number of Fourier basis functions to compute.
    t : float
        Point at which to evaluate the basis functions.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing the values of the first n Fourier basis
        functions at t.
    """
    tab = np.zeros(n)
    tab[0] = 1.0
    for i in range(1, n):
        tab[i] = (
            np.cos(2.0 * np.pi * i * t) if i % 2 == 0 else np.sin(2.0 * np.pi * i * t)
        )
        tab[i] *= np.sqrt(2.0)
    return tab


def set_plot_style() -> None:
    """Set default matplotlib parameters."""
    plt.rcParams["figure.figsize"] = [9.0, 7.0]
    sns.set_style(
        style="ticks",
        rc={
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    sns.set_context(
        context="poster",
        rc={
            "grid.linewidth": 1.0,
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "xx-small",
        },
    )


def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Perform simple linear regression (least squares) to fit y = alpha + beta * x.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of independent variable values.
    y : np.ndarray
        One-dimensional array of dependent variable values.

    Returns
    -------
    tuple[float, float]
        A tuple (alpha, beta) where:
        - alpha is the intercept of the regression line.
        - beta is the slope of the regression line.

    Raises
    ------
    ValueError
        If x or y are not one-dimensional arrays.
    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    cov = np.cov(x, y)
    cov_x_y = cov[0, 1]
    var_x = cov[0, 0]
    beta = cov_x_y / var_x
    alpha = y.mean() - beta * x.mean()
    return alpha, beta
