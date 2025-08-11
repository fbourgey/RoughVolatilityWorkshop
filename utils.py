import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from black import black_otm_impvol_mc


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


def var_swap_robust(ivol_data, slices=None):
    """Robust estimation of variance swap quotes."""
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
