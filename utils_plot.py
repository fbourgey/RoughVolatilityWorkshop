import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from utils import black_otm_impvol_mc


def plot_ivols_mc(ivol_data, slices=None, mc_matrix=None, plot=True, colnum=None):
    """
    Plot implied volatility curves with bid/ask and Monte Carlo data.

    Parameters
    ----------
    ivol_data : DataFrame
        Market data with columns [Bid, Ask, Texp, Fwd, Strike]
    slices : array-like, optional
        Indices of expiry slices to plot
    mc_matrix : ndarray, optional
        Monte Carlo simulation results
    plot : bool, optional
        Whether to display plots, by default True
    colnum : int, optional
        Number of subplot columns

    Returns
    -------
    dict
        ATM vols, skews and curvatures for market and MC data
    """

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
