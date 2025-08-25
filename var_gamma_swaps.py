"""
Robust variance- and gamma-swap estimators from implied-volatility strips.

This module provides var_swap_robust and gamma_swap_robust to compute
variance- and gamma-swap rates from market implied-volatility data.
Both functions handle bid/ask spreads, missing data and return mid/bid/ask
swap rates for each expiry.

Input: pandas.DataFrame with columns ['Bid', 'Ask', 'Texp', 'Strike', 'Fwd'].
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm


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
        z_minus = zp_in[ord_y_in][0]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zp_in[ord_y_in][-1]
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
