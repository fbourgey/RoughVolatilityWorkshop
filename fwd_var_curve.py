"""
Forward Variance Curve Computation Module.

This module provides functions to compute and optimize forward variance curves
from a set of input variance values and expiry times. It includes both piecewise
constant and smooth (cubic spline) implementations.

Functions:
    obj_w: Creates objective function for variance curve optimization
    xi_curve: Computes piecewise constant forward variance curve
    xi_curve_smooth: Computes smooth forward variance curve using cubic splines
"""

import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize


def _objective_w(expiries, w_in):
    """
    Create an objective function for optimizing forward variance curve.

    Parameters
    ----------
    expiries : ndarray
        Array of expiry times.
    w_in : ndarray
        Array of input variance values.

    Returns
    -------
    callable
        Objective function that takes error vector as input and returns
        optimization score.
    """

    def objective(err_vec):
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        xi_vec = np.concatenate(
            ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
        )
        dxi_dt = np.diff(xi_vec) / np.diff(expiries)
        w_out = (
            np.concatenate(([0], np.cumsum(xi_vec[1:] * np.diff(expiries))))
            + xi_vec[0] * expiries[0]
        )
        res = np.sum((w_in - w_out) ** 2) + np.sum(dxi_dt**2)
        return res * 1e3

    return objective


def xi_curve(expiries, w_in, eps=0):
    """
    Compute piecewise constant forward variance curve.

    Parameters
    ----------
    expiries : ndarray
        Array of expiry times.
    w_in : ndarray
        Array of input variance values.
    eps : float, optional
        Error bound for optimization, by default 0.

    Returns
    -------
    dict
        Dictionary containing:
        - xi_vec : ndarray
            Forward variance values at expiry points
        - xi_curve : callable
            Vectorized function for forward variance curve
        - fit_errs : ndarray
            Fitting errors at expiry points
        - w_out : ndarray
            Optimized variance values
    """
    n = len(w_in)
    if eps > 0:
        res_optim = minimize(
            _objective_w(expiries, w_in),
            np.zeros(n),
            method="L-BFGS-B",
            bounds=[(-eps, eps)] * n,
        )
        err_vec = res_optim.x
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    else:
        w_in_1 = w_in
    xi_vec_out = np.concatenate(
        ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
    )

    def xi_curve_raw(t):
        if t <= expiries[-1]:
            return xi_vec_out[np.sum(expiries < t)]
        else:
            return xi_vec_out[-1]

    xi_curve_out = np.vectorize(xi_curve_raw)
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {
        "xi_vec": xi_vec_out,
        "xi_curve": xi_curve_out,
        "fit_errs": fit_errs,
        "w_out": w_in_1,
    }


def xi_curve_smooth(expiries, w_in, xi=True, eps=0.0):
    """
    Compute smooth forward variance curve using cubic splines.

    Parameters
    ----------
    expiries : ndarray
        Array of expiry times.
    w_in : ndarray
        Array of input variance values.
    xi : bool, optional
        If True returns forward variance, if False returns variance, by default True.
    eps : float, optional
        Error bound for optimization, by default 0.0.

    Returns
    -------
    dict
        Dictionary containing:
        - xi_curve : callable
            Vectorized function for forward variance curve
        - fit_errs : ndarray
            Fitting errors at expiry points
        - w_out : ndarray
            Optimized variance values
    """

    def phi(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return 1 - min_val**3 / 6 + x * tau * (2 + min_val) / 2

        return func

    def phi_deri(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return tau - min_val**2 / 2 + tau * min_val

        return func

    n = len(expiries)
    A = np.array([[phi(expiries[i])(expiries[j]) for j in range(n)] for i in range(n)])
    A_inv = inv(A)

    def obj_1(err_vec):
        v = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        return v.T @ A_inv @ v

    res_optim = minimize(
        obj_1, np.zeros(n), method="L-BFGS-B", bounds=[(-eps, eps)] * n
    )
    err_vec = res_optim.x
    w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    Z = A_inv @ w_in_1

    def curve_raw(x):
        sum_curve = sum(Z[i] * phi(expiries[i])(x) for i in range(n))
        sum_curve_deri = sum(Z[i] * phi_deri(expiries[i])(x) for i in range(n))
        return sum_curve_deri if xi else sum_curve

    xi_curve_out = np.vectorize(curve_raw)
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {"xi_curve": xi_curve_out, "fit_errs": fit_errs, "w_out": w_in_1}
