"""
Padé approximations for rough Heston model auxiliary functions.

Implements [n,n] Padé schemes for h(a, tau), its derivative, and related functions.
"""

import numpy as np
import scipy.special as sp


def _h_pade_22(tau, a, params):
    """
    Calculates Padé [2,2] approximation for the rough Heston model characteristic
    function.

    Parameters
    ----------
    tau : array_like
        Time parameter
    a : float or complex
        Transform variable
    params : dict
        Model parameters with keys:
        - H : float
            Hurst parameter
        - rho : float
            Correlation coefficient
        - nu : float
            Volatility of volatility
        - lbd : float
            Mean reversion rate

    Returns
    -------
    complex
        Padé approximation value h(a,tau)
    """
    # Extract parameters
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    al = H + 1 / 2
    lbd = params["lbd"]

    # Calculate intermediates
    lbdp = lbd / nu
    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa

    # Calculate coefficients
    b1 = -a * (a + (0 + 1j)) / 2 * 1 / sp.gamma(1 + al)
    b2 = -b1 * lbd_tilde * nu * sp.gamma(1 + al) / sp.gamma(1 + 2 * al)

    g0 = rm / nu
    g1 = 0 if al == 1 else -1 / aa * 1 / sp.gamma(1 - al) * g0 / nu

    # Calculate denominator and q coefficients
    den = g0**2 + b1 * g1
    q1 = (b1 * g0 - b2 * g1) / den
    q2 = (b1**2 + b2 * g0) / den

    # Calculate p coefficients and final result
    p1 = b1
    p2 = b2 + b1 * q1
    y = tau**al

    return (p1 * y + p2 * y**2) / (1 + q1 * y + q2 * y**2)


def _h_pade_33(tau, a, params):
    """
    Calculates Padé [3,3] approximation for the rough Heston model characteristic
    function.

    Parameters
    ----------
    tau : array_like
        Time parameter
    a : float or complex
        Transform variable
    params : dict
        Model parameters with keys:
        - H : float
            Hurst parameter
        - rho : float
            Correlation coefficient
        - nu : float
            Volatility of volatility
        - lbd : float
            Mean reversion rate

    Returns
    -------
    complex
        Padé [3,3] approximation value h(a, tau)
    """
    tau = np.asarray(tau, dtype=np.float64)
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    al = H + 0.5
    lbd = params["lbd"]

    lbdp = lbd / nu
    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa

    b1 = -a * (a + (0 + 1j)) / 2 * 1 / sp.gamma(1 + al)
    b2 = -b1 * lbd_tilde * nu * sp.gamma(1 + al) / sp.gamma(1 + 2 * al)
    b3 = (
        (-b2 * lbd_tilde * nu + nu**2 * b1**2 / 2)
        * sp.gamma(1 + 2 * al)
        / sp.gamma(1 + 3 * al)
    )

    g0 = rm / nu
    g1 = -1 / (aa * nu) * 1 / sp.gamma(1 - al) * g0
    g2 = (
        -1
        / (aa * nu)
        * (sp.gamma(1 - al) / sp.gamma(1 - 2 * al) * g1 - 0.5 * nu**2 * g1 * g1)
    )

    den = g0**3 + 2 * b1 * g0 * g1 - b2 * g1**2 + b1**2 * g2 + b2 * g0 * g2

    p1 = b1
    p2 = (
        b1**2 * g0**2
        + b2 * g0**3
        + b1**3 * g1
        + b1 * b2 * g0 * g1
        - b2**2 * g1**2
        + b1 * b3 * g1**2
        + b2**2 * g0 * g2
        - b1 * b3 * g0 * g2
    ) / den
    q1 = (
        b1 * g0**2
        + b1**2 * g1
        - b2 * g0 * g1
        + b3 * g1**2
        - b1 * b2 * g2
        - b3 * g0 * g2
    ) / den
    q2 = (
        b1**2 * g0
        + b2 * g0**2
        - b1 * b2 * g1
        - b3 * g0 * g1
        + b2**2 * g2
        - b1 * b3 * g2
    ) / den
    q3 = (b1**3 + 2 * b1 * b2 * g0 + b3 * g0**2 - b2**2 * g1 + b1 * b3 * g1) / den
    p3 = g0 * q3

    y = tau**al

    return (p1 * y + p2 * y**2 + p3 * y**3) / (1 + q1 * y + q2 * y**2 + q3 * y**3)


def _h_pade_44(tau, a, params):
    """
    Calculates Padé [4,4] approximation for the rough Heston model characteristic
    function.

    Parameters
    ----------
    tau : array_like
        Time parameter
    a : float or complex
        Transform variable
    params : dict
        Model parameters with keys:
        - H : float
            Hurst parameter
        - rho : float
            Correlation coefficient
        - nu : float
            Volatility of volatility
        - lbd : float
            Mean reversion rate

    Returns
    -------
    complex
        Padé [4,4] approximation value h(a, tau)
    """
    tau = np.asarray(tau, dtype=np.float64)
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    al = H + 0.5
    lbd = params["lbd"]

    lbdp = lbd / nu
    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa

    b1 = -a * (a + (0 + 1j)) / 2 * 1 / sp.gamma(1 + al)
    b2 = -b1 * lbd_tilde * nu * sp.gamma(1 + al) / sp.gamma(1 + 2 * al)
    b3 = (
        (-b2 * lbd_tilde * nu + nu**2 * b1**2 / 2)
        * sp.gamma(1 + 2 * al)
        / sp.gamma(1 + 3 * al)
    )
    b4 = (
        (-b3 * lbd_tilde * nu + nu**2 * b1 * b2)
        * sp.gamma(1 + 3 * al)
        / sp.gamma(1 + 4 * al)
    )

    g0 = rm / nu
    g1 = 0 if al == 1 else -1 / (aa * nu) * 1 / sp.gamma(1 - al) * g0
    g2 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - al) / sp.gamma(1 - 2 * al) * g1 - 0.5 * nu**2 * g1 * g1)
    )
    g3 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - 2 * al) / sp.gamma(1 - 3 * al) * g2 - nu**2 * g1 * g2)
    )

    den = (
        g0**4
        + 3 * b1 * g0**2 * g1
        + b1**2 * g1**2
        - 2 * b2 * g0 * g1**2
        + b3 * g1**3
        + 2 * b1**2 * g0 * g2
        + 2 * b2 * g0**2 * g2
        - 2 * b1 * b2 * g1 * g2
        - 2 * b3 * g0 * g1 * g2
        + b2**2 * g2**2
        - b1 * b3 * g2**2
        + b1**3 * g3
        + 2 * b1 * b2 * g0 * g3
        + b3 * g0**2 * g3
        - b2**2 * g1 * g3
        + b1 * b3 * g1 * g3
    )

    q1 = (
        b1 * g0**3
        + 2 * b1**2 * g0 * g1
        - b2 * g0**2 * g1
        - 2 * b1 * b2 * g1**2
        + b3 * g0 * g1**2
        - b4 * g1**3
        + b1**3 * g2
        - b3 * g0**2 * g2
        + b2**2 * g1 * g2
        + b1 * b3 * g1 * g2
        + 2 * b4 * g0 * g1 * g2
        - b2 * b3 * g2**2
        + b1 * b4 * g2**2
        - b1**2 * b2 * g3
        - b2**2 * g0 * g3
        - b1 * b3 * g0 * g3
        - b4 * g0**2 * g3
        + b2 * b3 * g1 * g3
        - b1 * b4 * g1 * g3
    ) / den

    q2 = (
        b1**2 * g0**2
        + b2 * g0**3
        + b1**3 * g1
        - b3 * g0**2 * g1
        + b1 * b3 * g1**2
        + b4 * g0 * g1**2
        - b1**2 * b2 * g2
        + b2**2 * g0 * g2
        - 3 * b1 * b3 * g0 * g2
        - b4 * g0**2 * g2
        - b2 * b3 * g1 * g2
        + b1 * b4 * g1 * g2
        + b3**2 * g2**2
        - b2 * b4 * g2**2
        + b1 * b2**2 * g3
        - b1**2 * b3 * g3
        + b2 * b3 * g0 * g3
        - b1 * b4 * g0 * g3
        - b3**2 * g1 * g3
        + b2 * b4 * g1 * g3
    ) / den

    q3 = (
        b1**3 * g0
        + 2 * b1 * b2 * g0**2
        + b3 * g0**3
        - b1**2 * b2 * g1
        - 2 * b2**2 * g0 * g1
        - b4 * g0**2 * g1
        + b2 * b3 * g1**2
        - b1 * b4 * g1**2
        + b1 * b2**2 * g2
        - b1**2 * b3 * g2
        + b2 * b3 * g0 * g2
        - b1 * b4 * g0 * g2
        - b3**2 * g1 * g2
        + b2 * b4 * g1 * g2
        - b2**3 * g3
        + 2 * b1 * b2 * b3 * g3
        - b1**2 * b4 * g3
        + b3**2 * g0 * g3
        - b2 * b4 * g0 * g3
    ) / den

    q4 = (
        b1**4
        + 3 * b1**2 * b2 * g0
        + b2**2 * g0**2
        + 2 * b1 * b3 * g0**2
        + b4 * g0**3
        - 2 * b1 * b2**2 * g1
        + 2 * b1**2 * b3 * g1
        - 2 * b2 * b3 * g0 * g1
        + 2 * b1 * b4 * g0 * g1
        + b3**2 * g1**2
        - b2 * b4 * g1**2
        + b2**3 * g2
        - 2 * b1 * b2 * b3 * g2
        + b1**2 * b4 * g2
        - b3**2 * g0 * g2
        + b2 * b4 * g0 * g2
    ) / den

    p1 = b1
    p2 = b2 + b1 * q1
    p3 = b3 + b1 * q2 + b2 * q1
    p4 = g0 * q4

    y = tau**al

    return (p1 * y + p2 * y**2 + p3 * y**3 + p4 * y**4) / (
        1 + q1 * y + q2 * y**2 + q3 * y**3 + q4 * y**4
    )


def _h_pade_55(tau, a, params):
    """
    Calculates Padé [5,5] approximation for the rough Heston model characteristic
    function.

    Parameters
    ----------
    tau : array_like
        Time parameter
    a : float or complex
        Transform variable
    params : dict
        Model parameters with keys:
        - H : float
            Hurst parameter
        - rho : float
            Correlation coefficient
        - nu : float
            Volatility of volatility
        - lbd : float
            Mean reversion rate

    Returns
    -------
    complex
        Padé [5,5] approximation value h(a, tau)
    """
    tau = np.asarray(tau, dtype=np.float64)
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    al = H + 0.5
    lbd = params["lbd"]

    lbdp = lbd / nu
    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa

    b1 = -a * (a + (0 + 1j)) / 2 * 1 / sp.gamma(1 + al)
    b2 = -b1 * lbd_tilde * nu * sp.gamma(1 + al) / sp.gamma(1 + 2 * al)
    b3 = (
        (-b2 * lbd_tilde * nu + nu**2 * b1**2 / 2)
        * sp.gamma(1 + 2 * al)
        / sp.gamma(1 + 3 * al)
    )
    b4 = (
        (-b3 * lbd_tilde * nu + nu**2 * b1 * b2)
        * sp.gamma(1 + 3 * al)
        / sp.gamma(1 + 4 * al)
    )
    b5 = (
        (-b4 * lbd_tilde * nu + nu**2 * (1 / 2 * b2 * b2 + b1 * b3))
        * sp.gamma(1 + 4 * al)
        / sp.gamma(1 + 5 * al)
    )

    g0 = rm / nu
    g1 = 0 if al == 1 else -1 / (aa * nu) * 1 / sp.gamma(1 - al) * g0
    g2 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - al) / sp.gamma(1 - 2 * al) * g1 - 1 / 2 * nu**2 * g1 * g1)
    )
    g3 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - 2 * al) / sp.gamma(1 - 3 * al) * g2 - nu**2 * g1 * g2)
    )
    g4 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (
            sp.gamma(1 - 3 * al) / sp.gamma(1 - 4 * al) * g3
            - nu**2 * (1 / 2 * g2 * g2 + g1 * g3)
        )
    )

    den = (
        -(g0**5)
        - 4 * b1 * g0**3 * g1
        - 3 * b1**2 * g0 * g1**2
        + 3 * b2 * g0**2 * g1**2
        + 2 * b1 * b2 * g1**3
        - 2 * b3 * g0 * g1**3
        + b4 * g1**4
        - 3 * b1**2 * g0**2 * g2
        - 3 * b2 * g0**3 * g2
        - 2 * b1**3 * g1 * g2
        + 2 * b1 * b2 * g0 * g1 * g2
        + 4 * b3 * g0**2 * g1 * g2
        - b2**2 * g1**2 * g2
        - 2 * b1 * b3 * g1**2 * g2
        - 3 * b4 * g0 * g1**2 * g2
        + b1**2 * b2 * g2**2
        - 2 * b2**2 * g0 * g2**2
        + 4 * b1 * b3 * g0 * g2**2
        + b4 * g0**2 * g2**2
        + 2 * b2 * b3 * g1 * g2**2
        - 2 * b1 * b4 * g1 * g2**2
        - b3**2 * g2**3
        + b2 * b4 * g2**3
        - 2 * b1**3 * g0 * g3
        - 4 * b1 * b2 * g0**2 * g3
        - 2 * b3 * g0**3 * g3
        + 2 * b1**2 * b2 * g1 * g3
        + 4 * b2**2 * g0 * g1 * g3
        + 2 * b4 * g0**2 * g1 * g3
        - 2 * b2 * b3 * g1**2 * g3
        + 2 * b1 * b4 * g1**2 * g3
        - 2 * b1 * b2**2 * g2 * g3
        + 2 * b1**2 * b3 * g2 * g3
        - 2 * b2 * b3 * g0 * g2 * g3
        + 2 * b1 * b4 * g0 * g2 * g3
        + 2 * b3**2 * g1 * g2 * g3
        - 2 * b2 * b4 * g1 * g2 * g3
        + b2**3 * g3**2
        - 2 * b1 * b2 * b3 * g3**2
        + b1**2 * b4 * g3**2
        - b3**2 * g0 * g3**2
        + b2 * b4 * g0 * g3**2
        - b1**4 * g4
        - 3 * b1**2 * b2 * g0 * g4
        - b2**2 * g0**2 * g4
        - 2 * b1 * b3 * g0**2 * g4
        - b4 * g0**3 * g4
        + 2 * b1 * b2**2 * g1 * g4
        - 2 * b1**2 * b3 * g1 * g4
        + 2 * b2 * b3 * g0 * g1 * g4
        - 2 * b1 * b4 * g0 * g1 * g4
        - b3**2 * g1**2 * g4
        + b2 * b4 * g1**2 * g4
        - b2**3 * g2 * g4
        + 2 * b1 * b2 * b3 * g2 * g4
        - b1**2 * b4 * g2 * g4
        + b3**2 * g0 * g2 * g4
        - b2 * b4 * g0 * g2 * g4
    )

    # Calculate q coefficients
    q1 = (
        -(b1 * g0**4)
        - 3 * b1**2 * g0**2 * g1
        + b2 * g0**3 * g1
        - b1**3 * g1**2
        + 4 * b1 * b2 * g0 * g1**2
        - b3 * g0**2 * g1**2
        - b2**2 * g1**3
        - 2 * b1 * b3 * g1**3
        + b4 * g0 * g1**3
        - b5 * g1**4
        - 2 * b1**3 * g0 * g2
        - b1 * b2 * g0**2 * g2
        + b3 * g0**3 * g2
        + 4 * b1**2 * b2 * g1 * g2
        + 2 * b1 * b3 * g0 * g1 * g2
        - 2 * b4 * g0**2 * g1 * g2
        + 2 * b2 * b3 * g1**2 * g2
        + b1 * b4 * g1**2 * g2
        + 3 * b5 * g0 * g1**2 * g2
        - 2 * b1 * b2**2 * g2**2
        + b1**2 * b3 * g2**2
        - 2 * b1 * b4 * g0 * g2**2
        - b5 * g0**2 * g2**2
        - b3**2 * g1 * g2**2
        - b2 * b4 * g1 * g2**2
        + 2 * b1 * b5 * g1 * g2**2
        + b3 * b4 * g2**3
        - b2 * b5 * g2**3
        - b1**4 * g3
        - b1**2 * b2 * g0 * g3
        + b2**2 * g0**2 * g3
        + b4 * g0**3 * g3
        - 2 * b1**2 * b3 * g1 * g3
        - 4 * b2 * b3 * g0 * g1 * g3
        - 2 * b5 * g0**2 * g1 * g3
        + b3**2 * g1**2 * g3
        + b2 * b4 * g1**2 * g3
        - 2 * b1 * b5 * g1**2 * g3
        + b2**3 * g2 * g3
        - b1**2 * b4 * g2 * g3
        + b3**2 * g0 * g2 * g3
        + b2 * b4 * g0 * g2 * g3
        - 2 * b1 * b5 * g0 * g2 * g3
        - 2 * b3 * b4 * g1 * g2 * g3
        + 2 * b2 * b5 * g1 * g2 * g3
        - b2**2 * b3 * g3**2
        + b1 * b3**2 * g3**2
        + b1 * b2 * b4 * g3**2
        - b1**2 * b5 * g3**2
        + b3 * b4 * g0 * g3**2
        - b2 * b5 * g0 * g3**2
        + b1**3 * b2 * g4
        + 2 * b1 * b2**2 * g0 * g4
        + b1**2 * b3 * g0 * g4
        + 2 * b2 * b3 * g0**2 * g4
        + b1 * b4 * g0**2 * g4
        + b5 * g0**3 * g4
        - b2**3 * g1 * g4
        + b1**2 * b4 * g1 * g4
        - b3**2 * g0 * g1 * g4
        - b2 * b4 * g0 * g1 * g4
        + 2 * b1 * b5 * g0 * g1 * g4
        + b3 * b4 * g1**2 * g4
        - b2 * b5 * g1**2 * g4
        + b2**2 * b3 * g2 * g4
        - b1 * b3**2 * g2 * g4
        - b1 * b2 * b4 * g2 * g4
        + b1**2 * b5 * g2 * g4
        - b3 * b4 * g0 * g2 * g4
        + b2 * b5 * g0 * g2 * g4
    ) / den

    q2 = (
        -(b1**2 * g0**3)
        - b2 * g0**4
        - 2 * b1**3 * g0 * g1
        - b1 * b2 * g0**2 * g1
        + b3 * g0**3 * g1
        + 2 * b1**2 * b2 * g1**2
        + b2**2 * g0 * g1**2
        - b4 * g0**2 * g1**2
        + b1 * b4 * g1**3
        + b5 * g0 * g1**3
        - b1**4 * g2
        - b1**2 * b2 * g0 * g2
        - 2 * b2**2 * g0**2 * g2
        + 3 * b1 * b3 * g0**2 * g2
        + b4 * g0**3 * g2
        - 2 * b1 * b2**2 * g1 * g2
        - 4 * b1 * b4 * g0 * g1 * g2
        - 2 * b5 * g0**2 * g1 * g2
        - b2 * b4 * g1**2 * g2
        + b1 * b5 * g1**2 * g2
        + 2 * b1 * b2 * b3 * g2**2
        - 2 * b1**2 * b4 * g2**2
        - b3**2 * g0 * g2**2
        + 3 * b2 * b4 * g0 * g2**2
        - 2 * b1 * b5 * g0 * g2**2
        + b3 * b4 * g1 * g2**2
        - b2 * b5 * g1 * g2**2
        - b4**2 * g2**3
        + b3 * b5 * g2**3
        + b1**3 * b2 * g3
        + 3 * b1**2 * b3 * g0 * g3
        + 3 * b1 * b4 * g0**2 * g3
        + b5 * g0**3 * g3
        + b2**3 * g1 * g3
        - 2 * b1 * b2 * b3 * g1 * g3
        + b1**2 * b4 * g1 * g3
        + b3**2 * g0 * g1 * g3
        - b2 * b4 * g0 * g1 * g3
        - b3 * b4 * g1**2 * g3
        + b2 * b5 * g1**2 * g3
        - b2**2 * b3 * g2 * g3
        - b1 * b3**2 * g2 * g3
        + 3 * b1 * b2 * b4 * g2 * g3
        - b1**2 * b5 * g2 * g3
        - b3 * b4 * g0 * g2 * g3
        + b2 * b5 * g0 * g2 * g3
        + 2 * b4**2 * g1 * g2 * g3
        - 2 * b3 * b5 * g1 * g2 * g3
        + b2 * b3**2 * g3**2
        - b2**2 * b4 * g3**2
        - b1 * b3 * b4 * g3**2
        + b1 * b2 * b5 * g3**2
        - b4**2 * g0 * g3**2
        + b3 * b5 * g0 * g3**2
        - b1**2 * b2**2 * g4
        + b1**3 * b3 * g4
        - b2**3 * g0 * g4
        + b1**2 * b4 * g0 * g4
        - b2 * b4 * g0**2 * g4
        + b1 * b5 * g0**2 * g4
        + b2**2 * b3 * g1 * g4
        + b1 * b3**2 * g1 * g4
        - 3 * b1 * b2 * b4 * g1 * g4
        + b1**2 * b5 * g1 * g4
        + b3 * b4 * g0 * g1 * g4
        - b2 * b5 * g0 * g1 * g4
        - b4**2 * g1**2 * g4
        + b3 * b5 * g1**2 * g4
        - b2 * b3**2 * g2 * g4
        + b2**2 * b4 * g2 * g4
        + b1 * b3 * b4 * g2 * g4
        - b1 * b2 * b5 * g2 * g4
        + b4**2 * g0 * g2 * g4
        - b3 * b5 * g0 * g2 * g4
    ) / den

    q3 = (
        -(b1**3 * g0**2)
        - 2 * b1 * b2 * g0**3
        - b3 * g0**4
        - b1**4 * g1
        - b1**2 * b2 * g0 * g1
        + 2 * b2**2 * g0**2 * g1
        - b1 * b3 * g0**2 * g1
        + b4 * g0**3 * g1
        + b1 * b2**2 * g1**2
        - 2 * b1**2 * b3 * g1**2
        - 2 * b2 * b3 * g0 * g1**2
        - b5 * g0**2 * g1**2
        + b2 * b4 * g1**3
        - b1 * b5 * g1**3
        + b1**3 * b2 * g2
        + 3 * b1**2 * b3 * g0 * g2
        + 3 * b1 * b4 * g0**2 * g2
        + b5 * g0**3 * g2
        + 2 * b3**2 * g0 * g1 * g2
        - 2 * b2 * b4 * g0 * g1 * g2
        - b3 * b4 * g1**2 * g2
        + b2 * b5 * g1**2 * g2
        - b1 * b3**2 * g2**2
        + b1 * b2 * b4 * g2**2
        - b3 * b4 * g0 * g2**2
        + b2 * b5 * g0 * g2**2
        + b4**2 * g1 * g2**2
        - b3 * b5 * g1 * g2**2
        - b1**2 * b2**2 * g3
        + b1**3 * b3 * g3
        + b2**3 * g0 * g3
        - 4 * b1 * b2 * b3 * g0 * g3
        + 3 * b1**2 * b4 * g0 * g3
        - 2 * b3**2 * g0**2 * g3
        + b2 * b4 * g0**2 * g3
        + b1 * b5 * g0**2 * g3
        - b2**2 * b3 * g1 * g3
        + 3 * b1 * b3**2 * g1 * g3
        - b1 * b2 * b4 * g1 * g3
        - b1**2 * b5 * g1 * g3
        + 3 * b3 * b4 * g0 * g1 * g3
        - 3 * b2 * b5 * g0 * g1 * g3
        - b4**2 * g1**2 * g3
        + b3 * b5 * g1**2 * g3
        + b2 * b3**2 * g2 * g3
        - b2**2 * b4 * g2 * g3
        - b1 * b3 * b4 * g2 * g3
        + b1 * b2 * b5 * g2 * g3
        - b4**2 * g0 * g2 * g3
        + b3 * b5 * g0 * g2 * g3
        - b3**3 * g3**2
        + 2 * b2 * b3 * b4 * g3**2
        - b1 * b4**2 * g3**2
        - b2**2 * b5 * g3**2
        + b1 * b3 * b5 * g3**2
        + b1 * b2**3 * g4
        - 2 * b1**2 * b2 * b3 * g4
        + b1**3 * b4 * g4
        + b2**2 * b3 * g0 * g4
        - 2 * b1 * b3**2 * g0 * g4
        + b1**2 * b5 * g0 * g4
        - b3 * b4 * g0**2 * g4
        + b2 * b5 * g0**2 * g4
        - b2 * b3**2 * g1 * g4
        + b2**2 * b4 * g1 * g4
        + b1 * b3 * b4 * g1 * g4
        - b1 * b2 * b5 * g1 * g4
        + b4**2 * g0 * g1 * g4
        - b3 * b5 * g0 * g1 * g4
        + b3**3 * g2 * g4
        - 2 * b2 * b3 * b4 * g2 * g4
        + b1 * b4**2 * g2 * g4
        + b2**2 * b5 * g2 * g4
        - b1 * b3 * b5 * g2 * g4
    ) / den

    q4 = (
        -(b1**4 * g0)
        - 3 * b1**2 * b2 * g0**2
        - b2**2 * g0**3
        - 2 * b1 * b3 * g0**3
        - b4 * g0**4
        + b1**3 * b2 * g1
        + 4 * b1 * b2**2 * g0 * g1
        - b1**2 * b3 * g0 * g1
        + 4 * b2 * b3 * g0**2 * g1
        - b1 * b4 * g0**2 * g1
        + b5 * g0**3 * g1
        - b2**3 * g1**2
        + b1**2 * b4 * g1**2
        - 2 * b3**2 * g0 * g1**2
        + 2 * b1 * b5 * g0 * g1**2
        + b3 * b4 * g1**3
        - b2 * b5 * g1**3
        - b1**2 * b2**2 * g2
        + b1**3 * b3 * g2
        - 2 * b2**3 * g0 * g2
        + 2 * b1 * b2 * b3 * g0 * g2
        + b3**2 * g0**2 * g2
        - 2 * b2 * b4 * g0**2 * g2
        + b1 * b5 * g0**2 * g2
        + 2 * b2**2 * b3 * g1 * g2
        - 4 * b1 * b2 * b4 * g1 * g2
        + 2 * b1**2 * b5 * g1 * g2
        - b4**2 * g1**2 * g2
        + b3 * b5 * g1**2 * g2
        - b2 * b3**2 * g2**2
        + b2**2 * b4 * g2**2
        + b1 * b3 * b4 * g2**2
        - b1 * b2 * b5 * g2**2
        + b4**2 * g0 * g2**2
        - b3 * b5 * g0 * g2**2
        + b1 * b2**3 * g3
        - 2 * b1**2 * b2 * b3 * g3
        + b1**3 * b4 * g3
        + b2**2 * b3 * g0 * g3
        - 2 * b1 * b3**2 * g0 * g3
        + b1**2 * b5 * g0 * g3
        - b3 * b4 * g0**2 * g3
        + b2 * b5 * g0**2 * g3
        - b2 * b3**2 * g1 * g3
        + b2**2 * b4 * g1 * g3
        + b1 * b3 * b4 * g1 * g3
        - b1 * b2 * b5 * g1 * g3
        + b4**2 * g0 * g1 * g3
        - b3 * b5 * g0 * g1 * g3
        + b3**3 * g2 * g3
        - 2 * b2 * b3 * b4 * g2 * g3
        + b1 * b4**2 * g2 * g3
        + b2**2 * b5 * g2 * g3
        - b1 * b3 * b5 * g2 * g3
        - b2**4 * g4
        + 3 * b1 * b2**2 * b3 * g4
        - b1**2 * b3**2 * g4
        - 2 * b1**2 * b2 * b4 * g4
        + b1**3 * b5 * g4
        + 2 * b2 * b3**2 * g0 * g4
        - 2 * b2**2 * b4 * g0 * g4
        - 2 * b1 * b3 * b4 * g0 * g4
        + 2 * b1 * b2 * b5 * g0 * g4
        - b4**2 * g0**2 * g4
        + b3 * b5 * g0**2 * g4
        - b3**3 * g1 * g4
        + 2 * b2 * b3 * b4 * g1 * g4
        - b1 * b4**2 * g1 * g4
        - b2**2 * b5 * g1 * g4
        + b1 * b3 * b5 * g1 * g4
    ) / den

    q5 = (
        -(b1**5)
        - 4 * b1**3 * b2 * g0
        - 3 * b1 * b2**2 * g0**2
        - 3 * b1**2 * b3 * g0**2
        - 2 * b2 * b3 * g0**3
        - 2 * b1 * b4 * g0**3
        - b5 * g0**4
        + 3 * b1**2 * b2**2 * g1
        - 3 * b1**3 * b3 * g1
        + 2 * b2**3 * g0 * g1
        + 2 * b1 * b2 * b3 * g0 * g1
        - 4 * b1**2 * b4 * g0 * g1
        + b3**2 * g0**2 * g1
        + 2 * b2 * b4 * g0**2 * g1
        - 3 * b1 * b5 * g0**2 * g1
        - b2**2 * b3 * g1**2
        - 2 * b1 * b3**2 * g1**2
        + 4 * b1 * b2 * b4 * g1**2
        - b1**2 * b5 * g1**2
        - 2 * b3 * b4 * g0 * g1**2
        + 2 * b2 * b5 * g0 * g1**2
        + b4**2 * g1**3
        - b3 * b5 * g1**3
        - 2 * b1 * b2**3 * g2
        + 4 * b1**2 * b2 * b3 * g2
        - 2 * b1**3 * b4 * g2
        - 2 * b2**2 * b3 * g0 * g2
        + 4 * b1 * b3**2 * g0 * g2
        - 2 * b1**2 * b5 * g0 * g2
        + 2 * b3 * b4 * g0**2 * g2
        - 2 * b2 * b5 * g0**2 * g2
        + 2 * b2 * b3**2 * g1 * g2
        - 2 * b2**2 * b4 * g1 * g2
        - 2 * b1 * b3 * b4 * g1 * g2
        + 2 * b1 * b2 * b5 * g1 * g2
        - 2 * b4**2 * g0 * g1 * g2
        + 2 * b3 * b5 * g0 * g1 * g2
        - b3**3 * g2**2
        + 2 * b2 * b3 * b4 * g2**2
        - b1 * b4**2 * g2**2
        - b2**2 * b5 * g2**2
        + b1 * b3 * b5 * g2**2
        + b2**4 * g3
        - 3 * b1 * b2**2 * b3 * g3
        + b1**2 * b3**2 * g3
        + 2 * b1**2 * b2 * b4 * g3
        - b1**3 * b5 * g3
        - 2 * b2 * b3**2 * g0 * g3
        + 2 * b2**2 * b4 * g0 * g3
        + 2 * b1 * b3 * b4 * g0 * g3
        - 2 * b1 * b2 * b5 * g0 * g3
        + b4**2 * g0**2 * g3
        - b3 * b5 * g0**2 * g3
        + b3**3 * g1 * g3
        - 2 * b2 * b3 * b4 * g1 * g3
        + b1 * b4**2 * g1 * g3
        + b2**2 * b5 * g1 * g3
        - b1 * b3 * b5 * g1 * g3
    ) / den

    # Calculate p coefficients
    p1 = b1
    p2 = b2 + b1 * q1
    p3 = b3 + b1 * q2 + b2 * q1
    p4 = b4 + b3 * q1 + b2 * q2 + b1 * q3
    p5 = g0 * q5

    y = tau**al
    h_pade = (p1 * y + p2 * y**2 + p3 * y**3 + p4 * y**4 + p5 * y**5) / (
        1 + q1 * y + q2 * y**2 + q3 * y**3 + q4 * y**4 + q5 * y**5
    )
    return h_pade


def _h_pade_66(tau, a, params):
    """
    Calculates Padé [6,6] approximation for the rough Heston model characteristic
    function.

    Parameters
    ----------
    tau : array_like
        Time parameter
    a : float or complex
        Transform variable
    params : dict
        Model parameters with keys:
        - H : float
            Hurst parameter
        - rho : float
            Correlation coefficient
        - nu : float
            Volatility of volatility
        - lbd : float
            Mean reversion rate

    Returns
    -------
    complex
        Padé [6,6] approximation value h(a,tau)
    """
    tau = np.asarray(tau, dtype=np.float64)
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    al = H + 0.5
    lbd = params["lbd"]

    lbdp = lbd / nu
    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa

    # Calculate b coefficients
    b1 = -a * (a + (0 + 1j)) / 2 * 1 / sp.gamma(1 + al)
    b2 = -b1 * lbd_tilde * nu * sp.gamma(1 + al) / sp.gamma(1 + 2 * al)
    b3 = (
        (-b2 * lbd_tilde * nu + nu**2 * b1**2 / 2)
        * sp.gamma(1 + 2 * al)
        / sp.gamma(1 + 3 * al)
    )
    b4 = (
        (-b3 * lbd_tilde * nu + nu**2 * b1 * b2)
        * sp.gamma(1 + 3 * al)
        / sp.gamma(1 + 4 * al)
    )
    b5 = (
        (-b4 * lbd_tilde * nu + nu**2 * (1 / 2 * b2 * b2 + b1 * b3))
        * sp.gamma(1 + 4 * al)
        / sp.gamma(1 + 5 * al)
    )
    b6 = (
        (-b5 * lbd_tilde * nu + nu**2 * (b2 * b3 + b1 * b4))
        * sp.gamma(1 + 5 * al)
        / sp.gamma(1 + 6 * al)
    )

    # Calculate g coefficients
    g0 = rm / nu
    g1 = 0 if al == 1 else -1 / (aa * nu) * 1 / sp.gamma(1 - al) * g0
    g2 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - al) / sp.gamma(1 - 2 * al) * g1 - 1 / 2 * nu**2 * g1 * g1)
    )
    g3 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (sp.gamma(1 - 2 * al) / sp.gamma(1 - 3 * al) * g2 - nu**2 * g1 * g2)
    )
    g4 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (
            sp.gamma(1 - 3 * al) / sp.gamma(1 - 4 * al) * g3
            - nu**2 * (1 / 2 * g2 * g2 + g1 * g3)
        )
    )
    g5 = (
        0
        if al == 1
        else -1
        / (aa * nu)
        * (
            sp.gamma(1 - 4 * al) / sp.gamma(1 - 5 * al) * g4
            - nu**2 * (g2 * g3 + g1 * g4)
        )
    )
    den = (
        -(g0**6)
        - 5 * b1 * g0**4 * g1
        - 6 * b1**2 * g0**2 * g1**2
        + 4 * b2 * g0**3 * g1**2
        - b1**3 * g1**3
        + 6 * b1 * b2 * g0 * g1**3
        - 3 * b3 * g0**2 * g1**3
        - b2**2 * g1**4
        - 2 * b1 * b3 * g1**4
        + 2 * b4 * g0 * g1**4
        - b5 * g1**5
        - 4 * b1**2 * g0**3 * g2
        - 4 * b2 * g0**4 * g2
        - 6 * b1**3 * g0 * g1 * g2
        + 6 * b3 * g0**3 * g1 * g2
        + 6 * b1**2 * b2 * g1**2 * g2
        - 6 * b4 * g0**2 * g1**2 * g2
        + 2 * b2 * b3 * g1**3 * g2
        + 2 * b1 * b4 * g1**3 * g2
        + 4 * b5 * g0 * g1**3 * g2
        - b1**4 * g2**2
        - 4 * b2**2 * g0**2 * g2**2
        + 7 * b1 * b3 * g0**2 * g2**2
        + 2 * b4 * g0**3 * g2**2
        - 4 * b1 * b2**2 * g1 * g2**2
        + b1**2 * b3 * g1 * g2**2
        + 2 * b2 * b3 * g0 * g1 * g2**2
        - 8 * b1 * b4 * g0 * g1 * g2**2
        - 3 * b5 * g0**2 * g1 * g2**2
        - b3**2 * g1**2 * g2**2
        - 2 * b2 * b4 * g1**2 * g2**2
        + 3 * b1 * b5 * g1**2 * g2**2
        + 2 * b1 * b2 * b3 * g2**3
        - 2 * b1**2 * b4 * g2**3
        - 2 * b3**2 * g0 * g2**3
        + 4 * b2 * b4 * g0 * g2**3
        - 2 * b1 * b5 * g0 * g2**3
        + 2 * b3 * b4 * g1 * g2**3
        - 2 * b2 * b5 * g1 * g2**3
        - b4**2 * g2**4
        + b3 * b5 * g2**4
        - 3 * b1**3 * g0**2 * g3
        - 6 * b1 * b2 * g0**3 * g3
        - 3 * b3 * g0**4 * g3
        - 2 * b1**4 * g1 * g3
        + 7 * b2**2 * g0**2 * g1 * g3
        - b1 * b3 * g0**2 * g1 * g3
        + 4 * b4 * g0**3 * g1 * g3
        + b1 * b2**2 * g1**2 * g3
        - 4 * b1**2 * b3 * g1**2 * g3
        - 8 * b2 * b3 * g0 * g1**2 * g3
        + 2 * b1 * b4 * g0 * g1**2 * g3
        - 3 * b5 * g0**2 * g1**2 * g3
        + b3**2 * g1**3 * g3
        + 2 * b2 * b4 * g1**3 * g3
        - 3 * b1 * b5 * g1**3 * g3
        + 2 * b1**3 * b2 * g2 * g3
        - 2 * b1 * b2**2 * g0 * g2 * g3
        + 8 * b1**2 * b3 * g0 * g2 * g3
        - 2 * b2 * b3 * g0**2 * g2 * g3
        + 8 * b1 * b4 * g0**2 * g2 * g3
        + 2 * b5 * g0**3 * g2 * g3
        + 2 * b2**3 * g1 * g2 * g3
        - 2 * b1 * b2 * b3 * g1 * g2 * g3
        + 6 * b3**2 * g0 * g1 * g2 * g3
        - 4 * b2 * b4 * g0 * g1 * g2 * g3
        - 2 * b1 * b5 * g0 * g1 * g2 * g3
        - 4 * b3 * b4 * g1**2 * g2 * g3
        + 4 * b2 * b5 * g1**2 * g2 * g3
        - b2**2 * b3 * g2**2 * g3
        - 2 * b1 * b3**2 * g2**2 * g3
        + 4 * b1 * b2 * b4 * g2**2 * g3
        - b1**2 * b5 * g2**2 * g3
        - 2 * b3 * b4 * g0 * g2**2 * g3
        + 2 * b2 * b5 * g0 * g2**2 * g3
        + 3 * b4**2 * g1 * g2**2 * g3
        - 3 * b3 * b5 * g1 * g2**2 * g3
        - b1**2 * b2**2 * g3**2
        + b1**3 * b3 * g3**2
        + 2 * b2**3 * g0 * g3**2
        - 6 * b1 * b2 * b3 * g0 * g3**2
        + 4 * b1**2 * b4 * g0 * g3**2
        - 3 * b3**2 * g0**2 * g3**2
        + 2 * b2 * b4 * g0**2 * g3**2
        + b1 * b5 * g0**2 * g3**2
        - 2 * b2**2 * b3 * g1 * g3**2
        + 4 * b1 * b3**2 * g1 * g3**2
        - 2 * b1**2 * b5 * g1 * g3**2
        + 4 * b3 * b4 * g0 * g1 * g3**2
        - 4 * b2 * b5 * g0 * g1 * g3**2
        - b4**2 * g1**2 * g3**2
        + b3 * b5 * g1**2 * g3**2
        + 2 * b2 * b3**2 * g2 * g3**2
        - 2 * b2**2 * b4 * g2 * g3**2
        - 2 * b1 * b3 * b4 * g2 * g3**2
        + 2 * b1 * b2 * b5 * g2 * g3**2
        - 2 * b4**2 * g0 * g2 * g3**2
        + 2 * b3 * b5 * g0 * g2 * g3**2
        - b3**3 * g3**3
        + 2 * b2 * b3 * b4 * g3**3
        - b1 * b4**2 * g3**3
        - b2**2 * b5 * g3**3
        + b1 * b3 * b5 * g3**3
        - 2 * b1**4 * g0 * g4
        - 6 * b1**2 * b2 * g0**2 * g4
        - 2 * b2**2 * g0**3 * g4
        - 4 * b1 * b3 * g0**3 * g4
        - 2 * b4 * g0**4 * g4
        + 2 * b1**3 * b2 * g1 * g4
        + 8 * b1 * b2**2 * g0 * g1 * g4
        - 2 * b1**2 * b3 * g0 * g1 * g4
        + 8 * b2 * b3 * g0**2 * g1 * g4
        - 2 * b1 * b4 * g0**2 * g1 * g4
        + 2 * b5 * g0**3 * g1 * g4
        - 2 * b2**3 * g1**2 * g4
        + 2 * b1**2 * b4 * g1**2 * g4
        - 4 * b3**2 * g0 * g1**2 * g4
        + 4 * b1 * b5 * g0 * g1**2 * g4
        + 2 * b3 * b4 * g1**3 * g4
        - 2 * b2 * b5 * g1**3 * g4
        - 2 * b1**2 * b2**2 * g2 * g4
        + 2 * b1**3 * b3 * g2 * g4
        - 4 * b2**3 * g0 * g2 * g4
        + 4 * b1 * b2 * b3 * g0 * g2 * g4
        + 2 * b3**2 * g0**2 * g2 * g4
        - 4 * b2 * b4 * g0**2 * g2 * g4
        + 2 * b1 * b5 * g0**2 * g2 * g4
        + 4 * b2**2 * b3 * g1 * g2 * g4
        - 8 * b1 * b2 * b4 * g1 * g2 * g4
        + 4 * b1**2 * b5 * g1 * g2 * g4
        - 2 * b4**2 * g1**2 * g2 * g4
        + 2 * b3 * b5 * g1**2 * g2 * g4
        - 2 * b2 * b3**2 * g2**2 * g4
        + 2 * b2**2 * b4 * g2**2 * g4
        + 2 * b1 * b3 * b4 * g2**2 * g4
        - 2 * b1 * b2 * b5 * g2**2 * g4
        + 2 * b4**2 * g0 * g2**2 * g4
        - 2 * b3 * b5 * g0 * g2**2 * g4
        + 2 * b1 * b2**3 * g3 * g4
        - 4 * b1**2 * b2 * b3 * g3 * g4
        + 2 * b1**3 * b4 * g3 * g4
        + 2 * b2**2 * b3 * g0 * g3 * g4
        - 4 * b1 * b3**2 * g0 * g3 * g4
        + 2 * b1**2 * b5 * g0 * g3 * g4
        - 2 * b3 * b4 * g0**2 * g3 * g4
        + 2 * b2 * b5 * g0**2 * g3 * g4
        - 2 * b2 * b3**2 * g1 * g3 * g4
        + 2 * b2**2 * b4 * g1 * g3 * g4
        + 2 * b1 * b3 * b4 * g1 * g3 * g4
        - 2 * b1 * b2 * b5 * g1 * g3 * g4
        + 2 * b4**2 * g0 * g1 * g3 * g4
        - 2 * b3 * b5 * g0 * g1 * g3 * g4
        + 2 * b3**3 * g2 * g3 * g4
        - 4 * b2 * b3 * b4 * g2 * g3 * g4
        + 2 * b1 * b4**2 * g2 * g3 * g4
        + 2 * b2**2 * b5 * g2 * g3 * g4
        - 2 * b1 * b3 * b5 * g2 * g3 * g4
        - b2**4 * g4**2
        + 3 * b1 * b2**2 * b3 * g4**2
        - b1**2 * b3**2 * g4**2
        - 2 * b1**2 * b2 * b4 * g4**2
        + b1**3 * b5 * g4**2
        + 2 * b2 * b3**2 * g0 * g4**2
        - 2 * b2**2 * b4 * g0 * g4**2
        - 2 * b1 * b3 * b4 * g0 * g4**2
        + 2 * b1 * b2 * b5 * g0 * g4**2
        - b4**2 * g0**2 * g4**2
        + b3 * b5 * g0**2 * g4**2
        - b3**3 * g1 * g4**2
        + 2 * b2 * b3 * b4 * g1 * g4**2
        - b1 * b4**2 * g1 * g4**2
        - b2**2 * b5 * g1 * g4**2
        + b1 * b3 * b5 * g1 * g4**2
        - b1**5 * g5
        - 4 * b1**3 * b2 * g0 * g5
        - 3 * b1 * b2**2 * g0**2 * g5
        - 3 * b1**2 * b3 * g0**2 * g5
        - 2 * b2 * b3 * g0**3 * g5
        - 2 * b1 * b4 * g0**3 * g5
        - b5 * g0**4 * g5
        + 3 * b1**2 * b2**2 * g1 * g5
        - 3 * b1**3 * b3 * g1 * g5
        + 2 * b2**3 * g0 * g1 * g5
        + 2 * b1 * b2 * b3 * g0 * g1 * g5
        - 4 * b1**2 * b4 * g0 * g1 * g5
        + b3**2 * g0**2 * g1 * g5
        + 2 * b2 * b4 * g0**2 * g1 * g5
        - 3 * b1 * b5 * g0**2 * g1 * g5
        - b2**2 * b3 * g1**2 * g5
        - 2 * b1 * b3**2 * g1**2 * g5
        + 4 * b1 * b2 * b4 * g1**2 * g5
        - b1**2 * b5 * g1**2 * g5
        - 2 * b3 * b4 * g0 * g1**2 * g5
        + 2 * b2 * b5 * g0 * g1**2 * g5
        + b4**2 * g1**3 * g5
        - b3 * b5 * g1**3 * g5
        - 2 * b1 * b2**3 * g2 * g5
        + 4 * b1**2 * b2 * b3 * g2 * g5
        - 2 * b1**3 * b4 * g2 * g5
        - 2 * b2**2 * b3 * g0 * g2 * g5
        + 4 * b1 * b3**2 * g0 * g2 * g5
        - 2 * b1**2 * b5 * g0 * g2 * g5
        + 2 * b3 * b4 * g0**2 * g2 * g5
        - 2 * b2 * b5 * g0**2 * g2 * g5
        + 2 * b2 * b3**2 * g1 * g2 * g5
        - 2 * b2**2 * b4 * g1 * g2 * g5
        - 2 * b1 * b3 * b4 * g1 * g2 * g5
        + 2 * b1 * b2 * b5 * g1 * g2 * g5
        - 2 * b4**2 * g0 * g1 * g2 * g5
        + 2 * b3 * b5 * g0 * g1 * g2 * g5
        - b3**3 * g2**2 * g5
        + 2 * b2 * b3 * b4 * g2**2 * g5
        - b1 * b4**2 * g2**2 * g5
        - b2**2 * b5 * g2**2 * g5
        + b1 * b3 * b5 * g2**2 * g5
        + b2**4 * g3 * g5
        - 3 * b1 * b2**2 * b3 * g3 * g5
        + b1**2 * b3**2 * g3 * g5
        + 2 * b1**2 * b2 * b4 * g3 * g5
        - b1**3 * b5 * g3 * g5
        - 2 * b2 * b3**2 * g0 * g3 * g5
        + 2 * b2**2 * b4 * g0 * g3 * g5
        + 2 * b1 * b3 * b4 * g0 * g3 * g5
        - 2 * b1 * b2 * b5 * g0 * g3 * g5
        + b4**2 * g0**2 * g3 * g5
        - b3 * b5 * g0**2 * g3 * g5
        + b3**3 * g1 * g3 * g5
        - 2 * b2 * b3 * b4 * g1 * g3 * g5
        + b1 * b4**2 * g1 * g3 * g5
        + b2**2 * b5 * g1 * g3 * g5
        - b1 * b3 * b5 * g1 * g3 * g5
    )

    # Calculate q coefficients
    q1 = (
        -(b1 * g0**5)
        - 4 * b1**2 * g0**3 * g1
        + b2 * g0**4 * g1
        - 3 * b1**3 * g0 * g1**2
        + 6 * b1 * b2 * g0**2 * g1**2
        - b3 * g0**3 * g1**2
        + 3 * b1**2 * b2 * g1**3
        - 2 * b2**2 * g0 * g1**3
        - 4 * b1 * b3 * g0 * g1**3
        + b4 * g0**2 * g1**3
        + 2 * b2 * b3 * g1**4
        + 2 * b1 * b4 * g1**4
        - b5 * g0 * g1**4
        + b6 * g1**5
        - 3 * b1**3 * g0**2 * g2
        - 2 * b1 * b2 * g0**3 * g2
        + b3 * g0**4 * g2
        - 2 * b1**4 * g1 * g2
        + 6 * b1**2 * b2 * g0 * g1 * g2
        + b2**2 * g0**2 * g1 * g2
        + 5 * b1 * b3 * g0**2 * g1 * g2
        - 2 * b4 * g0**3 * g1 * g2
        - 5 * b1 * b2**2 * g1**2 * g2
        - 4 * b1**2 * b3 * g1**2 * g2
        - 2 * b2 * b3 * g0 * g1**2 * g2
        - 4 * b1 * b4 * g0 * g1**2 * g2
        + 3 * b5 * g0**2 * g1**2 * g2
        - b3**2 * g1**3 * g2
        - 2 * b2 * b4 * g1**3 * g2
        - b1 * b5 * g1**3 * g2
        - 4 * b6 * g0 * g1**3 * g2
        + 2 * b1**3 * b2 * g2**2
        - 2 * b1 * b2**2 * g0 * g2**2
        + 5 * b1**2 * b3 * g0 * g2**2
        + b2 * b3 * g0**2 * g2**2
        - b1 * b4 * g0**2 * g2**2
        - b5 * g0**3 * g2**2
        + 2 * b2**3 * g1 * g2**2
        + 4 * b1 * b2 * b3 * g1 * g2**2
        - 3 * b1**2 * b4 * g1 * g2**2
        + 2 * b2 * b4 * g0 * g1 * g2**2
        + 4 * b1 * b5 * g0 * g1 * g2**2
        + 3 * b6 * g0**2 * g1 * g2**2
        + 2 * b3 * b4 * g1**2 * g2**2
        + b2 * b5 * g1**2 * g2**2
        - 3 * b1 * b6 * g1**2 * g2**2
        - b2**2 * b3 * g2**3
        - 2 * b1 * b3**2 * g2**3
        + 2 * b1 * b2 * b4 * g2**3
        + b1**2 * b5 * g2**3
        - 2 * b2 * b5 * g0 * g2**3
        + 2 * b1 * b6 * g0 * g2**3
        - b4**2 * g1 * g2**3
        - b3 * b5 * g1 * g2**3
        + 2 * b2 * b6 * g1 * g2**3
        + b4 * b5 * g2**4
        - b3 * b6 * g2**4
        - 2 * b1**4 * g0 * g3
        - 3 * b1**2 * b2 * g0**2 * g3
        + b2**2 * g0**3 * g3
        - b1 * b3 * g0**3 * g3
        + b4 * g0**4 * g3
        + 4 * b1**3 * b2 * g1 * g3
        + 6 * b1 * b2**2 * g0 * g1 * g3
        - 3 * b2 * b3 * g0**2 * g1 * g3
        + 3 * b1 * b4 * g0**2 * g1 * g3
        - 2 * b5 * g0**3 * g1 * g3
        - b2**3 * g1**2 * g3
        + 4 * b1**2 * b4 * g1**2 * g3
        + 3 * b3**2 * g0 * g1**2 * g3
        + 4 * b2 * b4 * g0 * g1**2 * g3
        - b1 * b5 * g0 * g1**2 * g3
        + 3 * b6 * g0**2 * g1**2 * g3
        - 2 * b3 * b4 * g1**3 * g3
        - b2 * b5 * g1**3 * g3
        + 3 * b1 * b6 * g1**3 * g3
        - 4 * b1**2 * b2**2 * g2 * g3
        + 2 * b1**3 * b3 * g2 * g3
        - 6 * b1 * b2 * b3 * g0 * g2 * g3
        - 2 * b2 * b4 * g0**2 * g2 * g3
        - 4 * b1 * b5 * g0**2 * g2 * g3
        - 2 * b6 * g0**3 * g2 * g3
        - 2 * b2**2 * b3 * g1 * g2 * g3
        + 4 * b1 * b3**2 * g1 * g2 * g3
        - 2 * b1 * b2 * b4 * g1 * g2 * g3
        - 4 * b3 * b4 * g0 * g1 * g2 * g3
        + 2 * b2 * b5 * g0 * g1 * g2 * g3
        + 2 * b1 * b6 * g0 * g1 * g2 * g3
        + 2 * b4**2 * g1**2 * g2 * g3
        + 2 * b3 * b5 * g1**2 * g2 * g3
        - 4 * b2 * b6 * g1**2 * g2 * g3
        + 2 * b2 * b3**2 * g2**2 * g3
        - b2**2 * b4 * g2**2 * g3
        - 2 * b1 * b2 * b5 * g2**2 * g3
        + b1**2 * b6 * g2**2 * g3
        + b4**2 * g0 * g2**2 * g3
        + b3 * b5 * g0 * g2**2 * g3
        - 2 * b2 * b6 * g0 * g2**2 * g3
        - 3 * b4 * b5 * g1 * g2**2 * g3
        + 3 * b3 * b6 * g1 * g2**2 * g3
        + 2 * b1 * b2**3 * g3**2
        - 3 * b1**2 * b2 * b3 * g3**2
        + b1**3 * b4 * g3**2
        + 2 * b1 * b2 * b4 * g0 * g3**2
        - 2 * b1**2 * b5 * g0 * g3**2
        + 2 * b3 * b4 * g0**2 * g3**2
        - b2 * b5 * g0**2 * g3**2
        - b1 * b6 * g0**2 * g3**2
        + 2 * b2**2 * b4 * g1 * g3**2
        - 4 * b1 * b3 * b4 * g1 * g3**2
        + 2 * b1**2 * b6 * g1 * g3**2
        - 2 * b4**2 * g0 * g1 * g3**2
        - 2 * b3 * b5 * g0 * g1 * g3**2
        + 4 * b2 * b6 * g0 * g1 * g3**2
        + b4 * b5 * g1**2 * g3**2
        - b3 * b6 * g1**2 * g3**2
        - b3**3 * g2 * g3**2
        + b1 * b4**2 * g2 * g3**2
        + b2**2 * b5 * g2 * g3**2
        + b1 * b3 * b5 * g2 * g3**2
        - 2 * b1 * b2 * b6 * g2 * g3**2
        + 2 * b4 * b5 * g0 * g2 * g3**2
        - 2 * b3 * b6 * g0 * g2 * g3**2
        + b3**2 * b4 * g3**3
        - b2 * b4**2 * g3**3
        - b2 * b3 * b5 * g3**3
        + b1 * b4 * b5 * g3**3
        + b2**2 * b6 * g3**3
        - b1 * b3 * b6 * g3**3
        - b1**5 * g4
        - 2 * b1**3 * b2 * g0 * g4
        + b1 * b2**2 * g0**2 * g4
        - b1**2 * b3 * g0**2 * g4
        + 2 * b2 * b3 * g0**3 * g4
        + b5 * g0**4 * g4
        + b1**2 * b2**2 * g1 * g4
        - 3 * b1**3 * b3 * g1 * g4
        - 2 * b2**3 * g0 * g1 * g4
        - 2 * b1 * b2 * b3 * g0 * g1 * g4
        - 2 * b1**2 * b4 * g0 * g1 * g4
        - 3 * b3**2 * g0**2 * g1 * g4
        - 4 * b2 * b4 * g0**2 * g1 * g4
        + b1 * b5 * g0**2 * g1 * g4
        - 2 * b6 * g0**3 * g1 * g4
        + 3 * b2**2 * b3 * g1**2 * g4
        - 2 * b1 * b3**2 * g1**2 * g4
        - b1**2 * b5 * g1**2 * g4
        + 4 * b3 * b4 * g0 * g1**2 * g4
        - 4 * b1 * b6 * g0 * g1**2 * g4
        - b4**2 * g1**3 * g4
        - b3 * b5 * g1**3 * g4
        + 2 * b2 * b6 * g1**3 * g4
        + 2 * b1**2 * b2 * b3 * g2 * g4
        - 2 * b1**3 * b4 * g2 * g4
        + 4 * b2**2 * b3 * g0 * g2 * g4
        - 4 * b1 * b2 * b4 * g0 * g2 * g4
        + 2 * b2 * b5 * g0**2 * g2 * g4
        - 2 * b1 * b6 * g0**2 * g2 * g4
        - 4 * b2 * b3**2 * g1 * g2 * g4
        + 4 * b1 * b3 * b4 * g1 * g2 * g4
        + 4 * b1 * b2 * b5 * g1 * g2 * g4
        - 4 * b1**2 * b6 * g1 * g2 * g4
        + 2 * b4 * b5 * g1**2 * g2 * g4
        - 2 * b3 * b6 * g1**2 * g2 * g4
        + b3**3 * g2**2 * g4
        - b1 * b4**2 * g2**2 * g4
        - b2**2 * b5 * g2**2 * g4
        - b1 * b3 * b5 * g2**2 * g4
        + 2 * b1 * b2 * b6 * g2**2 * g4
        - 2 * b4 * b5 * g0 * g2**2 * g4
        + 2 * b3 * b6 * g0 * g2**2 * g4
        - b2**4 * g3 * g4
    )
    q1 += (
        +b1 * b2**2 * b3 * g3 * g4
        + b1**2 * b3**2 * g3 * g4
        - b1**3 * b5 * g3 * g4
        - 2 * b2**2 * b4 * g0 * g3 * g4
        + 4 * b1 * b3 * b4 * g0 * g3 * g4
        - 2 * b1**2 * b6 * g0 * g3 * g4
        + b4**2 * g0**2 * g3 * g4
        + b3 * b5 * g0**2 * g3 * g4
        - 2 * b2 * b6 * g0**2 * g3 * g4
        + b3**3 * g1 * g3 * g4
        - b1 * b4**2 * g1 * g3 * g4
        - b2**2 * b5 * g1 * g3 * g4
        - b1 * b3 * b5 * g1 * g3 * g4
        + 2 * b1 * b2 * b6 * g1 * g3 * g4
        - 2 * b4 * b5 * g0 * g1 * g3 * g4
        + 2 * b3 * b6 * g0 * g1 * g3 * g4
        - 2 * b3**2 * b4 * g2 * g3 * g4
        + 2 * b2 * b4**2 * g2 * g3 * g4
        + 2 * b2 * b3 * b5 * g2 * g3 * g4
        - 2 * b1 * b4 * b5 * g2 * g3 * g4
        - 2 * b2**2 * b6 * g2 * g3 * g4
        + 2 * b1 * b3 * b6 * g2 * g3 * g4
        + b2**3 * b3 * g4**2
        - 2 * b1 * b2 * b3**2 * g4**2
        - b1 * b2**2 * b4 * g4**2
        + 2 * b1**2 * b3 * b4 * g4**2
        + b1**2 * b2 * b5 * g4**2
        - b1**3 * b6 * g4**2
        - b3**3 * g0 * g4**2
        + b1 * b4**2 * g0 * g4**2
        + b2**2 * b5 * g0 * g4**2
        + b1 * b3 * b5 * g0 * g4**2
        - 2 * b1 * b2 * b6 * g0 * g4**2
        + b4 * b5 * g0**2 * g4**2
        - b3 * b6 * g0**2 * g4**2
        + b3**2 * b4 * g1 * g4**2
        - b2 * b4**2 * g1 * g4**2
        - b2 * b3 * b5 * g1 * g4**2
        + b1 * b4 * b5 * g1 * g4**2
        + b2**2 * b6 * g1 * g4**2
        - b1 * b3 * b6 * g1 * g4**2
        + b1**4 * b2 * g5
        + 3 * b1**2 * b2**2 * g0 * g5
        + b1**3 * b3 * g0 * g5
        + b2**3 * g0**2 * g5
        + 4 * b1 * b2 * b3 * g0**2 * g5
        + b1**2 * b4 * g0**2 * g5
        + b3**2 * g0**3 * g5
        + 2 * b2 * b4 * g0**3 * g5
        + b1 * b5 * g0**3 * g5
        + b6 * g0**4 * g5
        - 2 * b1 * b2**3 * g1 * g5
        + b1**2 * b2 * b3 * g1 * g5
        + b1**3 * b4 * g1 * g5
        - 4 * b2**2 * b3 * g0 * g1 * g5
        + 2 * b1 * b2 * b4 * g0 * g1 * g5
        + 2 * b1**2 * b5 * g0 * g1 * g5
        - 2 * b3 * b4 * g0**2 * g1 * g5
        - b2 * b5 * g0**2 * g1 * g5
        + 3 * b1 * b6 * g0**2 * g1 * g5
        + 2 * b2 * b3**2 * g1**2 * g5
        - b2**2 * b4 * g1**2 * g5
        - 2 * b1 * b2 * b5 * g1**2 * g5
        + b1**2 * b6 * g1**2 * g5
        + b4**2 * g0 * g1**2 * g5
        + b3 * b5 * g0 * g1**2 * g5
        - 2 * b2 * b6 * g0 * g1**2 * g5
        - b4 * b5 * g1**3 * g5
        + b3 * b6 * g1**3 * g5
        + b2**4 * g2 * g5
        - b1 * b2**2 * b3 * g2 * g5
        - b1**2 * b3**2 * g2 * g5
        + b1**3 * b5 * g2 * g5
        + 2 * b2**2 * b4 * g0 * g2 * g5
        - 4 * b1 * b3 * b4 * g0 * g2 * g5
        + 2 * b1**2 * b6 * g0 * g2 * g5
        - b4**2 * g0**2 * g2 * g5
        - b3 * b5 * g0**2 * g2 * g5
        + 2 * b2 * b6 * g0**2 * g2 * g5
        - b3**3 * g1 * g2 * g5
        + b1 * b4**2 * g1 * g2 * g5
        + b2**2 * b5 * g1 * g2 * g5
        + b1 * b3 * b5 * g1 * g2 * g5
        - 2 * b1 * b2 * b6 * g1 * g2 * g5
        + 2 * b4 * b5 * g0 * g1 * g2 * g5
        - 2 * b3 * b6 * g0 * g1 * g2 * g5
        + b3**2 * b4 * g2**2 * g5
        - b2 * b4**2 * g2**2 * g5
        - b2 * b3 * b5 * g2**2 * g5
        + b1 * b4 * b5 * g2**2 * g5
        + b2**2 * b6 * g2**2 * g5
        - b1 * b3 * b6 * g2**2 * g5
        - b2**3 * b3 * g3 * g5
        + 2 * b1 * b2 * b3**2 * g3 * g5
        + b1 * b2**2 * b4 * g3 * g5
        - 2 * b1**2 * b3 * b4 * g3 * g5
        - b1**2 * b2 * b5 * g3 * g5
        + b1**3 * b6 * g3 * g5
        + b3**3 * g0 * g3 * g5
        - b1 * b4**2 * g0 * g3 * g5
        - b2**2 * b5 * g0 * g3 * g5
        - b1 * b3 * b5 * g0 * g3 * g5
        + 2 * b1 * b2 * b6 * g0 * g3 * g5
        - b4 * b5 * g0**2 * g3 * g5
        + b3 * b6 * g0**2 * g3 * g5
        - b3**2 * b4 * g1 * g3 * g5
        + b2 * b4**2 * g1 * g3 * g5
        + b2 * b3 * b5 * g1 * g3 * g5
        - b1 * b4 * b5 * g1 * g3 * g5
        - b2**2 * b6 * g1 * g3 * g5
        + b1 * b3 * b6 * g1 * g3 * g5
    )
    q1 /= den
    q2 = (
        -(b1**2 * g0**4)
        - b2 * g0**5
        - 3 * b1**3 * g0**2 * g1
        - 2 * b1 * b2 * g0**3 * g1
        + b3 * g0**4 * g1
        - b1**4 * g1**2
        + 3 * b1**2 * b2 * g0 * g1**2
        + 2 * b2**2 * g0**2 * g1**2
        + b1 * b3 * g0**2 * g1**2
        - b4 * g0**3 * g1**2
        - b1 * b2**2 * g1**3
        - 2 * b1**2 * b3 * g1**3
        - 2 * b2 * b3 * g0 * g1**3
        + b5 * g0**2 * g1**3
        - b1 * b5 * g1**4
        - b6 * g0 * g1**4
        - 2 * b1**4 * g0 * g2
        - 3 * b1**2 * b2 * g0**2 * g2
        - 3 * b2**2 * g0**3 * g2
        + 3 * b1 * b3 * g0**3 * g2
        + b4 * g0**4 * g2
        + 4 * b1**3 * b2 * g1 * g2
        + 6 * b1**2 * b3 * g0 * g1 * g2
        + 3 * b2 * b3 * g0**2 * g1 * g2
        - 3 * b1 * b4 * g0**2 * g1 * g2
        - 2 * b5 * g0**3 * g1 * g2
        + b2**3 * g1**2 * g2
        + 2 * b1 * b2 * b3 * g1**2 * g2
        + b3**2 * g0 * g1**2 * g2
        + 5 * b1 * b5 * g0 * g1**2 * g2
        + 3 * b6 * g0**2 * g1**2 * g2
        + b2 * b5 * g1**3 * g2
        - b1 * b6 * g1**3 * g2
        - 3 * b1**2 * b2**2 * g2**2
        + 2 * b1**3 * b3 * g2**2
        - 2 * b2**3 * g0 * g2**2
        + 2 * b1 * b2 * b3 * g0 * g2**2
        - 3 * b1**2 * b4 * g0 * g2**2
        - 2 * b3**2 * g0**2 * g2**2
        + 3 * b2 * b4 * g0**2 * g2**2
        - 4 * b1 * b5 * g0**2 * g2**2
        - b6 * g0**3 * g2**2
        - b2**2 * b3 * g1 * g2**2
        - 2 * b1 * b3**2 * g1 * g2**2
        + 3 * b1**2 * b5 * g1 * g2**2
        - 4 * b2 * b5 * g0 * g1 * g2**2
        + 4 * b1 * b6 * g0 * g1 * g2**2
        - b3 * b5 * g1**2 * g2**2
        + b2 * b6 * g1**2 * g2**2
        + b2**2 * b4 * g2**3
        + 2 * b1 * b3 * b4 * g2**3
        - 4 * b1 * b2 * b5 * g2**3
        + b1**2 * b6 * g2**3
        - b4**2 * g0 * g2**3
        + 3 * b3 * b5 * g0 * g2**3
        - 2 * b2 * b6 * g0 * g2**3
        + b4 * b5 * g1 * g2**3
        - b3 * b6 * g1 * g2**3
        - b5**2 * g2**4
        + b4 * b6 * g2**4
        - b1**5 * g3
        - 2 * b1**3 * b2 * g0 * g3
        - 2 * b1 * b2**2 * g0**2 * g3
        + 2 * b1**2 * b3 * g0**2 * g3
        - b2 * b3 * g0**3 * g3
        + 3 * b1 * b4 * g0**3 * g3
        + b5 * g0**4 * g3
        - b1**2 * b2**2 * g1 * g3
        - b1**3 * b3 * g1 * g3
        + 2 * b2**3 * g0 * g1 * g3
        - 8 * b1 * b2 * b3 * g0 * g1 * g3
        - 3 * b2 * b4 * g0**2 * g1 * g3
        - 3 * b1 * b5 * g0**2 * g1 * g3
        - 2 * b6 * g0**3 * g1 * g3
        - b2**2 * b3 * g1**2 * g3
        + 3 * b1 * b3**2 * g1**2 * g3
        - 2 * b1**2 * b5 * g1**2 * g3
        + b2 * b5 * g0 * g1**2 * g3
        - b1 * b6 * g0 * g1**2 * g3
        + b3 * b5 * g1**3 * g3
        - b2 * b6 * g1**3 * g3
        + 2 * b1 * b2**3 * g2 * g3
        - 2 * b1**3 * b4 * g2 * g3
        + 2 * b2**2 * b3 * g0 * g2 * g3
        - 2 * b1 * b3**2 * g0 * g2 * g3
        + 6 * b1 * b2 * b4 * g0 * g2 * g3
        - 6 * b1**2 * b5 * g0 * g2 * g3
        + 4 * b2 * b5 * g0**2 * g2 * g3
        - 4 * b1 * b6 * g0**2 * g2 * g3
        + 2 * b2 * b3**2 * g1 * g2 * g3
        - 2 * b2**2 * b4 * g1 * g2 * g3
        - 4 * b1 * b3 * b4 * g1 * g2 * g3
        + 4 * b1 * b2 * b5 * g1 * g2 * g3
        + 2 * b4**2 * g0 * g1 * g2 * g3
        - 4 * b3 * b5 * g0 * g1 * g2 * g3
        + 2 * b2 * b6 * g0 * g1 * g2 * g3
        - 2 * b4 * b5 * g1**2 * g2 * g3
        + 2 * b3 * b6 * g1**2 * g2 * g3
        - 2 * b2 * b3 * b4 * g2**2 * g3
        - b1 * b4**2 * g2**2 * g3
        + 2 * b2**2 * b5 * g2**2 * g3
        + 3 * b1 * b3 * b5 * g2**2 * g3
        - 2 * b1 * b2 * b6 * g2**2 * g3
        - b4 * b5 * g0 * g2**2 * g3
        + b3 * b6 * g0 * g2**2 * g3
        + 3 * b5**2 * g1 * g2**2 * g3
        - 3 * b4 * b6 * g1 * g2**2 * g3
        - 2 * b1 * b2**2 * b3 * g3**2
        + b1**2 * b3**2 * g3**2
    )
    q2 += (
        +3 * b1**2 * b2 * b4 * g3**2
        - 2 * b1**3 * b5 * g3**2
        - 2 * b2**2 * b4 * g0 * g3**2
        + 2 * b1 * b3 * b4 * g0 * g3**2
        + 2 * b1 * b2 * b5 * g0 * g3**2
        - 2 * b1**2 * b6 * g0 * g3**2
        - b4**2 * g0**2 * g3**2
        + 2 * b3 * b5 * g0**2 * g3**2
        - b2 * b6 * g0**2 * g3**2
        - b3**3 * g1 * g3**2
        + 2 * b2 * b3 * b4 * g1 * g3**2
        + b1 * b4**2 * g1 * g3**2
        - b2**2 * b5 * g1 * g3**2
        - b1 * b3 * b5 * g1 * g3**2
        + 2 * b4 * b5 * g0 * g1 * g3**2
        - 2 * b3 * b6 * g0 * g1 * g3**2
        - b5**2 * g1**2 * g3**2
        + b4 * b6 * g1**2 * g3**2
        + b3**2 * b4 * g2 * g3**2
        + b2 * b4**2 * g2 * g3**2
        - 3 * b2 * b3 * b5 * g2 * g3**2
        - b1 * b4 * b5 * g2 * g3**2
        + b2**2 * b6 * g2 * g3**2
        + b1 * b3 * b6 * g2 * g3**2
        - 2 * b5**2 * g0 * g2 * g3**2
        + 2 * b4 * b6 * g0 * g2 * g3**2
        - b3 * b4**2 * g3**3
        + b3**2 * b5 * g3**3
        + b2 * b4 * b5 * g3**3
        - b1 * b5**2 * g3**3
        - b2 * b3 * b6 * g3**3
        + b1 * b4 * b6 * g3**3
        + b1**4 * b2 * g4
        + b1**2 * b2**2 * g0 * g4
        + 3 * b1**3 * b3 * g0 * g4
        - b2**3 * g0**2 * g4
        + 4 * b1 * b2 * b3 * g0**2 * g4
        + 3 * b1**2 * b4 * g0**2 * g4
        + b3**2 * g0**3 * g4
        + 3 * b1 * b5 * g0**3 * g4
        + b6 * g0**4 * g4
        - b1**2 * b2 * b3 * g1 * g4
        + b1**3 * b4 * g1 * g4
        + 2 * b2**2 * b3 * g0 * g1 * g4
        - 6 * b1 * b2 * b4 * g0 * g1 * g4
        + 4 * b1**2 * b5 * g0 * g1 * g4
        - b2 * b5 * g0**2 * g1 * g4
        + b1 * b6 * g0**2 * g1 * g4
        - 2 * b2 * b3**2 * g1**2 * g4
        + b2**2 * b4 * g1**2 * g4
        + 2 * b1 * b3 * b4 * g1**2 * g4
        - b1**2 * b6 * g1**2 * g4
        - b4**2 * g0 * g1**2 * g4
        + b3 * b5 * g0 * g1**2 * g4
        + b4 * b5 * g1**3 * g4
        - b3 * b6 * g1**3 * g4
        - b2**4 * g2 * g4
        + 3 * b1 * b2**2 * b3 * g2 * g4
        - 3 * b1**2 * b3**2 * g2 * g4
        + b1**3 * b5 * g2 * g4
        - 2 * b2 * b3**2 * g0 * g2 * g4
        + 2 * b2**2 * b4 * g0 * g2 * g4
        - 2 * b1 * b3 * b4 * g0 * g2 * g4
        + 2 * b1 * b2 * b5 * g0 * g2 * g4
        + b4**2 * g0**2 * g2 * g4
        - 3 * b3 * b5 * g0**2 * g2 * g4
        + 2 * b2 * b6 * g0**2 * g2 * g4
        + b3**3 * g1 * g2 * g4
        + 2 * b2 * b3 * b4 * g1 * g2 * g4
        + b1 * b4**2 * g1 * g2 * g4
        - 3 * b2**2 * b5 * g1 * g2 * g4
        - 5 * b1 * b3 * b5 * g1 * g2 * g4
        + 4 * b1 * b2 * b6 * g1 * g2 * g4
        - 2 * b5**2 * g1**2 * g2 * g4
        + 2 * b4 * b6 * g1**2 * g2 * g4
        - b3**2 * b4 * g2**2 * g4
        - b2 * b4**2 * g2**2 * g4
        + 3 * b2 * b3 * b5 * g2**2 * g4
        + b1 * b4 * b5 * g2**2 * g4
        - b2**2 * b6 * g2**2 * g4
        - b1 * b3 * b6 * g2**2 * g4
        + 2 * b5**2 * g0 * g2**2 * g4
        - 2 * b4 * b6 * g0 * g2**2 * g4
        + b2**3 * b3 * g3 * g4
        - 3 * b1 * b2**2 * b4 * g3 * g4
        + 3 * b1**2 * b2 * b5 * g3 * g4
        - b1**3 * b6 * g3 * g4
        + b3**3 * g0 * g3 * g4
        - 2 * b2 * b3 * b4 * g0 * g3 * g4
        - b1 * b4**2 * g0 * g3 * g4
        + b2**2 * b5 * g0 * g3 * g4
        + b1 * b3 * b5 * g0 * g3 * g4
        - b4 * b5 * g0**2 * g3 * g4
        + b3 * b6 * g0**2 * g3 * g4
        - b3**2 * b4 * g1 * g3 * g4
        - b2 * b4**2 * g1 * g3 * g4
        + 3 * b2 * b3 * b5 * g1 * g3 * g4
        + b1 * b4 * b5 * g1 * g3 * g4
        - b2**2 * b6 * g1 * g3 * g4
        - b1 * b3 * b6 * g1 * g3 * g4
        + 2 * b5**2 * g0 * g1 * g3 * g4
        - 2 * b4 * b6 * g0 * g1 * g3 * g4
        + 2 * b3 * b4**2 * g2 * g3 * g4
        - 2 * b3**2 * b5 * g2 * g3 * g4
        - 2 * b2 * b4 * b5 * g2 * g3 * g4
        + 2 * b1 * b5**2 * g2 * g3 * g4
        + 2 * b2 * b3 * b6 * g2 * g3 * g4
        - 2 * b1 * b4 * b6 * g2 * g3 * g4
        - b2**2 * b3**2 * g4**2
        + b1 * b3**3 * g4**2
        + b2**3 * b4 * g4**2
        - b1 * b2**2 * b5 * g4**2
        - b1**2 * b3 * b5 * g4**2
        + b1**2 * b2 * b6 * g4**2
        + b3**2 * b4 * g0 * g4**2
        + b2 * b4**2 * g0 * g4**2
        - 3 * b2 * b3 * b5 * g0 * g4**2
        - b1 * b4 * b5 * g0 * g4**2
        + b2**2 * b6 * g0 * g4**2
        + b1 * b3 * b6 * g0 * g4**2
        - b5**2 * g0**2 * g4**2
        + b4 * b6 * g0**2 * g4**2
        - b3 * b4**2 * g1 * g4**2
        + b3**2 * b5 * g1 * g4**2
        + b2 * b4 * b5 * g1 * g4**2
        - b1 * b5**2 * g1 * g4**2
        - b2 * b3 * b6 * g1 * g4**2
        + b1 * b4 * b6 * g1 * g4**2
        - b1**3 * b2**2 * g5
        + b1**4 * b3 * g5
        - 2 * b1 * b2**3 * g0 * g5
        + b1**2 * b2 * b3 * g0 * g5
        + b1**3 * b4 * g0 * g5
        - 2 * b2**2 * b3 * g0**2 * g5
        + b1 * b3**2 * g0**2 * g5
        + b1**2 * b5 * g0**2 * g5
        - b2 * b5 * g0**3 * g5
        + b1 * b6 * g0**3 * g5
        + b2**4 * g1 * g5
        - b1 * b2**2 * b3 * g1 * g5
        + 2 * b1**2 * b3**2 * g1 * g5
        - 3 * b1**2 * b2 * b4 * g1 * g5
        + b1**3 * b5 * g1 * g5
        + 2 * b2 * b3**2 * g0 * g1 * g5
        - 4 * b1 * b2 * b5 * g0 * g1 * g5
        + 2 * b1**2 * b6 * g0 * g1 * g5
        + b3 * b5 * g0**2 * g1 * g5
        - b2 * b6 * g0**2 * g1 * g5
        - 2 * b2 * b3 * b4 * g1**2 * g5
        - b1 * b4**2 * g1**2 * g5
        + 2 * b2**2 * b5 * g1**2 * g5
        + 3 * b1 * b3 * b5 * g1**2 * g5
        - 2 * b1 * b2 * b6 * g1**2 * g5
        - b4 * b5 * g0 * g1**2 * g5
        + b3 * b6 * g0 * g1**2 * g5
        + b5**2 * g1**3 * g5
        - b4 * b6 * g1**3 * g5
        - b2**3 * b3 * g2 * g5
        + 3 * b1 * b2**2 * b4 * g2 * g5
        - 3 * b1**2 * b2 * b5 * g2 * g5
        + b1**3 * b6 * g2 * g5
        - b3**3 * g0 * g2 * g5
        + 2 * b2 * b3 * b4 * g0 * g2 * g5
        + b1 * b4**2 * g0 * g2 * g5
        - b2**2 * b5 * g0 * g2 * g5
        - b1 * b3 * b5 * g0 * g2 * g5
        + b4 * b5 * g0**2 * g2 * g5
        - b3 * b6 * g0**2 * g2 * g5
        + b3**2 * b4 * g1 * g2 * g5
        + b2 * b4**2 * g1 * g2 * g5
        - 3 * b2 * b3 * b5 * g1 * g2 * g5
        - b1 * b4 * b5 * g1 * g2 * g5
        + b2**2 * b6 * g1 * g2 * g5
        + b1 * b3 * b6 * g1 * g2 * g5
        - 2 * b5**2 * g0 * g1 * g2 * g5
        + 2 * b4 * b6 * g0 * g1 * g2 * g5
        - b3 * b4**2 * g2**2 * g5
        + b3**2 * b5 * g2**2 * g5
        + b2 * b4 * b5 * g2**2 * g5
        - b1 * b5**2 * g2**2 * g5
        - b2 * b3 * b6 * g2**2 * g5
        + b1 * b4 * b6 * g2**2 * g5
        + b2**2 * b3**2 * g3 * g5
        - b1 * b3**3 * g3 * g5
        - b2**3 * b4 * g3 * g5
        + b1 * b2**2 * b5 * g3 * g5
        + b1**2 * b3 * b5 * g3 * g5
        - b1**2 * b2 * b6 * g3 * g5
        - b3**2 * b4 * g0 * g3 * g5
        - b2 * b4**2 * g0 * g3 * g5
        + 3 * b2 * b3 * b5 * g0 * g3 * g5
        + b1 * b4 * b5 * g0 * g3 * g5
        - b2**2 * b6 * g0 * g3 * g5
        - b1 * b3 * b6 * g0 * g3 * g5
        + b5**2 * g0**2 * g3 * g5
        - b4 * b6 * g0**2 * g3 * g5
        + b3 * b4**2 * g1 * g3 * g5
        - b3**2 * b5 * g1 * g3 * g5
        - b2 * b4 * b5 * g1 * g3 * g5
        + b1 * b5**2 * g1 * g3 * g5
        + b2 * b3 * b6 * g1 * g3 * g5
        - b1 * b4 * b6 * g1 * g3 * g5
    )
    q2 /= den
    q3 = (
        -(b1**3 * g0**3)
        - 2 * b1 * b2 * g0**4
        - b3 * g0**5
        - 2 * b1**4 * g0 * g1
        - 3 * b1**2 * b2 * g0**2 * g1
        + 2 * b2**2 * g0**3 * g1
        - 2 * b1 * b3 * g0**3 * g1
        + b4 * g0**4 * g1
        + 2 * b1**3 * b2 * g1**2
        + 5 * b1 * b2**2 * g0 * g1**2
        - 2 * b1**2 * b3 * g0 * g1**2
        - b2 * b3 * g0**2 * g1**2
        + b1 * b4 * g0**2 * g1**2
        - b5 * g0**3 * g1**2
        - b2**3 * g1**3
        + 2 * b1**2 * b4 * g1**3
        + 2 * b2 * b4 * g0 * g1**3
        + b6 * g0**2 * g1**3
        - b2 * b5 * g1**4
        + b1 * b6 * g1**4
        - b1**5 * g2
        - 2 * b1**3 * b2 * g0 * g2
        - 2 * b1 * b2**2 * g0**2 * g2
        + 2 * b1**2 * b3 * g0**2 * g2
        - b2 * b3 * g0**3 * g2
        + 3 * b1 * b4 * g0**3 * g2
        + b5 * g0**4 * g2
        - b1**2 * b2**2 * g1 * g2
        - b1**3 * b3 * g1 * g2
        - 4 * b1 * b2 * b3 * g0 * g1 * g2
        - 2 * b1**2 * b4 * g0 * g1 * g2
        + 2 * b3**2 * g0**2 * g1 * g2
        - 5 * b2 * b4 * g0**2 * g1 * g2
        - 3 * b1 * b5 * g0**2 * g1 * g2
        - 2 * b6 * g0**3 * g1 * g2
        + b2**2 * b3 * g1**2 * g2
        + b1 * b3**2 * g1**2 * g2
        - 2 * b1 * b2 * b4 * g1**2 * g2
        - 2 * b3 * b4 * g0 * g1**2 * g2
        + 3 * b2 * b5 * g0 * g1**2 * g2
        - b1 * b6 * g0 * g1**2 * g2
        + b3 * b5 * g1**3 * g2
        - b2 * b6 * g1**3 * g2
        + 2 * b1**2 * b2 * b3 * g2**2
        - 2 * b1**3 * b4 * g2**2
        + b2**2 * b3 * g0 * g2**2
        + 2 * b1 * b2 * b4 * g0 * g2**2
        - 3 * b1**2 * b5 * g0 * g2**2
        + 2 * b2 * b5 * g0**2 * g2**2
        - 2 * b1 * b6 * g0**2 * g2**2
        - b2**2 * b4 * g1 * g2**2
        + 2 * b1 * b2 * b5 * g1 * g2**2
        - b1**2 * b6 * g1 * g2**2
        + 2 * b4**2 * g0 * g1 * g2**2
        - 2 * b3 * b5 * g0 * g1 * g2**2
        - b4 * b5 * g1**2 * g2**2
        + b3 * b6 * g1**2 * g2**2
        - b1 * b4**2 * g2**3
        + b1 * b3 * b5 * g2**3
        - b4 * b5 * g0 * g2**3
        + b3 * b6 * g0 * g2**3
        + b5**2 * g1 * g2**3
        - b4 * b6 * g1 * g2**3
        + b1**4 * b2 * g3
        + b1**2 * b2**2 * g0 * g3
        + 3 * b1**3 * b3 * g0 * g3
        + 2 * b2**3 * g0**2 * g3
        - 2 * b1 * b2 * b3 * g0**2 * g3
        + 6 * b1**2 * b4 * g0**2 * g3
        - 2 * b3**2 * g0**3 * g3
        + 3 * b2 * b4 * g0**3 * g3
        + 3 * b1 * b5 * g0**3 * g3
        + b6 * g0**4 * g3
        + 2 * b1 * b2**3 * g1 * g3
        - 5 * b1**2 * b2 * b3 * g1 * g3
        + 3 * b1**3 * b4 * g1 * g3
        - 2 * b2**2 * b3 * g0 * g1 * g3
        + 2 * b1 * b3**2 * g0 * g1 * g3
        + 4 * b3 * b4 * g0**2 * g1 * g3
        - 5 * b2 * b5 * g0**2 * g1 * g3
        + b1 * b6 * g0**2 * g1 * g3
        - b2 * b3**2 * g1**2 * g3
        + 3 * b2**2 * b4 * g1**2 * g3
        - 2 * b1 * b3 * b4 * g1**2 * g3
        - 2 * b1 * b2 * b5 * g1**2 * g3
        + 2 * b1**2 * b6 * g1**2 * g3
        - 2 * b4**2 * g0 * g1**2 * g3
        - b3 * b5 * g0 * g1**2 * g3
        + 3 * b2 * b6 * g0 * g1**2 * g3
        + b4 * b5 * g1**3 * g3
        - b3 * b6 * g1**3 * g3
        - 2 * b1 * b2**2 * b3 * g2 * g3
        + 4 * b1**2 * b2 * b4 * g2 * g3
        - 2 * b1**3 * b5 * g2 * g3
        - 2 * b2**2 * b4 * g0 * g2 * g3
        + 4 * b1 * b2 * b5 * g0 * g2 * g3
        - 2 * b1**2 * b6 * g0 * g2 * g3
        - 2 * b4**2 * g0**2 * g2 * g3
        + 2 * b3 * b5 * g0**2 * g2 * g3
        + 4 * b1 * b4**2 * g1 * g2 * g3
        - 4 * b1 * b3 * b5 * g1 * g2 * g3
        + 4 * b4 * b5 * g0 * g1 * g2 * g3
        - 4 * b3 * b6 * g0 * g1 * g2 * g3
        - 2 * b5**2 * g1**2 * g2 * g3
        + 2 * b4 * b6 * g1**2 * g2 * g3
        + b2 * b4**2 * g2**2 * g3
        - b2 * b3 * b5 * g2**2 * g3
        - b1 * b4 * b5 * g2**2 * g3
        + b1 * b3 * b6 * g2**2 * g3
        - b5**2 * g0 * g2**2 * g3
        + b4 * b6 * g0 * g2**2 * g3
        + 2 * b1 * b2 * b3**2 * g3**2
        - 2 * b1 * b2**2 * b4 * g3**2
        - 2 * b1**2 * b3 * b4 * g3**2
        + 2 * b1**2 * b2 * b5 * g3**2
        - b3**3 * g0 * g3**2
        + 4 * b2 * b3 * b4 * g0 * g3**2
        - 5 * b1 * b4**2 * g0 * g3**2
        - 3 * b2**2 * b5 * g0 * g3**2
        + 3 * b1 * b3 * b5 * g0 * g3**2
        + 2 * b1 * b2 * b6 * g0 * g3**2
        - 2 * b4 * b5 * g0**2 * g3**2
        + 2 * b3 * b6 * g0**2 * g3**2
        + b3**2 * b4 * g1 * g3**2
        - 3 * b2 * b4**2 * g1 * g3**2
        + b2 * b3 * b5 * g1 * g3**2
        + 3 * b1 * b4 * b5 * g1 * g3**2
        + b2**2 * b6 * g1 * g3**2
        - 3 * b1 * b3 * b6 * g1 * g3**2
        + 2 * b5**2 * g0 * g1 * g3**2
        - 2 * b4 * b6 * g0 * g1 * g3**2
        - b3 * b4**2 * g2 * g3**2
        + b3**2 * b5 * g2 * g3**2
        + b2 * b4 * b5 * g2 * g3**2
        - b1 * b5**2 * g2 * g3**2
        - b2 * b3 * b6 * g2 * g3**2
        + b1 * b4 * b6 * g2 * g3**2
        + b4**3 * g3**3
        - 2 * b3 * b4 * b5 * g3**3
        + b2 * b5**2 * g3**3
        + b3**2 * b6 * g3**3
        - b2 * b4 * b6 * g3**3
        - b1**3 * b2**2 * g4
        + b1**4 * b3 * g4
        - 3 * b1**2 * b2 * b3 * g0 * g4
        + 3 * b1**3 * b4 * g0 * g4
        - 3 * b1 * b3**2 * g0**2 * g4
        + 3 * b1**2 * b5 * g0**2 * g4
        - 2 * b3 * b4 * g0**3 * g4
        + b2 * b5 * g0**3 * g4
        + b1 * b6 * g0**3 * g4
        - b2**4 * g1 * g4
        + 3 * b1 * b2**2 * b3 * g1 * g4
        + 2 * b1**2 * b3**2 * g1 * g4
        - 5 * b1**2 * b2 * b4 * g1 * g4
        + b1**3 * b5 * g1 * g4
        + 2 * b2 * b3**2 * g0 * g1 * g4
        - 2 * b2**2 * b4 * g0 * g1 * g4
        + 4 * b1 * b3 * b4 * g0 * g1 * g4
        - 4 * b1 * b2 * b5 * g0 * g1 * g4
        + 2 * b4**2 * g0**2 * g1 * g4
        + b3 * b5 * g0**2 * g1 * g4
        - 3 * b2 * b6 * g0**2 * g1 * g4
        - 3 * b1 * b4**2 * g1**2 * g4
        + 3 * b1 * b3 * b5 * g1**2 * g4
        - 3 * b4 * b5 * g0 * g1**2 * g4
        + 3 * b3 * b6 * g0 * g1**2 * g4
        + b5**2 * g1**3 * g4
        - b4 * b6 * g1**3 * g4
        + b2**3 * b3 * g2 * g4
        - 2 * b1 * b2 * b3**2 * g2 * g4
        - b1 * b2**2 * b4 * g2 * g4
        + 2 * b1**2 * b3 * b4 * g2 * g4
        + b1**2 * b2 * b5 * g2 * g4
        - b1**3 * b6 * g2 * g4
        + b3**3 * g0 * g2 * g4
        - 4 * b2 * b3 * b4 * g0 * g2 * g4
        + 3 * b1 * b4**2 * g0 * g2 * g4
        + 3 * b2**2 * b5 * g0 * g2 * g4
        - b1 * b3 * b5 * g0 * g2 * g4
        - 2 * b1 * b2 * b6 * g0 * g2 * g4
        + b4 * b5 * g0**2 * g2 * g4
        - b3 * b6 * g0**2 * g2 * g4
        - b3**2 * b4 * g1 * g2 * g4
        + b2 * b4**2 * g1 * g2 * g4
        + b2 * b3 * b5 * g1 * g2 * g4
        - b1 * b4 * b5 * g1 * g2 * g4
        - b2**2 * b6 * g1 * g2 * g4
    )
    q3 += (
        +b1 * b3 * b6 * g1 * g2 * g4
        + b3 * b4**2 * g2**2 * g4
        - b3**2 * b5 * g2**2 * g4
        - b2 * b4 * b5 * g2**2 * g4
        + b1 * b5**2 * g2**2 * g4
        + b2 * b3 * b6 * g2**2 * g4
        - b1 * b4 * b6 * g2**2 * g4
        - b2**2 * b3**2 * g3 * g4
        - b1 * b3**3 * g3 * g4
        + b2**3 * b4 * g3 * g4
        + 4 * b1 * b2 * b3 * b4 * g3 * g4
        - 2 * b1**2 * b4**2 * g3 * g4
        - 3 * b1 * b2**2 * b5 * g3 * g4
        + b1**2 * b3 * b5 * g3 * g4
        + b1**2 * b2 * b6 * g3 * g4
        - b3**2 * b4 * g0 * g3 * g4
        + 3 * b2 * b4**2 * g0 * g3 * g4
        - b2 * b3 * b5 * g0 * g3 * g4
        - 3 * b1 * b4 * b5 * g0 * g3 * g4
        - b2**2 * b6 * g0 * g3 * g4
        + 3 * b1 * b3 * b6 * g0 * g3 * g4
        - b5**2 * g0**2 * g3 * g4
        + b4 * b6 * g0**2 * g3 * g4
        + b3 * b4**2 * g1 * g3 * g4
        - b3**2 * b5 * g1 * g3 * g4
        - b2 * b4 * b5 * g1 * g3 * g4
        + b1 * b5**2 * g1 * g3 * g4
        + b2 * b3 * b6 * g1 * g3 * g4
        - b1 * b4 * b6 * g1 * g3 * g4
        - 2 * b4**3 * g2 * g3 * g4
        + 4 * b3 * b4 * b5 * g2 * g3 * g4
        - 2 * b2 * b5**2 * g2 * g3 * g4
        - 2 * b3**2 * b6 * g2 * g3 * g4
        + 2 * b2 * b4 * b6 * g2 * g3 * g4
        + b2 * b3**3 * g4**2
        - 2 * b2**2 * b3 * b4 * g4**2
        - b1 * b3**2 * b4 * g4**2
        + 2 * b1 * b2 * b4**2 * g4**2
        + b2**3 * b5 * g4**2
        - b1**2 * b4 * b5 * g4**2
        - b1 * b2**2 * b6 * g4**2
        + b1**2 * b3 * b6 * g4**2
        - b3 * b4**2 * g0 * g4**2
        + b3**2 * b5 * g0 * g4**2
        + b2 * b4 * b5 * g0 * g4**2
        - b1 * b5**2 * g0 * g4**2
        - b2 * b3 * b6 * g0 * g4**2
        + b1 * b4 * b6 * g0 * g4**2
        + b4**3 * g1 * g4**2
        - 2 * b3 * b4 * b5 * g1 * g4**2
        + b2 * b5**2 * g1 * g4**2
        + b3**2 * b6 * g1 * g4**2
        - b2 * b4 * b6 * g1 * g4**2
        + b1**2 * b2**3 * g5
        - 2 * b1**3 * b2 * b3 * g5
        + b1**4 * b4 * g5
        + b2**4 * g0 * g5
        - b1 * b2**2 * b3 * g0 * g5
        - 2 * b1**2 * b3**2 * g0 * g5
        + b1**2 * b2 * b4 * g0 * g5
        + b1**3 * b5 * g0 * g5
        - b2 * b3**2 * g0**2 * g5
        + 2 * b2**2 * b4 * g0**2 * g5
        - 2 * b1 * b3 * b4 * g0**2 * g5
        + b1**2 * b6 * g0**2 * g5
        - b3 * b5 * g0**3 * g5
        + b2 * b6 * g0**3 * g5
        - b2**3 * b3 * g1 * g5
        + 3 * b1 * b2**2 * b4 * g1 * g5
        - 3 * b1**2 * b2 * b5 * g1 * g5
        + b1**3 * b6 * g1 * g5
        + 2 * b1 * b4**2 * g0 * g1 * g5
        - 2 * b1 * b3 * b5 * g0 * g1 * g5
        + b4 * b5 * g0**2 * g1 * g5
        - b3 * b6 * g0**2 * g1 * g5
        + b2 * b4**2 * g1**2 * g5
        - b2 * b3 * b5 * g1**2 * g5
        - b1 * b4 * b5 * g1**2 * g5
        + b1 * b3 * b6 * g1**2 * g5
        - b5**2 * g0 * g1**2 * g5
        + b4 * b6 * g0 * g1**2 * g5
        + b2**2 * b3**2 * g2 * g5
        + b1 * b3**3 * g2 * g5
        - b2**3 * b4 * g2 * g5
        - 4 * b1 * b2 * b3 * b4 * g2 * g5
        + 2 * b1**2 * b4**2 * g2 * g5
        + 3 * b1 * b2**2 * b5 * g2 * g5
        - b1**2 * b3 * b5 * g2 * g5
        - b1**2 * b2 * b6 * g2 * g5
        + b3**2 * b4 * g0 * g2 * g5
        - 3 * b2 * b4**2 * g0 * g2 * g5
        + b2 * b3 * b5 * g0 * g2 * g5
        + 3 * b1 * b4 * b5 * g0 * g2 * g5
        + b2**2 * b6 * g0 * g2 * g5
        - 3 * b1 * b3 * b6 * g0 * g2 * g5
        + b5**2 * g0**2 * g2 * g5
        - b4 * b6 * g0**2 * g2 * g5
        - b3 * b4**2 * g1 * g2 * g5
        + b3**2 * b5 * g1 * g2 * g5
        + b2 * b4 * b5 * g1 * g2 * g5
        - b1 * b5**2 * g1 * g2 * g5
        - b2 * b3 * b6 * g1 * g2 * g5
        + b1 * b4 * b6 * g1 * g2 * g5
        + b4**3 * g2**2 * g5
        - 2 * b3 * b4 * b5 * g2**2 * g5
        + b2 * b5**2 * g2**2 * g5
        + b3**2 * b6 * g2**2 * g5
        - b2 * b4 * b6 * g2**2 * g5
        - b2 * b3**3 * g3 * g5
        + 2 * b2**2 * b3 * b4 * g3 * g5
        + b1 * b3**2 * b4 * g3 * g5
        - 2 * b1 * b2 * b4**2 * g3 * g5
        - b2**3 * b5 * g3 * g5
        + b1**2 * b4 * b5 * g3 * g5
        + b1 * b2**2 * b6 * g3 * g5
        - b1**2 * b3 * b6 * g3 * g5
        + b3 * b4**2 * g0 * g3 * g5
        - b3**2 * b5 * g0 * g3 * g5
        - b2 * b4 * b5 * g0 * g3 * g5
        + b1 * b5**2 * g0 * g3 * g5
        + b2 * b3 * b6 * g0 * g3 * g5
        - b1 * b4 * b6 * g0 * g3 * g5
        - b4**3 * g1 * g3 * g5
        + 2 * b3 * b4 * b5 * g1 * g3 * g5
        - b2 * b5**2 * g1 * g3 * g5
        - b3**2 * b6 * g1 * g3 * g5
        + b2 * b4 * b6 * g1 * g3 * g5
    )
    q3 /= den
    q4 = (
        -(b1**4 * g0**2)
        - 3 * b1**2 * b2 * g0**3
        - b2**2 * g0**4
        - 2 * b1 * b3 * g0**4
        - b4 * g0**5
        - b1**5 * g1
        - 2 * b1**3 * b2 * g0 * g1
        + 3 * b1 * b2**2 * g0**2 * g1
        - 3 * b1**2 * b3 * g0**2 * g1
        + 4 * b2 * b3 * g0**3 * g1
        - 2 * b1 * b4 * g0**3 * g1
        + b5 * g0**4 * g1
        + 2 * b1**2 * b2**2 * g1**2
        - 3 * b1**3 * b3 * g1**2
        - b2**3 * g0 * g1**2
        - 2 * b1**2 * b4 * g0 * g1**2
        - 3 * b3**2 * g0**2 * g1**2
        - b2 * b4 * g0**2 * g1**2
        + b1 * b5 * g0**2 * g1**2
        - b6 * g0**3 * g1**2
        + b2**2 * b3 * g1**3
        - 2 * b1 * b3**2 * g1**3
        + 2 * b1 * b2 * b4 * g1**3
        - b1**2 * b5 * g1**3
        + 2 * b3 * b4 * g0 * g1**3
        - 2 * b1 * b6 * g0 * g1**3
        - b3 * b5 * g1**4
        + b2 * b6 * g1**4
        + b1**4 * b2 * g2
        + b1**2 * b2**2 * g0 * g2
        + 3 * b1**3 * b3 * g0 * g2
        - 2 * b2**3 * g0**2 * g2
        + 6 * b1 * b2 * b3 * g0**2 * g2
        + 2 * b1**2 * b4 * g0**2 * g2
        + 2 * b3**2 * g0**3 * g2
        - b2 * b4 * g0**3 * g2
        + 3 * b1 * b5 * g0**3 * g2
        + b6 * g0**4 * g2
        - 2 * b1 * b2**3 * g1 * g2
        + 3 * b1**2 * b2 * b3 * g1 * g2
        - b1**3 * b4 * g1 * g2
        + 2 * b2**2 * b3 * g0 * g1 * g2
        + 2 * b1 * b3**2 * g0 * g1 * g2
        - 8 * b1 * b2 * b4 * g0 * g1 * g2
        + 4 * b1**2 * b5 * g0 * g1 * g2
        - b2 * b5 * g0**2 * g1 * g2
        + b1 * b6 * g0**2 * g1 * g2
        - b2 * b3**2 * g1**2 * g2
        - b2**2 * b4 * g1**2 * g2
        + 2 * b1 * b3 * b4 * g1**2 * g2
        + 2 * b1 * b2 * b5 * g1**2 * g2
        - 2 * b1**2 * b6 * g1**2 * g2
        - 2 * b4**2 * g0 * g1**2 * g2
        + 3 * b3 * b5 * g0 * g1**2 * g2
        - b2 * b6 * g0 * g1**2 * g2
        + b4 * b5 * g1**3 * g2
        - b3 * b6 * g1**3 * g2
        + b1 * b2**2 * b3 * g2**2
        - 2 * b1**2 * b3**2 * g2**2
        + b1**3 * b5 * g2**2
        - 2 * b2 * b3**2 * g0 * g2**2
        + 3 * b2**2 * b4 * g0 * g2**2
        - 2 * b1 * b3 * b4 * g0 * g2**2
        + b1**2 * b6 * g0 * g2**2
        + b4**2 * g0**2 * g2**2
        - 3 * b3 * b5 * g0**2 * g2**2
        + 2 * b2 * b6 * g0**2 * g2**2
        + 2 * b2 * b3 * b4 * g1 * g2**2
        - 2 * b2**2 * b5 * g1 * g2**2
        - 2 * b1 * b3 * b5 * g1 * g2**2
        + 2 * b1 * b2 * b6 * g1 * g2**2
        - b5**2 * g1**2 * g2**2
        + b4 * b6 * g1**2 * g2**2
        - b2 * b4**2 * g2**3
        + b2 * b3 * b5 * g2**3
        + b1 * b4 * b5 * g2**3
        - b1 * b3 * b6 * g2**3
        + b5**2 * g0 * g2**3
        - b4 * b6 * g0 * g2**3
        - b1**3 * b2**2 * g3
        + b1**4 * b3 * g3
        - 3 * b1**2 * b2 * b3 * g0 * g3
        + 3 * b1**3 * b4 * g0 * g3
        - 3 * b1 * b3**2 * g0**2 * g3
        + 3 * b1**2 * b5 * g0**2 * g3
        - 2 * b3 * b4 * g0**3 * g3
        + b2 * b5 * g0**3 * g3
        + b1 * b6 * g0**3 * g3
        + 3 * b1**2 * b3**2 * g1 * g3
        - 3 * b1**2 * b2 * b4 * g1 * g3
        + 6 * b1 * b3 * b4 * g0 * g1 * g3
        - 6 * b1 * b2 * b5 * g0 * g1 * g3
        + 3 * b4**2 * g0**2 * g1 * g3
        - 3 * b2 * b6 * g0**2 * g1 * g3
        + b3**3 * g1**2 * g3
        - 2 * b2 * b3 * b4 * g1**2 * g3
        - 2 * b1 * b4**2 * g1**2 * g3
        + b2**2 * b5 * g1**2 * g3
        + 2 * b1 * b3 * b5 * g1**2 * g3
        - 3 * b4 * b5 * g0 * g1**2 * g3
        + 3 * b3 * b6 * g0 * g1**2 * g3
        + b5**2 * g1**3 * g3
        - b4 * b6 * g1**3 * g3
        + 2 * b3**3 * g0 * g2 * g3
        - 4 * b2 * b3 * b4 * g0 * g2 * g3
        + 2 * b1 * b4**2 * g0 * g2 * g3
        + 2 * b2**2 * b5 * g0 * g2 * g3
        - 2 * b1 * b3 * b5 * g0 * g2 * g3
        - 2 * b3**2 * b4 * g1 * g2 * g3
        + 2 * b2 * b4**2 * g1 * g2 * g3
        + 2 * b2 * b3 * b5 * g1 * g2 * g3
        - 2 * b1 * b4 * b5 * g1 * g2 * g3
        - 2 * b2**2 * b6 * g1 * g2 * g3
        + 2 * b1 * b3 * b6 * g1 * g2 * g3
        + b3 * b4**2 * g2**2 * g3
        - b3**2 * b5 * g2**2 * g3
        - b2 * b4 * b5 * g2**2 * g3
        + b1 * b5**2 * g2**2 * g3
        + b2 * b3 * b6 * g2**2 * g3
        - b1 * b4 * b6 * g2**2 * g3
        - b1 * b3**3 * g3**2
        + 2 * b1 * b2 * b3 * b4 * g3**2
        - b1**2 * b4**2 * g3**2
        - b1 * b2**2 * b5 * g3**2
        + b1**2 * b3 * b5 * g3**2
        - b3**2 * b4 * g0 * g3**2
        + b2 * b4**2 * g0 * g3**2
        + b2 * b3 * b5 * g0 * g3**2
        - b1 * b4 * b5 * g0 * g3**2
        - b2**2 * b6 * g0 * g3**2
        + b1 * b3 * b6 * g0 * g3**2
        + b3 * b4**2 * g1 * g3**2
        - b3**2 * b5 * g1 * g3**2
        - b2 * b4 * b5 * g1 * g3**2
        + b1 * b5**2 * g1 * g3**2
        + b2 * b3 * b6 * g1 * g3**2
        - b1 * b4 * b6 * g1 * g3**2
        - b4**3 * g2 * g3**2
        + 2 * b3 * b4 * b5 * g2 * g3**2
        - b2 * b5**2 * g2 * g3**2
        - b3**2 * b6 * g2 * g3**2
        + b2 * b4 * b6 * g2 * g3**2
        + b1**2 * b2**3 * g4
        - 2 * b1**3 * b2 * b3 * g4
        + b1**4 * b4 * g4
        - b2**4 * g0 * g4
        + 5 * b1 * b2**2 * b3 * g0 * g4
        - 4 * b1**2 * b3**2 * g0 * g4
        - 3 * b1**2 * b2 * b4 * g0 * g4
        + 3 * b1**3 * b5 * g0 * g4
        + 3 * b2 * b3**2 * g0**2 * g4
        - 2 * b2**2 * b4 * g0**2 * g4
        - 6 * b1 * b3 * b4 * g0**2 * g4
        + 4 * b1 * b2 * b5 * g0**2 * g4
        + b1**2 * b6 * g0**2 * g4
        - 2 * b4**2 * g0**3 * g4
        + b3 * b5 * g0**3 * g4
        + b2 * b6 * g0**3 * g4
        + b2**3 * b3 * g1 * g4
        - 4 * b1 * b2 * b3**2 * g1 * g4
        + b1 * b2**2 * b4 * g1 * g4
        + 4 * b1**2 * b3 * b4 * g1 * g4
        - b1**2 * b2 * b5 * g1 * g4
        - b1**3 * b6 * g1 * g4
        - 4 * b3**3 * g0 * g1 * g4
        + 4 * b2 * b3 * b4 * g0 * g1 * g4
        + 2 * b1 * b4**2 * g0 * g1 * g4
        + 2 * b1 * b3 * b5 * g0 * g1 * g4
        - 4 * b1 * b2 * b6 * g0 * g1 * g4
        + 3 * b4 * b5 * g0**2 * g1 * g4
        - 3 * b3 * b6 * g0**2 * g1 * g4
        + 2 * b3**2 * b4 * g1**2 * g4
        - b2 * b4**2 * g1**2 * g4
        - 3 * b2 * b3 * b5 * g1**2 * g4
        + b1 * b4 * b5 * g1**2 * g4
        + 2 * b2**2 * b6 * g1**2 * g4
        - b1 * b3 * b6 * g1**2 * g4
        - b5**2 * g0 * g1**2 * g4
        + b4 * b6 * g0 * g1**2 * g4
        - b2**2 * b3**2 * g2 * g4
        + 3 * b1 * b3**3 * g2 * g4
        + b2**3 * b4 * g2 * g4
        - 4 * b1 * b2 * b3 * b4 * g2 * g4
        + 2 * b1**2 * b4**2 * g2 * g4
        + b1 * b2**2 * b5 * g2 * g4
        - 3 * b1**2 * b3 * b5 * g2 * g4
        + b1**2 * b2 * b6 * g2 * g4
        + 3 * b3**2 * b4 * g0 * g2 * g4
        - b2 * b4**2 * g0 * g2 * g4
        - 5 * b2 * b3 * b5 * g0 * g2 * g4
        + b1 * b4 * b5 * g0 * g2 * g4
        + 3 * b2**2 * b6 * g0 * g2 * g4
        - b1 * b3 * b6 * g0 * g2 * g4
        - b5**2 * g0**2 * g2 * g4
    )
    q4 += (
        +b4 * b6 * g0**2 * g2 * g4
        - 3 * b3 * b4**2 * g1 * g2 * g4
        + 3 * b3**2 * b5 * g1 * g2 * g4
        + 3 * b2 * b4 * b5 * g1 * g2 * g4
        - 3 * b1 * b5**2 * g1 * g2 * g4
        - 3 * b2 * b3 * b6 * g1 * g2 * g4
        + 3 * b1 * b4 * b6 * g1 * g2 * g4
        + b4**3 * g2**2 * g4
        - 2 * b3 * b4 * b5 * g2**2 * g4
        + b2 * b5**2 * g2**2 * g4
        + b3**2 * b6 * g2**2 * g4
        - b2 * b4 * b6 * g2**2 * g4
        + b2 * b3**3 * g3 * g4
        - 2 * b2**2 * b3 * b4 * g3 * g4
        - b1 * b3**2 * b4 * g3 * g4
        + 2 * b1 * b2 * b4**2 * g3 * g4
        + b2**3 * b5 * g3 * g4
        - b1**2 * b4 * b5 * g3 * g4
        - b1 * b2**2 * b6 * g3 * g4
        + b1**2 * b3 * b6 * g3 * g4
        - b3 * b4**2 * g0 * g3 * g4
        + b3**2 * b5 * g0 * g3 * g4
        + b2 * b4 * b5 * g0 * g3 * g4
        - b1 * b5**2 * g0 * g3 * g4
        - b2 * b3 * b6 * g0 * g3 * g4
        + b1 * b4 * b6 * g0 * g3 * g4
        + b4**3 * g1 * g3 * g4
        - 2 * b3 * b4 * b5 * g1 * g3 * g4
        + b2 * b5**2 * g1 * g3 * g4
        + b3**2 * b6 * g1 * g3 * g4
        - b2 * b4 * b6 * g1 * g3 * g4
        - b3**4 * g4**2
        + 3 * b2 * b3**2 * b4 * g4**2
        - b2**2 * b4**2 * g4**2
        - 2 * b1 * b3 * b4**2 * g4**2
        - 2 * b2**2 * b3 * b5 * g4**2
        + 2 * b1 * b3**2 * b5 * g4**2
        + 2 * b1 * b2 * b4 * b5 * g4**2
        - b1**2 * b5**2 * g4**2
        + b2**3 * b6 * g4**2
        - 2 * b1 * b2 * b3 * b6 * g4**2
        + b1**2 * b4 * b6 * g4**2
        - b4**3 * g0 * g4**2
        + 2 * b3 * b4 * b5 * g0 * g4**2
        - b2 * b5**2 * g0 * g4**2
        - b3**2 * b6 * g0 * g4**2
        + b2 * b4 * b6 * g0 * g4**2
        - b1 * b2**4 * g5
        + 3 * b1**2 * b2**2 * b3 * g5
        - b1**3 * b3**2 * g5
        - 2 * b1**3 * b2 * b4 * g5
        + b1**4 * b5 * g5
        - b2**3 * b3 * g0 * g5
        + 4 * b1 * b2 * b3**2 * g0 * g5
        - b1 * b2**2 * b4 * g0 * g5
        - 4 * b1**2 * b3 * b4 * g0 * g5
        + b1**2 * b2 * b5 * g0 * g5
        + b1**3 * b6 * g0 * g5
        + b3**3 * g0**2 * g5
        - 2 * b1 * b4**2 * g0**2 * g5
        - b2**2 * b5 * g0**2 * g5
        + 2 * b1 * b2 * b6 * g0**2 * g5
        - b4 * b5 * g0**3 * g5
        + b3 * b6 * g0**3 * g5
        + b2**2 * b3**2 * g1 * g5
        - 2 * b1 * b3**3 * g1 * g5
        - b2**3 * b4 * g1 * g5
        + 2 * b1 * b2 * b3 * b4 * g1 * g5
        - b1**2 * b4**2 * g1 * g5
        + 2 * b1**2 * b3 * b5 * g1 * g5
        - b1**2 * b2 * b6 * g1 * g5
        - 2 * b3**2 * b4 * g0 * g1 * g5
        + 4 * b2 * b3 * b5 * g0 * g1 * g5
        - 2 * b2**2 * b6 * g0 * g1 * g5
        + b5**2 * g0**2 * g1 * g5
        - b4 * b6 * g0**2 * g1 * g5
        + b3 * b4**2 * g1**2 * g5
        - b3**2 * b5 * g1**2 * g5
        - b2 * b4 * b5 * g1**2 * g5
        + b1 * b5**2 * g1**2 * g5
        + b2 * b3 * b6 * g1**2 * g5
        - b1 * b4 * b6 * g1**2 * g5
        - b2 * b3**3 * g2 * g5
        + 2 * b2**2 * b3 * b4 * g2 * g5
        + b1 * b3**2 * b4 * g2 * g5
        - 2 * b1 * b2 * b4**2 * g2 * g5
        - b2**3 * b5 * g2 * g5
        + b1**2 * b4 * b5 * g2 * g5
        + b1 * b2**2 * b6 * g2 * g5
        - b1**2 * b3 * b6 * g2 * g5
        + b3 * b4**2 * g0 * g2 * g5
        - b3**2 * b5 * g0 * g2 * g5
        - b2 * b4 * b5 * g0 * g2 * g5
        + b1 * b5**2 * g0 * g2 * g5
        + b2 * b3 * b6 * g0 * g2 * g5
        - b1 * b4 * b6 * g0 * g2 * g5
        - b4**3 * g1 * g2 * g5
        + 2 * b3 * b4 * b5 * g1 * g2 * g5
        - b2 * b5**2 * g1 * g2 * g5
        - b3**2 * b6 * g1 * g2 * g5
        + b2 * b4 * b6 * g1 * g2 * g5
        + b3**4 * g3 * g5
        - 3 * b2 * b3**2 * b4 * g3 * g5
        + b2**2 * b4**2 * g3 * g5
        + 2 * b1 * b3 * b4**2 * g3 * g5
        + 2 * b2**2 * b3 * b5 * g3 * g5
        - 2 * b1 * b3**2 * b5 * g3 * g5
        - 2 * b1 * b2 * b4 * b5 * g3 * g5
        + b1**2 * b5**2 * g3 * g5
        - b2**3 * b6 * g3 * g5
        + 2 * b1 * b2 * b3 * b6 * g3 * g5
        - b1**2 * b4 * b6 * g3 * g5
        + b4**3 * g0 * g3 * g5
        - 2 * b3 * b4 * b5 * g0 * g3 * g5
        + b2 * b5**2 * g0 * g3 * g5
        + b3**2 * b6 * g0 * g3 * g5
        - b2 * b4 * b6 * g0 * g3 * g5
    )
    q4 /= den

    q5 = (
        -(b1**5 * g0)
        - 4 * b1**3 * b2 * g0**2
        - 3 * b1 * b2**2 * g0**3
        - 3 * b1**2 * b3 * g0**3
        - 2 * b2 * b3 * g0**4
        - 2 * b1 * b4 * g0**4
        - b5 * g0**5
        + b1**4 * b2 * g1
        + 6 * b1**2 * b2**2 * g0 * g1
        - 2 * b1**3 * b3 * g0 * g1
        + 3 * b2**3 * g0**2 * g1
        + 6 * b1 * b2 * b3 * g0**2 * g1
        - 3 * b1**2 * b4 * g0**2 * g1
        + 2 * b3**2 * g0**3 * g1
        + 4 * b2 * b4 * g0**3 * g1
        - 2 * b1 * b5 * g0**3 * g1
        + b6 * g0**4 * g1
        - 2 * b1 * b2**3 * g1**2
        + b1**2 * b2 * b3 * g1**2
        + b1**3 * b4 * g1**2
        - 5 * b2**2 * b3 * g0 * g1**2
        - 2 * b1 * b3**2 * g0 * g1**2
        + 6 * b1 * b2 * b4 * g0 * g1**2
        + b1**2 * b5 * g0 * g1**2
        - 4 * b3 * b4 * g0**2 * g1**2
        + b2 * b5 * g0**2 * g1**2
        + 3 * b1 * b6 * g0**2 * g1**2
        + 2 * b2 * b3**2 * g1**3
        - b2**2 * b4 * g1**3
        - 2 * b1 * b2 * b5 * g1**3
        + b1**2 * b6 * g1**3
        + 2 * b4**2 * g0 * g1**3
        - 2 * b2 * b6 * g0 * g1**3
        - b4 * b5 * g1**4
        + b3 * b6 * g1**4
        - b1**3 * b2**2 * g2
        + b1**4 * b3 * g2
        - 4 * b1 * b2**3 * g0 * g2
        + 5 * b1**2 * b2 * b3 * g0 * g2
        - b1**3 * b4 * g0 * g2
        - 4 * b2**2 * b3 * g0**2 * g2
        + 5 * b1 * b3**2 * g0**2 * g2
        - b1**2 * b5 * g0**2 * g2
        + 2 * b3 * b4 * g0**3 * g2
        - 3 * b2 * b5 * g0**3 * g2
        + b1 * b6 * g0**3 * g2
        + 2 * b2**4 * g1 * g2
        - 2 * b1 * b2**2 * b3 * g1 * g2
        + b1**2 * b3**2 * g1 * g2
        - 3 * b1**2 * b2 * b4 * g1 * g2
        + 2 * b1**3 * b5 * g1 * g2
        + 4 * b2 * b3**2 * g0 * g1 * g2
        - 6 * b1 * b3 * b4 * g0 * g1 * g2
        - 2 * b1 * b2 * b5 * g0 * g1 * g2
        + 4 * b1**2 * b6 * g0 * g1 * g2
        - 3 * b4**2 * g0**2 * g1 * g2
        + 2 * b3 * b5 * g0**2 * g1 * g2
        + b2 * b6 * g0**2 * g1 * g2
        - b3**3 * g1**2 * g2
        - 2 * b2 * b3 * b4 * g1**2 * g2
        + 3 * b2**2 * b5 * g1**2 * g2
        + 4 * b1 * b3 * b5 * g1**2 * g2
        - 4 * b1 * b2 * b6 * g1**2 * g2
        + b4 * b5 * g0 * g1**2 * g2
        - b3 * b6 * g0 * g1**2 * g2
        + b5**2 * g1**3 * g2
        - b4 * b6 * g1**3 * g2
        - b2**3 * b3 * g2**2
        + 3 * b1 * b2**2 * b4 * g2**2
        - 3 * b1**2 * b2 * b5 * g2**2
        + b1**3 * b6 * g2**2
        - 2 * b3**3 * g0 * g2**2
        + 4 * b2 * b3 * b4 * g0 * g2**2
        - 2 * b2**2 * b5 * g0 * g2**2
        + b4 * b5 * g0**2 * g2**2
        - b3 * b6 * g0**2 * g2**2
        + 2 * b3**2 * b4 * g1 * g2**2
        - 4 * b2 * b3 * b5 * g1 * g2**2
        + 2 * b2**2 * b6 * g1 * g2**2
        - 2 * b5**2 * g0 * g1 * g2**2
        + 2 * b4 * b6 * g0 * g1 * g2**2
        - b3 * b4**2 * g2**3
        + b3**2 * b5 * g2**3
        + b2 * b4 * b5 * g2**3
        - b1 * b5**2 * g2**3
        - b2 * b3 * b6 * g2**3
        + b1 * b4 * b6 * g2**3
        + b1**2 * b2**3 * g3
        - 2 * b1**3 * b2 * b3 * g3
        + b1**4 * b4 * g3
        + 2 * b2**4 * g0 * g3
        - 4 * b1 * b2**2 * b3 * g0 * g3
        - b1**2 * b3**2 * g0 * g3
        + 3 * b1**2 * b2 * b4 * g0 * g3
        - 3 * b2 * b3**2 * g0**2 * g3
        + 4 * b2**2 * b4 * g0**2 * g3
        - 2 * b1 * b2 * b5 * g0**2 * g3
        + b1**2 * b6 * g0**2 * g3
        + b4**2 * g0**3 * g3
        - 2 * b3 * b5 * g0**3 * g3
        + b2 * b6 * g0**3 * g3
        - 2 * b2**3 * b3 * g1 * g3
        + 2 * b1 * b2 * b3**2 * g1 * g3
        + 4 * b1 * b2**2 * b4 * g1 * g3
        - 2 * b1**2 * b3 * b4 * g1 * g3
        - 4 * b1**2 * b2 * b5 * g1 * g3
        + 2 * b1**3 * b6 * g1 * g3
        + 2 * b3**3 * g0 * g1 * g3
        - 2 * b2 * b3 * b4 * g0 * g1 * g3
        + 2 * b1 * b4**2 * g0 * g1 * g3
        - 4 * b1 * b3 * b5 * g0 * g1 * g3
        + 2 * b1 * b2 * b6 * g0 * g1 * g3
        - b3**2 * b4 * g1**2 * g3
        + 2 * b2 * b4**2 * g1**2 * g3
        - 2 * b1 * b4 * b5 * g1**2 * g3
        - b2**2 * b6 * g1**2 * g3
        + 2 * b1 * b3 * b6 * g1**2 * g3
        - b5**2 * g0 * g1**2 * g3
        + b4 * b6 * g0 * g1**2 * g3
        + 2 * b2**2 * b3**2 * g2 * g3
        - 2 * b2**3 * b4 * g2 * g3
        - 4 * b1 * b2 * b3 * b4 * g2 * g3
        + 2 * b1**2 * b4**2 * g2 * g3
        + 4 * b1 * b2**2 * b5 * g2 * g3
        - 2 * b1**2 * b2 * b6 * g2 * g3
        - 4 * b2 * b4**2 * g0 * g2 * g3
        + 4 * b2 * b3 * b5 * g0 * g2 * g3
        + 4 * b1 * b4 * b5 * g0 * g2 * g3
        - 4 * b1 * b3 * b6 * g0 * g2 * g3
        + 2 * b5**2 * g0**2 * g2 * g3
        - 2 * b4 * b6 * g0**2 * g2 * g3
        + b4**3 * g2**2 * g3
        - 2 * b3 * b4 * b5 * g2**2 * g3
        + b2 * b5**2 * g2**2 * g3
        + b3**2 * b6 * g2**2 * g3
        - b2 * b4 * b6 * g2**2 * g3
        - b2 * b3**3 * g3**2
        + 2 * b2**2 * b3 * b4 * g3**2
        + b1 * b3**2 * b4 * g3**2
        - 2 * b1 * b2 * b4**2 * g3**2
        - b2**3 * b5 * g3**2
        + b1**2 * b4 * b5 * g3**2
        + b1 * b2**2 * b6 * g3**2
        - b1**2 * b3 * b6 * g3**2
        + b3 * b4**2 * g0 * g3**2
        - b3**2 * b5 * g0 * g3**2
        - b2 * b4 * b5 * g0 * g3**2
        + b1 * b5**2 * g0 * g3**2
        + b2 * b3 * b6 * g0 * g3**2
        - b1 * b4 * b6 * g0 * g3**2
        - b4**3 * g1 * g3**2
        + 2 * b3 * b4 * b5 * g1 * g3**2
        - b2 * b5**2 * g1 * g3**2
        - b3**2 * b6 * g1 * g3**2
        + b2 * b4 * b6 * g1 * g3**2
        - b1 * b2**4 * g4
        + 3 * b1**2 * b2**2 * b3 * g4
        - b1**3 * b3**2 * g4
        - 2 * b1**3 * b2 * b4 * g4
        + b1**4 * b5 * g4
        - b2**3 * b3 * g0 * g4
        + 4 * b1 * b2 * b3**2 * g0 * g4
        - b1 * b2**2 * b4 * g0 * g4
        - 4 * b1**2 * b3 * b4 * g0 * g4
        + b1**2 * b2 * b5 * g0 * g4
        + b1**3 * b6 * g0 * g4
        + b3**3 * g0**2 * g4
        - 2 * b1 * b4**2 * g0**2 * g4
        - b2**2 * b5 * g0**2 * g4
        + 2 * b1 * b2 * b6 * g0**2 * g4
        - b4 * b5 * g0**3 * g4
        + b3 * b6 * g0**3 * g4
        + b2**2 * b3**2 * g1 * g4
        - 2 * b1 * b3**3 * g1 * g4
        - b2**3 * b4 * g1 * g4
        + 2 * b1 * b2 * b3 * b4 * g1 * g4
        - b1**2 * b4**2 * g1 * g4
        + 2 * b1**2 * b3 * b5 * g1 * g4
        - b1**2 * b2 * b6 * g1 * g4
        - 2 * b3**2 * b4 * g0 * g1 * g4
    )

    q5 += (
        +4 * b2 * b3 * b5 * g0 * g1 * g4
        - 2 * b2**2 * b6 * g0 * g1 * g4
        + b5**2 * g0**2 * g1 * g4
        - b4 * b6 * g0**2 * g1 * g4
        + b3 * b4**2 * g1**2 * g4
        - b3**2 * b5 * g1**2 * g4
        - b2 * b4 * b5 * g1**2 * g4
        + b1 * b5**2 * g1**2 * g4
        + b2 * b3 * b6 * g1**2 * g4
        - b1 * b4 * b6 * g1**2 * g4
        - b2 * b3**3 * g2 * g4
        + 2 * b2**2 * b3 * b4 * g2 * g4
        + b1 * b3**2 * b4 * g2 * g4
        - 2 * b1 * b2 * b4**2 * g2 * g4
        - b2**3 * b5 * g2 * g4
        + b1**2 * b4 * b5 * g2 * g4
        + b1 * b2**2 * b6 * g2 * g4
        - b1**2 * b3 * b6 * g2 * g4
        + b3 * b4**2 * g0 * g2 * g4
        - b3**2 * b5 * g0 * g2 * g4
        - b2 * b4 * b5 * g0 * g2 * g4
        + b1 * b5**2 * g0 * g2 * g4
        + b2 * b3 * b6 * g0 * g2 * g4
        - b1 * b4 * b6 * g0 * g2 * g4
        - b4**3 * g1 * g2 * g4
        + 2 * b3 * b4 * b5 * g1 * g2 * g4
        - b2 * b5**2 * g1 * g2 * g4
        - b3**2 * b6 * g1 * g2 * g4
        + b2 * b4 * b6 * g1 * g2 * g4
        + b3**4 * g3 * g4
        - 3 * b2 * b3**2 * b4 * g3 * g4
        + b2**2 * b4**2 * g3 * g4
        + 2 * b1 * b3 * b4**2 * g3 * g4
        + 2 * b2**2 * b3 * b5 * g3 * g4
        - 2 * b1 * b3**2 * b5 * g3 * g4
        - 2 * b1 * b2 * b4 * b5 * g3 * g4
        + b1**2 * b5**2 * g3 * g4
        - b2**3 * b6 * g3 * g4
        + 2 * b1 * b2 * b3 * b6 * g3 * g4
        - b1**2 * b4 * b6 * g3 * g4
        + b4**3 * g0 * g3 * g4
        - 2 * b3 * b4 * b5 * g0 * g3 * g4
        + b2 * b5**2 * g0 * g3 * g4
        + b3**2 * b6 * g0 * g3 * g4
        - b2 * b4 * b6 * g0 * g3 * g4
        + b2**5 * g5
        - 4 * b1 * b2**3 * b3 * g5
        + 3 * b1**2 * b2 * b3**2 * g5
        + 3 * b1**2 * b2**2 * b4 * g5
        - 2 * b1**3 * b3 * b4 * g5
        - 2 * b1**3 * b2 * b5 * g5
        + b1**4 * b6 * g5
        - 3 * b2**2 * b3**2 * g0 * g5
        + 2 * b1 * b3**3 * g0 * g5
        + 3 * b2**3 * b4 * g0 * g5
        + 2 * b1 * b2 * b3 * b4 * g0 * g5
        - b1**2 * b4**2 * g0 * g5
        - 4 * b1 * b2**2 * b5 * g0 * g5
        - 2 * b1**2 * b3 * b5 * g0 * g5
        + 3 * b1**2 * b2 * b6 * g0 * g5
        + b3**2 * b4 * g0**2 * g5
        + 2 * b2 * b4**2 * g0**2 * g5
        - 4 * b2 * b3 * b5 * g0**2 * g5
        - 2 * b1 * b4 * b5 * g0**2 * g5
        + b2**2 * b6 * g0**2 * g5
        + 2 * b1 * b3 * b6 * g0**2 * g5
        - b5**2 * g0**3 * g5
        + b4 * b6 * g0**3 * g5
        + 2 * b2 * b3**3 * g1 * g5
        - 4 * b2**2 * b3 * b4 * g1 * g5
        - 2 * b1 * b3**2 * b4 * g1 * g5
        + 4 * b1 * b2 * b4**2 * g1 * g5
        + 2 * b2**3 * b5 * g1 * g5
        - 2 * b1**2 * b4 * b5 * g1 * g5
        - 2 * b1 * b2**2 * b6 * g1 * g5
        + 2 * b1**2 * b3 * b6 * g1 * g5
        - 2 * b3 * b4**2 * g0 * g1 * g5
        + 2 * b3**2 * b5 * g0 * g1 * g5
        + 2 * b2 * b4 * b5 * g0 * g1 * g5
        - 2 * b1 * b5**2 * g0 * g1 * g5
        - 2 * b2 * b3 * b6 * g0 * g1 * g5
        + 2 * b1 * b4 * b6 * g0 * g1 * g5
        + b4**3 * g1**2 * g5
        - 2 * b3 * b4 * b5 * g1**2 * g5
        + b2 * b5**2 * g1**2 * g5
        + b3**2 * b6 * g1**2 * g5
        - b2 * b4 * b6 * g1**2 * g5
        - b3**4 * g2 * g5
        + 3 * b2 * b3**2 * b4 * g2 * g5
        - b2**2 * b4**2 * g2 * g5
        - 2 * b1 * b3 * b4**2 * g2 * g5
        - 2 * b2**2 * b3 * b5 * g2 * g5
        + 2 * b1 * b3**2 * b5 * g2 * g5
        + 2 * b1 * b2 * b4 * b5 * g2 * g5
        - b1**2 * b5**2 * g2 * g5
        + b2**3 * b6 * g2 * g5
        - 2 * b1 * b2 * b3 * b6 * g2 * g5
        + b1**2 * b4 * b6 * g2 * g5
        - b4**3 * g0 * g2 * g5
        + 2 * b3 * b4 * b5 * g0 * g2 * g5
        - b2 * b5**2 * g0 * g2 * g5
        - b3**2 * b6 * g0 * g2 * g5
        + b2 * b4 * b6 * g0 * g2 * g5
    )

    q5 /= den

    q6 = (
        -(b1**6)
        - 5 * b1**4 * b2 * g0
        - 6 * b1**2 * b2**2 * g0**2
        - 4 * b1**3 * b3 * g0**2
        - b2**3 * g0**3
        - 6 * b1 * b2 * b3 * g0**3
        - 3 * b1**2 * b4 * g0**3
        - b3**2 * g0**4
        - 2 * b2 * b4 * g0**4
        - 2 * b1 * b5 * g0**4
        - b6 * g0**5
        + 4 * b1**3 * b2**2 * g1
        - 4 * b1**4 * b3 * g1
        + 6 * b1 * b2**3 * g0 * g1
        - 6 * b1**3 * b4 * g0 * g1
        + 6 * b2**2 * b3 * g0**2 * g1
        - 6 * b1**2 * b5 * g0**2 * g1
        + 2 * b3 * b4 * g0**3 * g1
        + 2 * b2 * b5 * g0**3 * g1
        - 4 * b1 * b6 * g0**3 * g1
        - b2**4 * g1**2
        - 4 * b1**2 * b3**2 * g1**2
        + 7 * b1**2 * b2 * b4 * g1**2
        - 2 * b1**3 * b5 * g1**2
        - 4 * b2 * b3**2 * g0 * g1**2
        + b2**2 * b4 * g0 * g1**2
        - 2 * b1 * b3 * b4 * g0 * g1**2
        + 8 * b1 * b2 * b5 * g0 * g1**2
        - 3 * b1**2 * b6 * g0 * g1**2
        - b4**2 * g0**2 * g1**2
        - 2 * b3 * b5 * g0**2 * g1**2
        + 3 * b2 * b6 * g0**2 * g1**2
        + 2 * b2 * b3 * b4 * g1**3
        + 2 * b1 * b4**2 * g1**3
        - 2 * b2**2 * b5 * g1**3
        - 4 * b1 * b3 * b5 * g1**3
        + 2 * b1 * b2 * b6 * g1**3
        + 2 * b4 * b5 * g0 * g1**3
        - 2 * b3 * b6 * g0 * g1**3
        - b5**2 * g1**4
        + b4 * b6 * g1**4
        - 3 * b1**2 * b2**3 * g2
        + 6 * b1**3 * b2 * b3 * g2
        - 3 * b1**4 * b4 * g2
        - 2 * b2**4 * g0 * g2
        + 7 * b1**2 * b3**2 * g0 * g2
        - b1**2 * b2 * b4 * g0 * g2
        - 4 * b1**3 * b5 * g0 * g2
        + b2 * b3**2 * g0**2 * g2
        - 4 * b2**2 * b4 * g0**2 * g2
        + 8 * b1 * b3 * b4 * g0**2 * g2
        - 2 * b1 * b2 * b5 * g0**2 * g2
        - 3 * b1**2 * b6 * g0**2 * g2
        + b4**2 * g0**3 * g2
        + 2 * b3 * b5 * g0**3 * g2
        - 3 * b2 * b6 * g0**3 * g2
        + 2 * b2**3 * b3 * g1 * g2
        + 2 * b1 * b2 * b3**2 * g1 * g2
        - 8 * b1 * b2**2 * b4 * g1 * g2
        - 2 * b1**2 * b3 * b4 * g1 * g2
        + 8 * b1**2 * b2 * b5 * g1 * g2
        - 2 * b1**3 * b6 * g1 * g2
        + 2 * b3**3 * g0 * g1 * g2
        - 2 * b2 * b3 * b4 * g0 * g1 * g2
        - 6 * b1 * b4**2 * g0 * g1 * g2
        + 4 * b1 * b3 * b5 * g0 * g1 * g2
        + 2 * b1 * b2 * b6 * g0 * g1 * g2
        - 4 * b4 * b5 * g0**2 * g1 * g2
        + 4 * b3 * b6 * g0**2 * g1 * g2
        - b3**2 * b4 * g1**2 * g2
        - 2 * b2 * b4**2 * g1**2 * g2
        + 4 * b2 * b3 * b5 * g1**2 * g2
        + 2 * b1 * b4 * b5 * g1**2 * g2
        - b2**2 * b6 * g1**2 * g2
        - 2 * b1 * b3 * b6 * g1**2 * g2
        + 3 * b5**2 * g0 * g1**2 * g2
        - 3 * b4 * b6 * g0 * g1**2 * g2
        - b2**2 * b3**2 * g2**2
        - 2 * b1 * b3**3 * g2**2
        + b2**3 * b4 * g2**2
        + 6 * b1 * b2 * b3 * b4 * g2**2
        - 3 * b1**2 * b4**2 * g2**2
        - 4 * b1 * b2**2 * b5 * g2**2
        + 2 * b1**2 * b3 * b5 * g2**2
        + b1**2 * b2 * b6 * g2**2
        - 2 * b3**2 * b4 * g0 * g2**2
        + 4 * b2 * b4**2 * g0 * g2**2
        - 4 * b1 * b4 * b5 * g0 * g2**2
        - 2 * b2**2 * b6 * g0 * g2**2
        + 4 * b1 * b3 * b6 * g0 * g2**2
        - b5**2 * g0**2 * g2**2
        + b4 * b6 * g0**2 * g2**2
        + 2 * b3 * b4**2 * g1 * g2**2
        - 2 * b3**2 * b5 * g1 * g2**2
        - 2 * b2 * b4 * b5 * g1 * g2**2
        + 2 * b1 * b5**2 * g1 * g2**2
        + 2 * b2 * b3 * b6 * g1 * g2**2
        - 2 * b1 * b4 * b6 * g1 * g2**2
        - b4**3 * g2**3
        + 2 * b3 * b4 * b5 * g2**3
        - b2 * b5**2 * g2**3
        - b3**2 * b6 * g2**3
        + b2 * b4 * b6 * g2**3
        + 2 * b1 * b2**4 * g3
        - 6 * b1**2 * b2**2 * b3 * g3
        + 2 * b1**3 * b3**2 * g3
        + 4 * b1**3 * b2 * b4 * g3
        - 2 * b1**4 * b5 * g3
        + 2 * b2**3 * b3 * g0 * g3
        - 8 * b1 * b2 * b3**2 * g0 * g3
        + 2 * b1 * b2**2 * b4 * g0 * g3
        + 8 * b1**2 * b3 * b4 * g0 * g3
        - 2 * b1**2 * b2 * b5 * g0 * g3
        - 2 * b1**3 * b6 * g0 * g3
        - 2 * b3**3 * g0**2 * g3
        + 4 * b1 * b4**2 * g0**2 * g3
        + 2 * b2**2 * b5 * g0**2 * g3
        - 4 * b1 * b2 * b6 * g0**2 * g3
        + 2 * b4 * b5 * g0**3 * g3
        - 2 * b3 * b6 * g0**3 * g3
        - 2 * b2**2 * b3**2 * g1 * g3
        + 4 * b1 * b3**3 * g1 * g3
        + 2 * b2**3 * b4 * g1 * g3
        - 4 * b1 * b2 * b3 * b4 * g1 * g3
        + 2 * b1**2 * b4**2 * g1 * g3
        - 4 * b1**2 * b3 * b5 * g1 * g3
        + 2 * b1**2 * b2 * b6 * g1 * g3
        + 4 * b3**2 * b4 * g0 * g1 * g3
        - 8 * b2 * b3 * b5 * g0 * g1 * g3
        + 4 * b2**2 * b6 * g0 * g1 * g3
        - 2 * b5**2 * g0**2 * g1 * g3
        + 2 * b4 * b6 * g0**2 * g1 * g3
        - 2 * b3 * b4**2 * g1**2 * g3
        + 2 * b3**2 * b5 * g1**2 * g3
        + 2 * b2 * b4 * b5 * g1**2 * g3
        - 2 * b1 * b5**2 * g1**2 * g3
        - 2 * b2 * b3 * b6 * g1**2 * g3
        + 2 * b1 * b4 * b6 * g1**2 * g3
        + 2 * b2 * b3**3 * g2 * g3
        - 4 * b2**2 * b3 * b4 * g2 * g3
        - 2 * b1 * b3**2 * b4 * g2 * g3
        + 4 * b1 * b2 * b4**2 * g2 * g3
        + 2 * b2**3 * b5 * g2 * g3
        - 2 * b1**2 * b4 * b5 * g2 * g3
        - 2 * b1 * b2**2 * b6 * g2 * g3
        + 2 * b1**2 * b3 * b6 * g2 * g3
        - 2 * b3 * b4**2 * g0 * g2 * g3
        + 2 * b3**2 * b5 * g0 * g2 * g3
        + 2 * b2 * b4 * b5 * g0 * g2 * g3
        - 2 * b1 * b5**2 * g0 * g2 * g3
        - 2 * b2 * b3 * b6 * g0 * g2 * g3
        + 2 * b1 * b4 * b6 * g0 * g2 * g3
        + 2 * b4**3 * g1 * g2 * g3
        - 4 * b3 * b4 * b5 * g1 * g2 * g3
        + 2 * b2 * b5**2 * g1 * g2 * g3
        + 2 * b3**2 * b6 * g1 * g2 * g3
        - 2 * b2 * b4 * b6 * g1 * g2 * g3
        - b3**4 * g3**2
        + 3 * b2 * b3**2 * b4 * g3**2
        - b2**2 * b4**2 * g3**2
        - 2 * b1 * b3 * b4**2 * g3**2
        - 2 * b2**2 * b3 * b5 * g3**2
        + 2 * b1 * b3**2 * b5 * g3**2
        + 2 * b1 * b2 * b4 * b5 * g3**2
        - b1**2 * b5**2 * g3**2
        + b2**3 * b6 * g3**2
        - 2 * b1 * b2 * b3 * b6 * g3**2
        + b1**2 * b4 * b6 * g3**2
        - b4**3 * g0 * g3**2
        + 2 * b3 * b4 * b5 * g0 * g3**2
        - b2 * b5**2 * g0 * g3**2
        - b3**2 * b6 * g0 * g3**2
        + b2 * b4 * b6 * g0 * g3**2
        - b2**5 * g4
        + 4 * b1 * b2**3 * b3 * g4
        - 3 * b1**2 * b2 * b3**2 * g4
        - 3 * b1**2 * b2**2 * b4 * g4
        + 2 * b1**3 * b3 * b4 * g4
        + 2 * b1**3 * b2 * b5 * g4
        - b1**4 * b6 * g4
        + 3 * b2**2 * b3**2 * g0 * g4
        - 2 * b1 * b3**3 * g0 * g4
        - 3 * b2**3 * b4 * g0 * g4
        - 2 * b1 * b2 * b3 * b4 * g0 * g4
        + b1**2 * b4**2 * g0 * g4
        + 4 * b1 * b2**2 * b5 * g0 * g4
        + 2 * b1**2 * b3 * b5 * g0 * g4
        - 3 * b1**2 * b2 * b6 * g0 * g4
        - b3**2 * b4 * g0**2 * g4
        - 2 * b2 * b4**2 * g0**2 * g4
        + 4 * b2 * b3 * b5 * g0**2 * g4
        + 2 * b1 * b4 * b5 * g0**2 * g4
        - b2**2 * b6 * g0**2 * g4
        - 2 * b1 * b3 * b6 * g0**2 * g4
        + b5**2 * g0**3 * g4
        - b4 * b6 * g0**3 * g4
        - 2 * b2 * b3**3 * g1 * g4
        + 4 * b2**2 * b3 * b4 * g1 * g4
        + 2 * b1 * b3**2 * b4 * g1 * g4
        - 4 * b1 * b2 * b4**2 * g1 * g4
        - 2 * b2**3 * b5 * g1 * g4
        + 2 * b1**2 * b4 * b5 * g1 * g4
        + 2 * b1 * b2**2 * b6 * g1 * g4
        - 2 * b1**2 * b3 * b6 * g1 * g4
        + 2 * b3 * b4**2 * g0 * g1 * g4
        - 2 * b3**2 * b5 * g0 * g1 * g4
        - 2 * b2 * b4 * b5 * g0 * g1 * g4
        + 2 * b1 * b5**2 * g0 * g1 * g4
        + 2 * b2 * b3 * b6 * g0 * g1 * g4
        - 2 * b1 * b4 * b6 * g0 * g1 * g4
        - b4**3 * g1**2 * g4
        + 2 * b3 * b4 * b5 * g1**2 * g4
        - b2 * b5**2 * g1**2 * g4
        - b3**2 * b6 * g1**2 * g4
        + b2 * b4 * b6 * g1**2 * g4
        + b3**4 * g2 * g4
        - 3 * b2 * b3**2 * b4 * g2 * g4
        + b2**2 * b4**2 * g2 * g4
        + 2 * b1 * b3 * b4**2 * g2 * g4
        + 2 * b2**2 * b3 * b5 * g2 * g4
        - 2 * b1 * b3**2 * b5 * g2 * g4
        - 2 * b1 * b2 * b4 * b5 * g2 * g4
        + b1**2 * b5**2 * g2 * g4
        - b2**3 * b6 * g2 * g4
        + 2 * b1 * b2 * b3 * b6 * g2 * g4
        - b1**2 * b4 * b6 * g2 * g4
        + b4**3 * g0 * g2 * g4
        - 2 * b3 * b4 * b5 * g0 * g2 * g4
        + b2 * b5**2 * g0 * g2 * g4
        + b3**2 * b6 * g0 * g2 * g4
        - b2 * b4 * b6 * g0 * g2 * g4
    ) / den

    # Calculate p coefficients
    p1 = b1
    p2 = b2 + b1 * q1
    p3 = b3 + b1 * q2 + b2 * q1
    p4 = b4 + b3 * q1 + b2 * q2 + b1 * q3
    p5 = b5 + b4 * q1 + b3 * q2 + b2 * q3 + b1 * q4
    p6 = g0 * q6

    y = tau**al

    return (p1 * y + p2 * y**2 + p3 * y**3 + p4 * y**4 + p5 * y**5 + p6 * y**6) / (
        1 + q1 * y + q2 * y**2 + q3 * y**3 + q4 * y**4 + q5 * y**5 + q6 * y**6
    )


def _h_pade(tau, a, params, n_pade: int):
    """
    Padé approximation h(a, tau) for rough Heston.

    Selects and evaluates Padé [n,n] for the auxiliary function h(a, tau).

    Parameters
    ----------
    tau : float or array_like
        Time parameter.
    a : float or complex
        Fourier variable.
    params : dict
        Model parameters.
    n_pade : int
        Padé order (2-6).

    Returns
    -------
    complex or ndarray
        Padé approximation value.
    """
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")
    if n_pade == 2:
        return _h_pade_22(tau, a, params)
    elif n_pade == 3:
        return _h_pade_33(tau, a, params)
    elif n_pade == 4:
        return _h_pade_44(tau, a, params)
    elif n_pade == 5:
        return _h_pade_55(tau, a, params)
    elif n_pade == 6:
        return _h_pade_66(tau, a, params)


def _deriv_h_pade(tau, a, params, n_pade: int):
    """
    Auxiliary derivative for rough Heston Padé scheme.

    Computes quadratic expression in h(a, tau) for Riccati equation.

    Parameters
    ----------
    tau : float or array_like
        Time parameter.
    a : float or complex
        Fourier variable.
    params : dict
        Model parameters.
    n_pade : int
        Padé order (2-6).

    Returns
    -------
    complex or ndarray
        Derivative value.
    """
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")

    rho = params["rho"]
    nu = params["nu"]
    lbd = params["lbd"]

    lbdp = lbd / nu

    lbd_tilde = lbdp - (0 + 1j) * rho * a
    aa = np.sqrt(a * (a + (0 + 1j)) + lbd_tilde**2)
    rm = lbd_tilde - aa
    rp = lbd_tilde + aa

    # Call the appropriate Padé approximation based on order n
    h_pade_n = _h_pade(tau, a, params, n_pade)

    return 0.5 * (nu * h_pade_n - rm) * (nu * h_pade_n - rp)


def g(a, t, params, n_pade):
    r"""
    Auxiliary function g(a, t) for rough Heston.

    Used in log characteristic function integral:
    log E[e^{i * a * log(S_t/S_0)}] = \int_t^T \xi_t(s) * g(T-s; a) ds.

    Parameters
    ----------
    a : float or complex
        Fourier variable.
    t : float or array_like
        Time parameter.
    params : dict
        Model parameters.
    n_pade : int
        Padé order (2-6).

    Returns
    -------
    complex or ndarray
        Value of g(a, t).
    """
    lbd = params["lbd"]
    return lbd * _h_pade(t, a, params, n_pade) + _deriv_h_pade(t, a, params, n_pade)
