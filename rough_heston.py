import numpy as np
import scipy.special as sp
from scipy import integrate
from heston import psi_minus, psi_plus
from utils import black_impvol, gauss_legendre, lewis_formula_otm_price


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


def _g(a, t, params, n_pade):
    r"""
    Compute the auxiliary function g(a, t) for the rough Heston model.
    The function is needed to compute the log characteristic function:
        log E[e^{i * a * log(S_t/S_0)}] = \int_t^T \xi_t(s) * g(T-s; a) ds.
    """
    lbd = params["lbd"]
    return lbd * _h_pade(t, a, params, n_pade) + _deriv_h_pade(t, a, params, n_pade)


def phi_rheston_rational(u, tau, params, xi_curve, n_pade: int = 2, n_quad: int = 30):
    """
    Rational approximation for the rough Heston characteristic function
    E[exp(i * u * X_tau)] where X_tau = log(S_tau/S_0).
    """
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")

    tau = np.atleast_1d(np.asarray(tau))
    if n_quad > 0:
        # Gauss quadrature
        x_le, w_le = gauss_legendre(0, 1, n_quad)
        log_charfunc = (
            w_le[None, :]
            * xi_curve(tau[:, None] * (1 - x_le[None, :]))
            * _g(
                a=u,
                t=tau * x_le[None, :],
                params=params,
                n_pade=n_pade,
            )
        )
        log_charfunc = np.sum(log_charfunc, axis=1)
    else:
        # Using scipy.quad_vec() - This is longer.
        log_charfunc = integrate.quad_vec(
            lambda t: xi_curve(tau * (1 - t))
            * _g(a=u, t=tau * t, params=params, n_pade=n_pade),
            0,
            1,
            epsrel=1e-10,
            limit=1000,
        )[0]
    charfunc = np.exp(tau * log_charfunc)
    return charfunc


def impvol_rheston_rational(k, tau, params, xi, n_pade: int, n_quad: int = 40):
    """
    Calculate implied volatility in the rough Heston model using a rational
    approximation of the characteristic function.

    Parameters
    ----------
    k : float
        Log strike
    tau : float
        Time to maturity
    params : dict
        Model parameters
    xi : ndarray
        Volatility curve values
    n : int
        Order of rational approximation

    Returns
    -------
    float
        Black implied volatility
    """
    otm_price = lewis_formula_otm_price(
        lambda u, tau: phi_rheston_rational(
            u=u, tau=tau, params=params, xi_curve=xi, n_pade=n_pade, n_quad=n_quad
        ),
        k=k,
        tau=tau,
    )
    opttype = 2 * (k > 0) - 1  # otm options
    impvol = black_impvol(K=np.exp(k), T=tau, F=1, value=otm_price, opttype=opttype)
    return impvol


def gamma_kernel(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the gamma kernel at x.

    The gamma kernel is defined as
        k(x) = (nu/Gamma(alpha)) * x^(alpha - 1) * exp(-lbd * x)
    with alpha = H + 1/2
    """
    nu = params["nu"]
    H = params["H"]
    alpha = H + 1 / 2
    lbd = params["lbd"]
    return (nu / sp.gamma(alpha)) * x ** (alpha - 1) * np.exp(-lbd * x)


def integral_gamma_kernel_K0(x, params, quad_scipy=False):
    r"""
    Compute the integral K0(x) = \int_0^x k(s) ds where k(s) is the gamma kernel
    function at s.
    """
    if quad_scipy:
        return integrate.quad(lambda s: gamma_kernel(s, params), 0.0, x)[0]
    else:
        nu = params["nu"]
        H = params["H"]
        alpha = H + 1 / 2
        lbd = params["lbd"]
        x = np.atleast_1d(np.asarray(x))
        mask = x > 0
        res = np.empty_like(x)
        res[mask] = sp.gamma(alpha) * sp.gammainc(alpha, lbd * x[mask]) / lbd ** (alpha)
        res[~mask] = x[~mask] ** alpha / alpha
        res *= nu / sp.gamma(alpha)
        return res


def integral_gamma_kernel_K00(x, params, quad_scipy=False):
    r"""
    Compute the integral K00(x) = \int_0^x k(s)^2 ds where k(s) is the gamma kernel
    function at s.
    Note: in SciPy, gammainc is the regularized lower incomplete gamma function.
    """
    if quad_scipy:
        return integrate.quad(lambda s: gamma_kernel(s, params) ** 2, 0.0, x)[0]
    else:
        nu = params["nu"]
        H = params["H"]
        alpha = H + 1 / 2
        lbd = params["lbd"]
        x = np.atleast_1d(np.asarray(x))
        mask = x > 0
        res = np.empty_like(x)
        res[mask] = (
            sp.gamma(2.0 * H)
            * sp.gammainc(2.0 * H, 2 * lbd * x[mask])
            / (2.0 * lbd) ** (2.0 * H)
        )
        res[~mask] = x[~mask] ** (2.0 * H) / (2.0 * H)
        res *= (nu / sp.gamma(alpha)) ** 2
        return res


def integral_gamma_kernel_Ki(x, i, params, quad_scipy=True):
    r"""
    Compute the integral Kij(x) = \int_0^x k(s + i * x) * k(s + j * x) ds where k is the
    gamma kernel.
    """
    if quad_scipy:
        return integrate.quad(lambda s: gamma_kernel(s + i * x, params), 0.0, x)[0]


def integral_gamma_kernel_Kij(x, i, j, params, quad_scipy=True):
    r"""
    Compute the integral Kij(x) = \int_0^x k(s + i * x) * k(s + j * x) ds where k is the
    gamma kernel.
    """
    if quad_scipy:
        return integrate.quad(
            lambda s: gamma_kernel(s + i * x, params) * gamma_kernel(s + j * x, params),
            0.0,
            x,
        )[0]


def rho_u_chi(x, params):
    x = np.atleast_1d(np.asarray(x))
    if np.any(x <= 0):
        raise ValueError("Input x must be positive for rho_u_chi.")
    K0 = integral_gamma_kernel_K0(x, params)
    K00 = integral_gamma_kernel_K00(x, params)
    if np.any(K00 == 0):
        raise ValueError("K00(x) cannot be zero for any x in the input.")
    return K0 / (x * K00) ** 0.5


def rho_u_xi(x, params):
    x = np.atleast_1d(np.asarray(x))
    if np.any(x <= 0):
        raise ValueError("Input x must be positive for rho_u_chi.")
    K01 = integral_gamma_kernel_Kij(x, 0, 1, params)
    K00 = integral_gamma_kernel_K00(x, params)
    K11 = integral_gamma_kernel_Kij(x, 1, 1, params)
    if np.any(K00 == 0) or np.any(K11 == 0):
        raise ValueError("K00(x) or K11(x) cannot be zero for any x in the input.")
    return K01 / (K00 * K11) ** 0.5


def rho_xi_chi(x, params):
    x = np.atleast_1d(np.asarray(x))
    if np.any(x <= 0):
        raise ValueError("Input x must be positive for rho_u_chi.")
    K1 = integral_gamma_kernel_Ki(x, 1, params)
    K11 = integral_gamma_kernel_Kij(x, 1, 1, params)

    if np.any(K11 == 0):
        raise ValueError("K11(x) cannot be zero for any x in the input.")
    return K1 / (x * K11) ** 0.5


def simulate_rsqe(T, params, xi, n_mc, n_steps):
    """
    Simulate the Riemann-sum Quadratic Exponential scheme (RSQE).
    """
    H = params["H"]
    rho = params["rho"]
    rho2m1 = np.sqrt(1 - rho**2)
    eps_0 = 1e-10

    # Generate random variables
    W = np.random.normal(size=(n_steps, n_mc))
    Wperp = np.random.normal(size=(n_steps, n_mc))
    U = np.random.uniform(size=(n_steps, n_mc))

    # Time n_steps
    dt = T / n_steps
    tj = np.arange(1, n_steps + 1) * dt
    xi_tj = np.array([xi(t) for t in tj])

    # Initialize arrays
    K00_delta = float(integral_gamma_kernel_K00(dt, params))
    K00_j = np.zeros(n_steps + 1)
    K00_j[1:] = integral_gamma_kernel_K00(tj, params)
    bstar = np.sqrt(np.diff(K00_j) / dt)

    u = np.zeros((n_steps, n_mc))
    v = np.full(n_mc, xi(0))
    xihat = np.full(n_mc, xi_tj[0])
    x = np.zeros(n_mc)
    y = np.zeros(n_mc)
    w = np.zeros(n_mc)

    for j in range(n_steps):
        varv = (xihat + 2 * H * v) / (1 + 2 * H) * K00_delta
        psi = varv / xihat**2
        if np.any(psi < 0):
            raise ValueError("psi must be non-negative for all j and n_mc.")
        vf = np.where(
            psi < 1.5, psi_minus(psi, xihat, W[j]), psi_plus(psi, xihat, U[j])
        )

        u[j] = vf - xihat
        dw = (v + vf) / 2 * dt
        w += dw
        dy = u[j] / bstar[0]
        y += dy
        x += -dw / 2 + np.sqrt(dw) * rho2m1 * Wperp[j] + rho * dy

        if j < n_steps - 1:
            btilde = bstar[1 : j + 2][::-1]
            xihat = (
                xi_tj[j + 1] + np.tensordot(btilde, u[: j + 1, :], axes=1) / bstar[0]
            )

        xihat = np.maximum(xihat, eps_0)
        v = vf

    return {"x": x, "v": v, "w": w}


def simulate_hqe(T, params, xi, n_mc, n_steps):
    """
    Simulate the Hybrid Quadratic Exponential scheme (HQE).
    """
    H = params["H"]
    rho = params["rho"]
    rho2m1 = np.sqrt(1 - rho * rho)
    eps_0 = 1e-10

    # Generate random matrices
    W = np.random.normal(size=(n_steps, n_mc))
    Wperp = np.random.normal(size=(n_steps, n_mc))
    Z = np.random.normal(size=(n_steps, n_mc))
    U = np.random.uniform(size=(n_steps, n_mc))
    Uperp = np.random.uniform(size=(n_steps, n_mc))

    dt = T / n_steps
    tj = np.arange(1, n_steps + 1) * dt
    xij = xi(tj)

    K0_delta = integral_gamma_kernel_K0(dt, params)
    K_jj = np.array(
        [integral_gamma_kernel_Kij(dt, i, i, params) for i in range(n_steps)]
    )
    K00_delta = K_jj[0]

    bstar = np.sqrt(K_jj / dt)
    rho_vchi = K0_delta / np.sqrt(K00_delta * dt)
    beta_vchi = K0_delta / dt

    u = np.zeros((n_steps, n_mc))
    chi = np.zeros((n_steps, n_mc))
    v = np.full(n_mc, xi(0))
    xihat = np.full(n_mc, xij[0])
    x = np.zeros(n_mc)
    y = np.zeros(n_mc)
    w = np.zeros(n_mc)

    for j in range(n_steps):
        xibar = (xihat + 2 * H * v) / (1 + 2 * H)
        psi_chi = 4 * K00_delta * rho_vchi**2 * xibar / xihat**2
        psi_eps = 4 * K00_delta * (1 - rho_vchi**2) * xibar / xihat**2

        z_chi = np.where(
            psi_chi < 3 / 2,
            psi_minus(psi_chi, xihat / 2, W[j]),
            psi_plus(psi_chi, xihat / 2, U[j]),
        )
        z_eps = np.where(
            psi_eps < 3 / 2,
            psi_minus(psi_eps, xihat / 2, Wperp[j]),
            psi_plus(psi_eps, xihat / 2, Uperp[j]),
        )

        chi[j] = (z_chi - xihat / 2) / beta_vchi
        eps = z_eps - xihat / 2
        u[j] = beta_vchi * chi[j] + eps
        vf = xihat + u[j]
        vf = np.maximum(vf, eps_0)
        dw = (v + vf) / 2 * dt
        w += dw
        y += chi[j]
        x = x - dw / 2 + np.sqrt(dw) * rho2m1 * Z[j] + rho * chi[j]

        if j < n_steps - 1:
            btilde = bstar[1 : j + 2][::-1]
            xihat = xij[j + 1] + np.dot(btilde, chi[: j + 1])
        v = vf

    return {"x": x, "v": v, "w": w}
