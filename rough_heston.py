"""
Rough Heston model: characteristic function, kernel integrals, and simulation schemes.

Implements Padé-based characteristic function, gamma kernel integrals, and RSQE/HQE
path simulation.
`"""

import numpy as np
import scipy.special as sp
from scipy import integrate

import pade
from heston import psi_minus, psi_plus
from utils import black_impvol, gauss_legendre, lewis_formula_otm_price


def phi_rheston_rational(u, tau, params, xi_curve, n_pade: int = 2, n_quad: int = 30):
    """
    Padé-based rational approximation of the rough Heston characteristic function.

    Uses a Padé approximant for the integrand and Gauss-Legendre (or quad_vec)
    integration to compute E[exp(i u X_tau)] for given u and tau.

    Parameters
    ----------
    u : float or ndarray
        Fourier argument(s) for the characteristic function.
    tau : float or ndarray
        Time(s) to maturity.
    params : dict
        Model parameters for rough Heston (e.g., nu, H, lbd, rho).
    xi_curve : callable
        Forward variance curve function xi_curve(t).
    n_pade : int, optional
        Order of Padé approximation (default: 2).
    n_quad : int, optional
        Number of quadrature points for integration (default: 30).

    Returns
    -------
    ndarray
        Characteristic function values for each tau.
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
            * pade.g(
                a=u,
                t=tau * x_le[None, :],
                params=params,
                n_pade=n_pade,
            )
        )
        log_charfunc = np.sum(log_charfunc, axis=1)
    else:
        # Using scipy.quad_vec() is longer.
        log_charfunc = integrate.quad_vec(
            lambda t: xi_curve(tau * (1 - t))
            * pade.g(a=u, t=tau * t, params=params, n_pade=n_pade),
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
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")

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
    Compute the gamma kernel k(x) for the rough Heston model.

    The gamma kernel is defined as
        k(x) = (nu/Gamma(alpha)) * x^(alpha - 1) * exp(-lbd * x)
    with alpha = H + 1/2.

    Parameters
    ----------
    x : ndarray
        Input values.
    params : dict
        Model parameters (nu, H, lbd).

    Returns
    -------
    ndarray
        Gamma kernel values.
    """
    nu = params["nu"]
    H = params["H"]
    alpha = H + 1 / 2
    lbd = params["lbd"]
    return (nu / sp.gamma(alpha)) * x ** (alpha - 1) * np.exp(-lbd * x)


def integral_gamma_kernel_K0(x, params, quad_scipy=False):
    r"""
    Compute the integral K0(x) = \int_0^x k(s) ds where k is the gamma kernel.

    Parameters
    ----------
    x : float or ndarray
        Upper integration bound(s).
    params : dict
        Model parameters.
    quad_scipy : bool, optional
        Use scipy.quad for integration (default: False).

    Returns
    -------
    float or ndarray
        Value(s) of the integral.
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
    Compute the integral K00(x) = \int_0^x k(s)^2 ds where k is the gamma kernel
    function at s.
    Note: in SciPy, gammainc is the regularized lower incomplete gamma function.

    Parameters
    ----------
    x : float or ndarray
        Upper integration bound(s).
    params : dict
        Model parameters.
    quad_scipy : bool, optional
        Use scipy.quad for integration (default: False).

    Returns
    -------
    float or ndarray
        Value(s) of the integral.
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

    Parameters
    ----------
    x : float
        Upper integration bound.
    i : int
        Shift index.
    params : dict
        Model parameters.
    quad_scipy : bool, optional
        Use scipy.quad for integration (default: True).

    Returns
    -------
    float
        Value of the integral.
    """
    if quad_scipy:
        return integrate.quad(lambda s: gamma_kernel(s + i * x, params), 0.0, x)[0]


def integral_gamma_kernel_Kij(x, i, j, params, quad_scipy=True):
    r"""
    Compute the integral Kij(x) = \int_0^x k(s + i * x) * k(s + j * x) ds where k is the
    gamma kernel.

    Parameters
    ----------
    x : float
        Upper integration bound.
    i, j : int
        Shift indices.
    params : dict
        Model parameters.
    quad_scipy : bool, optional
        Use scipy.quad for integration (default: True).

    Returns
    -------
    float
        Value of the integral.
    """
    if quad_scipy:
        return integrate.quad(
            lambda s: gamma_kernel(s + i * x, params) * gamma_kernel(s + j * x, params),
            0.0,
            x,
        )[0]


def rho_u_chi(x, params):
    """
    Compute correlation coefficient rho_u_chi for the gamma kernel.

    Parameters
    ----------
    x : float or ndarray
        Input values.
    params : dict
        Model parameters.

    Returns
    -------
    float or ndarray
        Correlation coefficient(s).
    """
    x = np.atleast_1d(np.asarray(x))
    if np.any(x <= 0):
        raise ValueError("Input x must be positive for rho_u_chi.")
    K0 = integral_gamma_kernel_K0(x, params)
    K00 = integral_gamma_kernel_K00(x, params)
    if np.any(K00 == 0):
        raise ValueError("K00(x) cannot be zero for any x in the input.")
    return K0 / (x * K00) ** 0.5


def rho_u_xi(x, params):
    """
    Compute correlation coefficient rho_u_xi for the gamma kernel.

    Parameters
    ----------
    x : float or ndarray
        Input values.
    params : dict
        Model parameters.

    Returns
    -------
    float or ndarray
        Correlation coefficient(s).
    """
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
    """
    Compute correlation coefficient rho_xi_chi for the gamma kernel.

    Parameters
    ----------
    x : float or ndarray
        Input values.
    params : dict
        Model parameters.

    Returns
    -------
    float or ndarray
        Correlation coefficient(s).
    """
    x = np.atleast_1d(np.asarray(x))
    if np.any(x <= 0):
        raise ValueError("Input x must be positive for rho_u_chi.")
    K1 = integral_gamma_kernel_Ki(x, 1, params)
    K11 = integral_gamma_kernel_Kij(x, 1, 1, params)

    if np.any(K11 == 0):
        raise ValueError("K11(x) cannot be zero for any x in the input.")
    return K1 / (x * K11) ** 0.5


def simulate_rsqe(T: float, params: dict, xi, n_mc: int, n_steps: int, seed=None):
    """
    Simulate paths using the Riemann-sum Quadratic Exponential (RSQE) scheme.

    Parameters
    ----------
    T : float
        Time horizon.
    params : dict
        Model parameters.
    xi : callable
        Forward variance curve function.
    n_mc : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.

    Returns
    -------
    dict
        Simulated paths: keys 'x', 'v', 'w'.
    """
    if not n_steps > 0:
        raise ValueError("Number of steps must be positive.")
    if not n_mc > 0:
        raise ValueError("Number of Monte Carlo paths must be positive.")

    if seed is not None:
        np.random.seed(seed)

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


def simulate_hqe(T: float, params: dict, xi, n_mc: int, n_steps: int, seed=None):
    """
    Simulate paths using the Hybrid Quadratic Exponential (HQE) scheme.

    Parameters
    ----------
    T : float
        Time horizon.
    params : dict
        Model parameters.
    xi : callable
        Forward variance curve function.
    n_mc : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.

    Returns
    -------
    dict
        Simulated paths: keys 'x', 'v', 'w'.
    """
    if not n_steps > 0:
        raise ValueError("Number of steps must be positive.")
    if not n_mc > 0:
        raise ValueError("Number of Monte Carlo paths must be positive.")

    if seed is not None:
        np.random.seed(seed)

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
