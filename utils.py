import numpy as np
from scipy import optimize, stats
from scipy.integrate import quad, quad_vec

# Module-level constants
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

    # Set mid to NaN where the tolerance is not met
    mid = np.where(np.abs(diff) < TOL, mid, np.nan)
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


def lewis_formula_otm_price(phi, k, tau):
    """
    Compute the OTM (Out-of-The-Money) option price using the Lewis formula.

    The Lewis formula is used to price European options by applying Fourier transform
    methods. It calculates the price of OTM options given the characteristic function
    of the log price.

    Parameters
    ----------
    phi : callable
        The characteristic function of the log price process.
        Should take two arguments: complex number u and time to maturity tau.
    k : float or array_like
        Log strike price(s). k = log(K/S) where K is strike and S is spot price.
    tau : float or array_like
        Time(s) to maturity in years.

    Returns
    -------
    ndarray
        OTM option prices. For k < 0, returns put prices; for k >= 0, returns
        call prices.

    Notes
    -----
    The formula uses the following representation:
    For k >= 0 (calls): C(k,T) = 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du
    For k < 0 (puts): P(k,T) = e^k - 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du

    References
    ----------
    Lewis, A. L. (2000). Option Valuation under Stochastic Volatility.
    """
    k = np.atleast_1d(np.asarray(k))
    tau = np.atleast_1d(np.asarray(tau))

    def integrand(u):
        return np.real(np.exp(-1j * u * k) * phi(u - 1j / 2, tau) / (u**2 + 1 / 4))

    k_minus = k * (k < 0)

    integral, _ = quad_vec(integrand, 0, np.inf, epsrel=1e-10, limit=1000)
    result = np.exp(k_minus) - np.exp(k / 2) / np.pi * integral

    return result


def vix2_from_data(ivol_data):
    """
    Compute VIX squared values for each expiry from option implied volatility data.

    Parameters
    ----------
    ivol_data : pd.DataFrame
        DataFrame containing columns 'Texp', 'Bid', 'Ask', 'Fwd', and 'Strike'.

    Returns
    -------
    vix_expiries : np.ndarray
        Array of unique expiry times.
    vix2_values : np.ndarray
        Array of VIX squared values for each expiry.
    """
    # Get unique expiries
    vix_expiries = ivol_data["Texp"].unique()
    vix2_values = np.zeros_like(vix_expiries, dtype=float)

    for i, t in enumerate(vix_expiries):
        # Filter data for this expiry
        mask = ivol_data["Texp"] == t
        bid_vol = ivol_data.loc[mask, "Bid"].astype(float)
        ask_vol = ivol_data.loc[mask, "Ask"].astype(float)
        mid_vol = (bid_vol + ask_vol) / 2
        f = ivol_data.loc[mask, "Fwd"].iloc[0]
        k = np.log(ivol_data.loc[mask, "Strike"] / f)  # Log-strike

        # Handle missing values
        include = ~bid_vol.isna()
        k_min = k[include].min()
        k_max = k[include].max()

        # Interpolation inputs
        k_in = k[~mid_vol.isna()]
        vol_in = mid_vol[~mid_vol.isna()]

        def vol_interp(k_out):
            if k_out < k_min:
                return mid_vol[k == k_min].iloc[0]
            elif k_out > k_max:
                return mid_vol[k == k_max].iloc[0]
            else:
                return np.interp(k_out, k_in, vol_in)

        # Vectorize the interpolation function
        vix_vol = np.vectorize(vol_interp)

        # Define integration functions
        def c_tilde(y):
            return np.exp(y) * black_price(
                F=1, K=np.exp(y), T=t, vol=vix_vol(y), opttype=1
            )

        def p_tilde(y):
            return np.exp(y) * black_price(
                F=1, K=np.exp(y), T=t, vol=vix_vol(y), opttype=-1
            )

        # Perform numerical integration
        call_integral = quad(c_tilde, 0, 10)[0]
        put_integral = quad(p_tilde, -10, 0)[0]

        vix2_values[i] = f**2 * (1 + 2 * (call_integral + put_integral))

    vix2_values /= 10**4  # fwd price is in percentage.

    return vix_expiries, vix2_values
