import numpy as np


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
