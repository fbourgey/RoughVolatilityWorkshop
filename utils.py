import numpy as np


def m_q_moment_diff_logvol(sig, q, lag):
    """
    Compute the mean of the q-th power of absolute log-vol differences at given lag.

    Parameters
    ----------
    sig : _type_
        _description_
    q : int
        _description_
    lag : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lag_vec = np.atleast_1d(np.asarray(lag, dtype=int))
    log_sig = np.log(sig)
    return np.array(
        [np.mean(np.abs(log_sig[lag:] - log_sig[:-lag]) ** q) for lag in lag_vec]
    )


def linreg(x, y, intercept=True):
    """
    Perform linear regression on two variables.
    y = alpha + beta * x

    Parameters
    ----------
    x : np.ndarray
        The independent variable.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    Tuple[float, float]
        The intercept and slope of the regression line.
    """
    if intercept:
        c = np.cov(x, y)
        beta = c[0, 1] / c[0, 0]
        alpha = np.mean(y) - beta * np.mean(x)
        return alpha, beta
    else:
        beta = np.mean(x * y) / np.mean(x**2)
        return beta
