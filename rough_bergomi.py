"""
Rough Bergomi model implementation.

Compact implementation providing path simulation, VIX pricing and approximations,
and local/implied volatility estimators for the rough Bergomi model.
"""

from collections.abc import Callable

import numpy as np
from scipy import integrate, optimize, special, stats
from tqdm import tqdm

import utils


class RoughBergomi:
    """
    Implementation of the rough Bergomi model.

    Parameters
    ----------
    s0 : float
        Initial spot price (must be positive).
    xi0 : callable
        Forward variance curve function xi0(t), must return positive values for t >= 0.
    H : float
        Hurst parameter of the fractional Brownian motion (must be positive).
    eta : float
        Volatility of volatility parameter (must be positive).
    rho : float
        Correlation between the spot and the volatility processes (must be in [-1, 1]).

    Notes
    -----
    The model is described in:
    Bayer, C., Friz, P., & Gatheral, J. (2016). "Pricing under rough volatility".
    Quantitative Finance, 16(6), 887-904.
    """

    def __init__(
        self,
        s0: float,
        xi0: Callable[[np.ndarray], np.ndarray],
        H: float,
        eta: float,
        rho: float,
    ) -> None:
        """
        Initialize the rough Bergomi model.
        See class docstring for parameter definitions.
        """
        if H <= 0.0:
            raise ValueError("Hurst parameter H must be positive.")
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")
        if eta <= 0.0:
            raise ValueError("Volatility of volatility eta must be positive.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("Correlation rho must be in [-1, 1].")
        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")
        # Check positivity for a range of t >= 0
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

        self.s0 = s0
        self.xi0 = xi0
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.H = H
        self.eta = eta
        self.rho = rho
        self.delta_vix = 1.0 / 12.0

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

    def kernel(self, u, t):
        """
        Compute the rough Bergomi kernel function.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u > t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel: eta * sqrt(2H) * (u - t)^{H - 0.5}.
        """
        return self.eta * np.sqrt(2.0 * self.H) * (u - t) ** (self.H - 0.5)

    def cov_levy_fbm(self, u, v):
        r"""
        Compute the covariance matrix of Levy's fractional Brownian motion.

        It corresponds to:

        Cov(W_u^H, W_v^H) = \int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds

        where:

        W_u^H = \int_0^u (u-s)^{H-1/2} dW_s

        Parameters
        ----------
        u : np.ndarray or float
            First set of time points.
        v : np.ndarray or float
            Second set of time points.

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v).
        """
        u_max_v = np.maximum(u, v)
        u_min_v = np.minimum(u, v)
        cov = (
            u_min_v ** (self.H + 0.5)
            * u_max_v ** (self.H - 0.5)
            * special.hyp2f1(1.0, 0.5 - self.H, 1.5 + self.H, u_min_v / u_max_v)
        )
        return cov / (self.H + 0.5)

    def cholesky_cov_matrix(
        self, tab_t, return_cov: bool = False, conditioning: bool = False
    ):
        r"""
        Compute the lower-triangular Cholesky factor
        of the covariance matrix of the Gaussian vector (Y_{t_i}, W_{t_i})
        for 1 <= i <= n, where t_i are the timesteps in tab_t.

        Here, W is a standard Brownian motion and
        Y_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.
        conditioning : bool, optional
            If True, compute the conditional covariance (see Bergomi's book, Chapter
            8, Appendix A).
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        n_disc = tab_t.shape[0] - 1
        # repeat tab_t[1:] n_disc times as columns (shape: n_disc x n_disc)
        u = np.tile(tab_t[1:], (n_disc, 1)).T
        cov_y = 2.0 * self.H * self.cov_levy_fbm(u, u.T)
        cov_w = np.minimum(u, u.T)
        cov_yw = u ** (self.H + 0.5) - (u - cov_w) ** (self.H + 0.5)
        cov_yw *= np.sqrt(2.0 * self.H) / (self.H + 0.5)
        if not conditioning:
            cov_yw *= self.rho
        cov = np.block(
            [
                [cov_y, cov_yw],
                [cov_yw.T, cov_w],
            ]
        )
        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = utils.cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def covariance_levy_fbm_vix(self, T, u, v):
        r"""
        Compute the covariance matrix for the VIX integrals of Levy's fractional
        Brownian motion.

        Specifically, computes the covariance:
            Cov(Y_T^u, Y_T^v)
        where
            Y_T^u = sqrt(2H) * \int_0^T (u - s)^{H - 1/2} dW_s,
        for u, v >= T, and W is a standard Brownian motion.

        This is used for simulating the VIX under the rough Bergomi model.

        Parameters
        ----------
        T : float
            Lower limit of the integral (typically the VIX start time).
        u : np.ndarray
            First set of time points (must satisfy u >= T).
        v : np.ndarray
            Second set of time points (must satisfy v >= T).

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v), with the same shape as u and v.
        """

        def func(x, y):
            tmp1 = x ** (self.H + 0.5) * special.hyp2f1(
                0.5 - self.H, 0.5 + self.H, 1.5 + self.H, -x / (y - x)
            )
            tmp2 = (x - T) ** (self.H + 0.5) * special.hyp2f1(
                0.5 - self.H, 0.5 + self.H, 1.5 + self.H, -(x - T) / (y - x)
            )
            return (
                (y - x) ** (self.H - 0.5)
                * (tmp1 - tmp2)
                * 2.0
                * self.H
                / (self.H + 0.5)
            )

        cov = np.zeros_like(u)
        cov[u == v] = u[u == v] ** (2.0 * self.H) - (u[u == v] - T) ** (2.0 * self.H)
        cov[u < v] = func(u[u < v], v[u < v])
        cov[u > v] = func(v[u > v], u[u > v])
        return np.real(cov)

    def cholesky_cov_matrix_vix(
        self, T: float, n_disc: int, return_cov: bool = False
    ) -> np.ndarray:
        r"""
        Compute the lower-triangular Cholesky factor of the covariance matrix of the
        Gaussian vector (Y_{T}^{u_i}) for 0 <= i <= n, where u_i = T + delta * i / n
        and delta = 30 / 365 (30 days in years).

        Here, W is a standard Brownian motion and
        Y_T^u = \sqrt{2H} \int_0^T (u-s)^{H-1/2} dW_s, for any u >= T.

        Parameters
        ----------
        T : float
            Maturity of interest.
        n_disc : int
            Number of time discretization steps (must be positive).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        u = np.tile(tab_u, (n_disc + 1, 1)).T
        v = u.T

        cov = self.covariance_levy_fbm_vix(T=T, u=u, v=v)

        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = utils.cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def simulate(
        self, tab_t, n_mc: int, n_loop: int = 1, seed=None, conditioning: bool = False
    ):
        """
        Simulate sample paths of the Rough Bergomi model.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        n_mc : int
            Total number of Monte Carlo paths to simulate.
        n_loop : int, optional
            Number of loops to split the simulation into (for memory efficiency).
            Must divide n_mc exactly. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. See Bergomi's book, Chapter 8, Appendix A.

        Returns
        -------
        dict
            Dictionary with the following keys:
                - 'v': np.ndarray, simulated variance paths
                    (shape: n_steps + 1, n_mc_loop)
                - 'y': np.ndarray, simulated log-volatility factor
                    (shape: n_steps + 1, n_mc_loop)
                - 'w': np.ndarray, simulated Brownian motion paths
                    (shape: n_steps + 1, n_mc_loop)
                - 'int_v_dt': np.ndarray, time-integrated variance for each path
                    (shape: n_mc,)
                - 'int_sqrt_v_dw': np.ndarray, stochastic integral for each path
                    (shape: n_mc,)

        Notes
        -----
        For H=0.5, the model reduces to SABR. For H<0.5, the full rough Bergomi model is
        simulated.
        """
        if seed is not None:
            np.random.seed(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        tab_v = np.zeros((n_disc + 1, n_mc))
        tab_w = np.zeros((n_disc + 1, n_mc))
        tab_y = np.zeros((n_disc + 1, n_mc))
        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        for i in range(n_loop):
            # print("iteration", i + 1, "of", n_loop)
            normal = np.random.randn(2 * n_disc, n_mc_loop)

            if self.H == 0.5:
                # The rough Bergomi model reduces to a SABR model.
                # The spot variance simply writes as
                # v_t = xi_0^t * exp(eta * W_t - 0.5 * eta^2 * t)
                y = dt**0.5 * np.cumsum(normal[:n_disc, :], axis=0)
                y = np.insert(y, 0, 0.0, axis=0)
                if conditioning:
                    w = y
                else:
                    w_perp = dt**0.5 * np.cumsum(normal[n_disc:, :], axis=0)
                    w_perp = np.insert(w_perp, 0, 0.0, axis=0)
                    w = self.rho * y + np.sqrt(1.0 - self.rho**2) * w_perp
                    w_perp = None
            else:
                normal = np.matmul(
                    self.cholesky_cov_matrix(tab_t, conditioning=conditioning), normal
                )
                y = normal[:n_disc, :]
                y = np.insert(y, 0, 0.0, axis=0)
                w = normal[n_disc:, :]
                w = np.insert(w, 0, 0.0, axis=0)

            normal = None  # free memory

            # spot variance process
            v = self.xi0(tab_t[:, None]) * np.exp(
                self.eta * y - 0.5 * self.eta**2 * tab_t[:, None] ** (2.0 * self.H)
            )
            int_sqrt_v_dw[i * n_mc_loop : (i + 1) * n_mc_loop] = np.sum(
                np.sqrt(v[:-1, :]) * (w[1:, :] - w[:-1, :]), axis=0
            )
            # trapezoidal rule for the integral
            int_v_dt[i * n_mc_loop : (i + 1) * n_mc_loop] = (
                0.5 * dt * np.sum(v[:-1, :] + v[1:, :], axis=0)
            )
            tab_v[:, i * n_mc_loop : (i + 1) * n_mc_loop] = v
            tab_y[:, i * n_mc_loop : (i + 1) * n_mc_loop] = y
            tab_w[:, i * n_mc_loop : (i + 1) * n_mc_loop] = w

        return {
            "v": tab_v,
            "y": tab_y,
            "w": tab_w,
            "int_v_dt": int_v_dt,
            "int_sqrt_v_dw": int_sqrt_v_dw,
        }

    def local_vol_from_paths(
        self, k, v_t, int_v_dt, int_sqrt_v_dw, return_skew: bool = False
    ):
        r"""
        Estimate the local volatility and (optionally) its skew using Monte Carlo
        simulation.

        This method computes the local volatility at a given log-moneyness and, if
        requested, the slope (skew) of the local volatility surface, based on a ratio of
        expectations as described in Bourgey et al., "Local volatility under rough
        volatility".

        Parameters
        ----------
        k : float or np.ndarray
            The log-moneyness (typically 0 for ATM).
        v_t : np.ndarray
            Array of instantaneous variances at time `t` (shape: n_samples,).
        int_v_dt : np.ndarray
            Array of time-integrated variances (shape: n_samples,).
        int_sqrt_v_dw : np.ndarray
            Array of stochastic integrals ∫ sqrt(v) dW (shape: n_samples,).
        return_skew : bool, optional
            If True, also return the estimated local volatility skew. Default is False.

        Returns
        -------
        float or tuple
            Estimated local volatility. If `return_skew` is True, returns a tuple
            (local_vol, local_vol_skew).

        References
        ----------
        Bourgey, F., et al. "Local volatility under rough volatility", Equations (25)
        and (26).
        """
        k = np.atleast_1d(k)
        k = k.reshape(-1, 1)
        int_v_dt = int_v_dt.reshape(1, -1)
        int_sqrt_v_dw = int_sqrt_v_dw.reshape(1, -1)
        v_t = v_t.reshape(1, -1)

        u = k + 0.5 * int_v_dt - self.rho * int_sqrt_v_dw
        weight = np.exp(-(u**2) / (2.0 * (1.0 - self.rho**2) * int_v_dt))
        weight /= int_v_dt**0.5
        exp_w = np.mean(weight, axis=1)
        exp_vw = np.mean(v_t * weight, axis=1)
        lv = np.sqrt(exp_vw / exp_w)

        if return_skew:
            lv_skew = exp_vw * np.mean(u * weight / int_v_dt, axis=1)
            lv_skew -= exp_w * np.mean(u * weight * v_t / int_v_dt, axis=1)
            lv_skew /= 2.0 * (1.0 - self.rho**2) * exp_vw**0.5 * exp_w**1.5

            return lv, lv_skew

        return lv

    def local_vol(
        self,
        k,
        T,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        return_skew: bool = False,
    ):
        """
        Estimate the local volatility (and optionally its skew) at a given log-moneyness
        and maturity using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        return_skew : bool, optional
            If True, also return the estimated local volatility skew. Default is False.

        Returns
        -------
        float or tuple
            Estimated local volatility. If `return_skew` is True, returns a tuple
            (local_vol, local_vol_skew).
        """
        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=True
        )
        return self.local_vol_from_paths(
            k=k,
            v_t=paths["v"][-1, :],
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            return_skew=return_skew,
        )

    def implied_vol(
        self,
        k,
        T: float,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        conditioning: bool = False,
        return_skew: bool = False,
    ):
        """
        Estimate the implied volatility (and optionally its skew) at a given
        log-moneyness and maturity using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. Default is False.
        return_skew : bool, optional
            If True, also return the estimated implied volatility skew. Default is
            False.

        Returns
        -------
        float or tuple
            Estimated implied volatility. If `return_skew` is True, returns a tuple
            (implied_vol, implied_vol_skew).
        """
        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=conditioning
        )
        return self.implied_vol_from_paths(
            k=k,
            T=T,
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            conditioning=conditioning,
            return_skew=return_skew,
        )

    def implied_vol_from_paths(
        self,
        T: float,
        int_v_dt: np.ndarray,
        int_sqrt_v_dw: np.ndarray,
        k: float | np.ndarray = 0.0,
        conditioning: bool = False,
        return_skew: bool = False,
    ):
        r"""
        Estimate the implied volatility (and optionally its skew) from simulated paths.

        Parameters
        ----------
        T : float
            Maturity.
        int_v_dt : np.ndarray
            Array of time-integrated variances (shape: n_samples,).
        int_sqrt_v_dw : np.ndarray
            Array of stochastic integrals \int sqrt(v) dW (shape: n_samples,).
        k : float or np.ndarray, optional
            Log-moneyness (default is 0.0 for ATM).
        conditioning : bool, optional
            If True, use the conditional law given the Brownian motion driving the
            spot. Default is False.
        return_skew : bool, optional
            If True, also return the estimated implied volatility skew.
            Default is False.

        Returns
        -------
        float or tuple
            Estimated implied volatility. If `return_skew` is True, returns a
            tuple (implied_vol, implied_vol_skew).
        """
        int_v_dt = int_v_dt.flatten()
        int_sqrt_v_dw = int_sqrt_v_dw.flatten()
        k = np.atleast_1d(k)
        S = self.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S.mean()
        K = F * np.exp(k)

        if conditioning:
            s0_cond = self.s0 * np.exp(
                -0.5 * self.rho**2 * int_v_dt + self.rho * int_sqrt_v_dw
            )
            vol_cond = np.sqrt((1.0 - self.rho**2) * int_v_dt / T)
            F_cond = np.mean(s0_cond)
            opttype = 2 * (self.s0 <= K) - 1
            impvol = np.zeros_like(K)
            if return_skew:
                digit = np.zeros_like(K)
            for i, K_i in enumerate(K):
                price_cond_i = utils.black_price(
                    K=K_i, T=T, F=s0_cond, vol=vol_cond, opttype=opttype[i]
                ).mean()
                impvol[i] = utils.black_impvol(
                    K=K_i, T=T, F=F_cond, value=price_cond_i, opttype=opttype[i]
                )
                if return_skew:
                    w_cond = vol_cond * T**0.5
                    d2_cond_i = np.log(s0_cond / K_i) / w_cond - 0.5 * w_cond
                    digit[i] = stats.norm.cdf(d2_cond_i).mean()
        else:
            impvol = utils.black_otm_impvol_mc(S=S, k=k, T=T)
            if return_skew:
                digit = np.mean((S >= K) * 1.0)

        if return_skew:
            w = impvol * T**0.5
            d2 = -k / w - 0.5 * w
            skew = stats.norm.cdf(d2) - digit
            skew /= stats.norm.pdf(d2) * T**0.5
            return impvol, skew

        return impvol

    def atm_implied_vol_local_vol_skew(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int = 1,
        seed: int | None = None,
        n_disc_split: int = 50,
    ):
        """
        Compute the ATM implied volatility skew and local volatility skew
        as a function of maturity.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        n_mc : int
            Number of Monte Carlo paths.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        n_disc_split : int, optional
            Number of time discretization in each split. Must divide the total number of
            time steps (n_steps) exactly. Default is 50.

        Returns
        -------
        tuple
            (tab_t_split, (atm_impvol, atm_impvol_skew), (atm_lv, atm_lv_skew)), where
            each is a np.ndarray of length n_split.
        """
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=True
        )
        v = paths["v"]
        w = paths["w"]
        paths = None
        dt = tab_t[1] - tab_t[0]
        n_disc = tab_t.shape[0] - 1

        int_sqrt_v_dw_cumsum = np.cumsum(
            np.sqrt(v[:-1, :]) * np.diff(w, axis=0), axis=0
        )
        int_v_dt_cumsum = 0.5 * dt * np.cumsum(v[:-1, :] + v[1:, :], axis=0)
        int_sqrt_v_dw_cumsum = np.insert(int_sqrt_v_dw_cumsum, 0, 0.0, axis=0)
        int_v_dt_cumsum = np.insert(int_v_dt_cumsum, 0, 0.0, axis=0)

        n_split, remainder = divmod(n_disc, n_disc_split)
        if remainder != 0:
            raise ValueError("n_disc must be divisible by n_disc_split for splitting.")

        int_sqrt_v_dw_split = np.array(
            [t[-1] for t in np.split(int_sqrt_v_dw_cumsum[1:], n_split, axis=0)]
        )
        int_v_dt_split = np.array(
            [t[-1] for t in np.split(int_v_dt_cumsum[1:], n_split, axis=0)]
        )
        v_split = np.array([t[-1] for t in np.split(v[1:, :], n_split, axis=0)])
        tab_t_split = np.array([t[-1] for t in np.split(tab_t[1:], n_split)])

        atm_impvol = np.zeros(n_split)
        atm_impvol_skew = np.zeros(n_split)
        atm_lv = np.zeros(n_split)
        atm_lv_skew = np.zeros(n_split)

        for i in range(n_split):
            atm_impvol[i], atm_impvol_skew[i] = self.implied_vol_from_paths(
                T=tab_t_split[i],
                int_v_dt=int_v_dt_split[i, :],
                int_sqrt_v_dw=int_sqrt_v_dw_split[i, :],
                conditioning=True,
                return_skew=True,
            )
            atm_lv[i], atm_lv_skew[i] = self.local_vol_from_paths(
                k=0.0,
                v_t=v_split[i, :],
                int_v_dt=int_v_dt_split[i, :],
                int_sqrt_v_dw=int_sqrt_v_dw_split[i, :],
                return_skew=True,
            )
        return tab_t_split, (atm_impvol, atm_impvol_skew), (atm_lv, atm_lv_skew)

    def rate_function(self, y: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the rate function minimizing path coefficients for the
        large deviations principle.

        Parameters
        ----------
        y : np.ndarray
            Array of y values (log-moneyness or displacement).
        N : int
            Number of Fourier basis functions for the Ritz projection.

        Returns
        -------
        tuple
            tab_a : np.ndarray
                Array of optimal coefficients (shape: len(y), N).
            tab_rate : np.ndarray
                Array of rate function values (shape: len(y),).
        """

        tab_rate = np.zeros_like(y)
        tab_a = np.zeros((np.shape(y)[0], N))
        sigma_0 = self.xi0_0**0.5
        for i, y_i in enumerate(y):
            a_guess = np.zeros(N)
            a_guess[
                0
            ] = y_i * self.rho / sigma_0 + y_i**2 / 2 * self.eta / sigma_0**2 * np.sqrt(
                2 * self.H
            ) / (
                self.H + 0.5
            ) * (
                (self.rho**2 + 1) * 0.65 - 3.0 * self.rho**2 / (self.H + 1.5)
            )
            optim = optimize.minimize(
                lambda a, y_i=y_i: self.objective_function_rate_function(a=a, y=y_i),
                x0=a_guess,
            )
            tab_a[i, :] = optim.x
            tab_rate[i] = optim.fun

        return tab_a, tab_rate

    def sigma_path_rate_function(self, y: np.ndarray, n_trunc: int) -> np.ndarray:
        r"""
        Compute sigma(\hat{h}_1^y), where h^y is the Cameron-Martin path minimizing
        the rate function for a given y.

        The path h^y is represented in a truncated Fourier basis:
            h_t = \sum_{k=0}^{N-1} a_k e_k(t)
        where the coefficients a_k are obtained by minimizing the rate function.

        Here,
            \hat{h}_t^y = \int_0^t kernel_rb(t, s) h_s ds
        and
            sigma(x) = sqrt(xi0^0) * exp(0.5 * eta * x)

        Parameters
        ----------
        y : np.ndarray
            Array of y values (log-moneyness).
        n_trunc : int
            Number of Fourier basis functions used in the truncation.

        Returns
        -------
        np.ndarray
            Array of sigma(\hat{h}_1^y) values, one for each y.
        """
        if n_trunc <= 0:
            raise ValueError("n_trunc must be a positive integer.")
        y = np.atleast_1d(y)
        a, _ = self.rate_function(y, n_trunc)
        x = np.array(
            [
                np.sum(
                    a[i, :] * self.kernel_fourier_integrals(n=n_trunc, t=1.0),
                )
                for i in range(y.shape[0])
            ]
        )
        return self.xi0_0**0.5 * np.exp(0.5 * self.eta * x)

    def kernel_fourier_integrals(self, n, t):
        r"""
        Compute the integrals of the rough Bergomi kernel times the Fourier basis
        functions.

        For each k in 0,...,n-1, computes:
            \int_0^t sqrt(2H) * (t-s)^{H-1/2} * e_k(s) ds,
        where e_k(s) is the k-th Fourier basis function.

        Parameters
        ----------
        n : int
            Number of Fourier basis functions.
        t : float
            Upper limit of integration.

        Returns
        -------
        np.ndarray
            Array of shape (n,) with the value of each integral for k = 0,...,n-1.
        """
        alpha = self.H + 0.5
        tab = np.zeros(n)
        tab[0] = np.sqrt(2.0 * self.H)
        for m in range(1, n):
            w_m = 2.0 * np.pi * m
            if m % 2 == 0.0:
                tab[m] = np.real(special.hyp1f1(1.0, alpha + 1.0, 1j * w_m * t))
            else:
                tab[m] = np.imag(special.hyp1f1(1.0, alpha + 1, 1j * w_m * t))

        return tab * t**alpha / alpha

    def objective_function_rate_function(self, a, y):
        """
        Compute the rate function objective for the large deviations principle in
        rough volatility models.

        Parameters
        ----------
        a : np.ndarray
            Array of Fourier coefficients (shape: n,).
        y : float
            Target value (e.g., log-moneyness).

        Returns
        -------
        float
            Value of the objective function.
        """
        n = np.shape(a)[0]

        def h_fourier(t):
            r"""
            Compute the Fourier series expansion
            h_fourier(a, t) = \sum_{k=0}^{n-1} a_k * e_k(t)
            """
            return np.sum(a * utils.fourier(n=n, t=t))

        def h_hat(t):
            r"""
            Compute \int_0^t kernel_rb(t, s, H) * h_prime(a, s) ds
            """

            return np.sum(a * self.kernel_fourier_integrals(n=n, t=t))

        def sigma(x):
            return self.xi0_0**0.5 * np.exp(0.5 * self.eta * x)

        norm_h_fourier_squared = np.sum(a**2)
        F = integrate.quad(lambda x: sigma(h_hat(x)) ** 2, 0, 1)[0]
        G = integrate.quad(lambda x: sigma(h_hat(x)) * h_fourier(x), 0, 1)[0]

        return (y - self.rho * G) ** 2 / (
            2.0 * (1.0 - self.rho**2) * F
        ) + 0.5 * norm_h_fourier_squared

    def fut_vix2(self, T: float) -> float:
        r"""
        Compute the fair value of a VIX squared futures contract at maturity T. It
        corresponds to:

            E[VIX_T^2] = 1/delta \int_{T}^{T+delta} \xi_0^u du

        where delta is the VIX window (30/365 years by default) and xi0(u) is the
        forward variance curve.

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).

        Returns
        -------
        float
            Fair value of the VIX squared futures contract at time T.

        Raises
        ------
        ValueError
            If T is negative.

        Notes
        -----
        This is a model-free quantity, depending only on the forward variance curve.
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")

        integral, _ = integrate.quad(lambda u: self.xi0(u), T, T + self.delta_vix)
        return integral / self.delta_vix

    def simulate_vix(
        self,
        T: float,
        n_mc: int,
        n_disc: int,
        rule="trap",
        seed=None,
        control_variate: bool = False,
        lbd=None,
        eta_2=None,
        return_xi: bool = False,
    ):
        """
        Simulate sample paths of the VIX.

        Parameters
        ----------
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'left'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance of the
            simulation. Default is False.
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        eta_2 : float or None, optional
            If provided, use a mixed model with two different eta values.
            This is the second eta value. Must be provided if `lbd` is not None.
        return_xi : bool, optional
            If True, return the simulated xi values instead of VIX values.
            Default is False.
        Returns
        -------
        np.ndarray
            Simulated VIX values (shape: n_mc,) or simulated xi values if
            `return_xi` is True (shape: n_disc + 1, n_mc).
        """
        if seed is not None:
            np.random.seed(seed)

        if rule not in ["left", "right", "trap"]:
            raise ValueError("rule must be one of 'left', 'right', or 'trap'.")

        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        n_disc = tab_u.shape[0] - 1
        y = self.cholesky_cov_matrix_vix(T, n_disc) @ np.random.randn(n_disc + 1, n_mc)
        var_y = tab_u ** (2.0 * self.H) - (tab_u - T) ** (2.0 * self.H)

        is_mixed = lbd is not None and eta_2 is not None
        if is_mixed:
            exp_1 = np.exp(self.eta * y - 0.5 * self.eta**2 * var_y[:, None])
            exp_2 = np.exp(eta_2 * y - 0.5 * eta_2**2 * var_y[:, None])
            xi = self.xi0(tab_u[:, None]) * (lbd * exp_1 + (1.0 - lbd) * exp_2)
        else:
            xi = self.xi0(tab_u[:, None]) * np.exp(
                self.eta * y - 0.5 * self.eta**2 * var_y[:, None]
            )

        if return_xi:
            return xi

        if control_variate:
            if is_mixed:
                raise ValueError("Control variate not implemented for the mixed case.")
            log_cv = (
                np.log(self.xi0(tab_u[:, None]))
                + self.eta * y
                - 0.5 * self.eta**2 * var_y[:, None]
            )

        if rule == "left":
            vix = np.sqrt(xi[:-1, :].mean(axis=0))
            if control_variate:
                cv = np.exp(log_cv[:-1, :].mean(axis=0)) ** 0.5

        elif rule == "right":
            vix = np.sqrt(xi[1:, :].mean(axis=0))
            if control_variate:
                cv = np.exp(log_cv[1:, :].mean(axis=0)) ** 0.5

        else:
            vix = np.sqrt(0.5 * (xi[:-1, :].mean(axis=0) + xi[1:, :].mean(axis=0)))
            if control_variate:
                cv = np.exp(log_cv[:-1, :].mean(axis=0)) ** 0.5
            # if control_variate:
            #     cv_left = np.exp(log_cv[:-1, :].sum(axis=0) / n_disc) ** 0.5
            #     cv_right = np.exp(log_cv[1:, :].sum(axis=0) / n_disc) ** 0.5
            #     cv = (cv_left, cv_right)

        return vix if not control_variate else (vix, cv)

    def implied_vol_vix(
        self, k, T, n_mc, n_disc, rule="trap", seed=None, lbd=None, eta_2=None
    ) -> np.ndarray:
        """
        Compute the implied volatility of a VIX option at a given log-moneyness
        using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness of the VIX option (typically 0 for ATM). Can be a scalar
            or array.
        T : float
            Maturity of the VIX option.
        n_mc : int
            Number of Monte Carlo simulation paths.
        n_disc : int
            Number of time discretization steps for the VIX simulation.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        np.ndarray
            Implied volatility values for the VIX option(s) at the specified
            log-moneyness.
        """
        vix = self.simulate_vix(
            T=T, n_mc=n_mc, n_disc=n_disc, rule=rule, seed=seed, lbd=lbd, eta_2=eta_2
        )
        return utils.black_otm_impvol_mc(S=vix, k=k, T=T)

    def price_vix(
        self,
        T: float,
        n_mc: int,
        n_disc: int,
        k: float | np.ndarray = 0.0,
        rule: str = "trap",
        seed: int | None = None,
        opttype: float | np.ndarray = 1.0,
        lbd: float | None = None,
        eta_2: float | None = None,
        return_opt: str = "price",
        control_variate: bool = False,
    ):
        """
        Compute a VIX option price at a given log-moneyness by Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        k : float or np.ndarray, optional
            Log-moneyness of the VIX option (typically 0 for ATM). Can be a scalar
            or array. Default is 0.0 (ATM).
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        opttype : float or np.ndarray, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        eta_2 : float or None, optional
            If provided, use a mixed model with two different eta values.
            This is the second eta value. Must be provided if `lbd` is not None.
        return_opt : str, optional
            Specifies what to return:
            - 'price': return the option price.
            - 'fut': return the VIX future price.
            - 'both': return both the option price and the VIX future price.
            Default is 'price'.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance. Default is False.

        Returns
        -------
        float or np.ndarray
            Estimated VIX option price.
        """
        if return_opt not in ["price", "fut", "both"]:
            raise ValueError("return_opt must be either 'price' or 'fut' or 'both'.")

        vix = self.simulate_vix(
            T=T,
            n_mc=n_mc,
            n_disc=n_disc,
            rule=rule,
            seed=seed,
            lbd=lbd,
            eta_2=eta_2,
            control_variate=control_variate,
        )
        if control_variate:
            vix_mc, vix_mc_cv = vix
            F = np.mean(vix_mc - vix_mc_cv) + self.price_vix_control_variate(
                T=T, n_disc=n_disc, rule=rule, return_fut=True
            )
        else:
            vix_mc = np.asarray(vix)
            F = np.mean(vix_mc)

        if return_opt == "fut":
            return F

        k = np.atleast_1d(np.asarray(k))
        opttype = np.atleast_1d(np.asarray(opttype))
        K = F * np.exp(k)
        payoff = np.maximum(opttype[None, :] * (vix_mc[:, None] - K[None, :]), 0.0)
        if control_variate:
            payoff_cv = np.maximum(
                opttype[None, :] * (vix_mc_cv[:, None] - K[None, :]), 0.0
            )
            price = np.mean(
                payoff - payoff_cv, axis=0
            ) + self.price_vix_control_variate(T=T, k=k, n_disc=n_disc, rule=rule)
        else:
            price = np.mean(payoff, axis=0)

        return price if return_opt == "price" else (price, F)

    def price_vix_fut(
        self,
        T,
        n_mc,
        n_disc,
        rule="trap",
        seed=None,
        control_variate: bool = False,
        lbd=None,
        eta_2=None,
    ):
        """
        Estimate the price of a VIX futures contract at maturity T using Monte Carlo
        simulation.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_mc : int
            Number of Monte Carlo simulation paths.
        n_disc : int
            Number of time discretization steps for the VIX simulation.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance.
            Default is False.

        Returns
        -------
        float
            Estimated VIX futures price at maturity T.
        """
        return self.price_vix(
            T=T,
            n_mc=n_mc,
            n_disc=n_disc,
            rule=rule,
            seed=seed,
            lbd=lbd,
            eta_2=eta_2,
            return_opt="fut",
            control_variate=control_variate,
        )

    def price_vix_control_variate(
        self,
        T: float,
        n_disc: int,
        k: float | np.ndarray = 0.0,
        rule: str = "left",
        opttype: float | np.ndarray = 1.0,
        return_fut: bool = False,
    ):
        """
        Price a VIX option using a control variate.

        This method computes the price of a VIX option with maturity `T` and strike `K`
        using a control variate technique to reduce variance in the Monte Carlo
        estimator. The control variate is typically chosen as an analytically tractable
        approximation or proxy for the VIX payoff.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float or np.ndarray, optional
            Log-moneyness (typically 0 for ATM).
        n_disc : int
            Number of time discretization steps for the simulation.
        rule : str, optional
            Numerical integration rule for the VIX calculation ('left', 'trap', etc.).
            Default is 'left'.
        opttype : int, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        return_fut : bool, optional
            If True, return the proxy for the VIX future instead of the option price.

        Returns
        -------
        float
            Estimated price of the VIX option using the control variate method.
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")
        if rule not in ["left", "right", "trap"]:
            raise ValueError("rule must be one of 'left', 'right', or 'trap'.")
        if opttype not in [-1, 1]:
            raise ValueError("opttype must be either -1 (put) or 1 (call).")
        n_disc = int(n_disc)
        if n_disc <= 0:
            raise ValueError("n_disc must be a positive integer.")

        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        cov_matrix = self.cholesky_cov_matrix_vix(T, n_disc, return_cov=True)
        cov_matrix *= self.eta**2
        mean_vec = np.log(self.xi0(tab_u)) - 0.5 * self.eta**2 * (
            tab_u ** (2.0 * self.H) - (tab_u - T) ** (2.0 * self.H)
        )
        if rule == "left":
            mean = np.mean(mean_vec[:-1])
            std = (np.sum(cov_matrix[:-1, :-1]) / n_disc**2) ** 0.5
        elif rule == "right":
            mean = np.mean(mean_vec[1:])
            std = (np.sum(cov_matrix[1:, 1:]) / n_disc**2) ** 0.5
        else:
            mean = np.mean(mean_vec)
            std = np.mean(cov_matrix.flatten()) ** 0.5

        F_cv = np.exp(0.5 * mean + 0.5 * (0.5 * std) ** 2)

        if return_fut:
            return F_cv

        k = np.atleast_1d(np.asarray(k))
        opttype = np.atleast_1d(np.asarray(opttype))
        K = F_cv * np.exp(k)

        return utils.black_price(
            K=K, T=T, F=F_cv, vol=0.5 * std / T**0.5, opttype=opttype
        )

    def limit_strong_error_vix_right(self, n, T):
        """
        Compute the asymptotic L2 strong error for the VIX option price using the right
        Riemann sum approximation.

        This function calculates the leading-order term of the strong error (in L2 norm)
        for the VIX option price when the VIX is discretized using the right Riemann
        sum, in the limit as the number of discretization steps n goes to infinity.
        The formula is valid only when the forward variance curve xi0 is flat (i.e.,
        constant).

        Parameters
        ----------
        n : int
            Number of time discretization steps (must be positive).
        T : float
            Maturity of the VIX option (must be positive).

        Returns
        -------
        float
            Leading-order term of the L2 strong error divided by n.

        Raises
        ------
        ValueError
            If xi0_flat is False (i.e., the forward variance curve is not flat).
        """
        if not self.xi0_flat:
            raise ValueError("xi0_flat must be set to True for limit strong error.")

        integral = integrate.quad(
            lambda t: t ** (self.H - 0.5) * (self.delta_vix + t) ** (self.H - 0.5), 0, T
        )[0]
        lim_strong_error = (
            np.exp(self.eta**2 * T ** (2.0 * self.H))
            + np.exp(
                self.eta**2
                * (
                    (T + self.delta_vix) ** (2.0 * self.H)
                    - self.delta_vix ** (2.0 * self.H)
                )
            )
            - 2.0 * np.exp(self.eta**2 * 2.0 * self.H * integral)
        )
        coef_lim = 0.5 * self.xi0_0 * lim_strong_error**0.5
        return coef_lim / n

    def compute_mse_atm_price(
        self,
        T: float,
        n_mse: int,
        true_price: float,
        true_fut: float,
        eps: float,
        opt="mc",
        rule: str = "trap",
        seed: int | None = None,
        n_disc_mlmc_0: int = 1,
    ):
        """
        Compute the mean squared error (MSE) and computational cost for ATM VIX call
        option pricing using Monte Carlo (MC) or Multilevel Monte Carlo (MLMC) methods.

        This method estimates the MSE of the ATM (at-the-money) VIX option price for a
        given target accuracy `eps`, comparing the estimated price to a provided
        reference price (`true_price`). It supports both standard Monte Carlo ("mc")
        and Multilevel Monte Carlo ("mlmc") approaches, and returns the empirical MSE
        and the total computational cost.

        Parameters
        ----------
        T : float
            Maturity of the VIX option (must be positive).
        n_mse : int
            Number of independent MSE replications (must be positive).
        true_price : float
            Reference (true) price of the ATM VIX option, used to compute the MSE.
        true_fut : float
            Reference (true) price of the ATM VIX future, used as the strike in the
            payoff.
        eps : float
            Target accuracy (root mean squared error) for the estimator (must be
            positive).
        opt : str, optional
            Estimation method: 'mc' for standard Monte Carlo, 'mlmc' for Multilevel
            Monte Carlo. Default is 'mc'.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        n_disc_mlmc_0 : int, optional
            Number of time discretization steps for the coarsest MLMC level. Default is
            1.

        Returns
        -------
        tuple
            mse_mc : float
                Empirical mean squared error of the ATM VIX option price estimator.
            cost : float
                Total computational cost (proportional to the number of simulated paths
                and time steps).

        Raises
        ------
        ValueError
            If any input parameter is invalid.
        """
        if rule not in ["left", "right", "trap"]:
            raise ValueError("rule must be one of 'left', 'right', or 'trap'.")
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if opt not in ["mc", "mlmc"]:
            raise ValueError("opt must be either 'mc' or 'mlmc'.")
        if n_mse <= 0:
            raise ValueError("n_mse must be a positive integer.")
        if eps <= 0:
            raise ValueError("eps must be a positive float.")
        if true_price is None:
            raise ValueError("true_price must be provided for MSE calculation.")
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = 1234

        def payoff(x):
            return np.maximum(x - true_fut, 0.0)

        price_mse = np.zeros(n_mse)

        if opt == "mc":
            n_mc_eps = np.ceil(eps ** (-2)).astype(int)
            n_disc_eps = np.ceil(eps ** (-1)).astype(int)
            print("n_mc_eps:", n_mc_eps)
            print("n_disc_eps:", n_disc_eps)
            for i in tqdm(range(n_mse)):
                seed_i = seed + i
                vix = self.simulate_vix(
                    T=T,
                    n_mc=n_mc_eps,
                    n_disc=n_disc_eps,
                    seed=seed_i,
                    rule=rule,
                )
                price_mse[i] = payoff(vix).mean()
        else:
            lip = 1.0 / (2.0 * true_fut)  # lipschitz constant for K = F for ATM options
            lim = self.limit_strong_error_vix_right(n=n_disc_mlmc_0, T=T)
            c2 = 10 * (lip * lim) ** 2
            c1 = lip * lim
            level_max = int(np.ceil(np.log(np.sqrt(2) * c1 / eps) / np.log(2)))
            if level_max <= 0:
                raise ValueError("level_max must be >= 1.")
            n_mc_0 = int(np.ceil((2.0 / eps**2) * c2 * (level_max + 1)))
            n_disc_eps = np.array(
                [int(n_disc_mlmc_0 * 2**level) for level in range(level_max + 1)]
            )
            if rule in ["left", "right"]:
                n_mc_eps = np.array(
                    [
                        max(1, int(n_mc_0 / 2 ** (2 * level)))
                        for level in range(level_max + 1)
                    ]
                )
            else:
                n_mc_eps = np.array(
                    [
                        max(1, int(n_mc_0 / 2 ** ((2.0 + self.H) * level)))
                        for level in range(level_max + 1)
                    ]
                )

            print("n_mc_eps:", n_mc_eps)
            print("n_disc_eps:", n_disc_eps)

            for i in tqdm(range(n_mse)):
                seed_i = seed + i

                vix_level_0 = self.simulate_vix(
                    T=T,
                    n_mc=n_mc_eps[0],
                    n_disc=n_disc_eps[0],
                    rule=rule,
                    seed=seed_i,
                )
                price_mlmc = payoff(vix_level_0).mean()
                for level in range(1, level_max + 1):
                    xi_level = self.simulate_vix(
                        T=T,
                        n_mc=n_mc_eps[level],
                        n_disc=n_disc_eps[level],
                        rule=rule,
                        seed=seed_i,
                        return_xi=True,
                    )
                    xi_level = np.asarray(xi_level)
                    if rule == "left":
                        xi_level = xi_level[:-1, :]
                    elif rule == "right":
                        xi_level = xi_level[1:, :]
                    else:
                        xi_level = 0.5 * (xi_level[:-1, :] + xi_level[1:, :])
                    # extract VIX values at current and previous levels
                    vix_level_high = np.sqrt(xi_level.mean(axis=0))
                    vix_level_low = np.sqrt(xi_level[::2, :].mean(axis=0))
                    price_mlmc += (
                        payoff(vix_level_high).mean() - payoff(vix_level_low).mean()
                    )
                price_mse[i] = price_mlmc

        cost = np.sum(n_disc_eps**2 * n_mc_eps)
        mse_mc = np.mean((price_mse - true_price) ** 2)

        return mse_mc, cost

    ####################################################################################
    # Weak approximation methods for VIX pricing
    ####################################################################################
    def price_vix_approx(self, k, T, opttype=1, order=3, return_fut=False) -> float:
        """
        Approximate the price of a VIX option using a proxy expansion.

        Parameters
        ----------
        k : float
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity of the VIX option.
        opttype : int, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.
        return_fut : bool, optional
            If True, return the proxy for the VIX future instead of the option price.
            Default is False.

        Returns
        -------
        float
            Approximated VIX option price (or VIX future if return_fut is True).

        Raises
        ------
        ValueError
            If order or opttype is invalid, or if T <= 0.
        """
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if opttype not in [-1, 1]:
            raise ValueError("opttype must be either -1 (put) or 1 (call).")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if order >= 1:
            gamma_1 = self.gamma_1_proxy(T=T)
        if order >= 2:
            gamma_2 = self.gamma_2_proxy(T=T)
        if order == 3:
            gamma_3 = self.gamma_3_proxy(T=T)

        meanp = self.mean_proxy_flat(T) + np.log(self.fut_vix2(T))
        tot_varp = self.var_proxy_flat(T)
        volp = np.sqrt(tot_varp / T)
        S = np.exp(0.5 * meanp + 0.125 * tot_varp)

        if order >= 0:
            F = S
        if order >= 1:
            F += gamma_1 * 0.5 * S
        if order >= 2:
            F += gamma_2 * 0.25 * S
        if order == 3:
            F += gamma_3 * 0.125 * S
        if return_fut:
            return F

        K = F * np.exp(k)
        # order 0
        price_0 = utils.black_price(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        if order == 0:
            return price_0
        # order 1
        price_1 = (
            0.5 * S * utils.black_delta(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        )
        if order == 1:
            return price_0 + gamma_1 * price_1
        # order 2
        price_2 = 0.5 * price_1
        price_2 += 0.25 * S**2 * utils.black_gamma(K=K, T=T, F=S, vol=0.5 * volp)
        if order == 2:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2
        # order 3
        price_3 = -0.5 * price_1 + 1.5 * price_2
        price_3 += 0.125 * S**3 * utils.black_speed(K=K, T=T, F=S, vol=0.5 * volp)
        if order == 3:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

        raise ValueError("Invalid order specified for VIX option price approximation.")

    def price_vix_fut_approx(self, T, order=3) -> float:
        """
        Approximate the price of a VIX futures contract at maturity T using a proxy
        expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.

        Returns
        -------
        float
            Approximated VIX futures price.

        Raises
        ------
        ValueError
            If order is invalid or T <= 0.
        """
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        meanp = self.mean_proxy_flat(T) + np.log(self.fut_vix2(T))
        tot_varp = self.var_proxy_flat(T)
        S = np.exp(meanp / 2.0 + tot_varp / 8.0)
        # order 0
        price_0 = S
        if order == 0:
            return price_0
        # order 1
        gamma_1 = self.gamma_1_proxy_flat(T=T)
        price_1 = 0.5 * S
        if order == 1:
            return price_0 + gamma_1 * price_1
        # order 2
        gamma_2 = self.gamma_2_proxy_flat(T=T)
        price_2 = 0.25 * S
        if order == 2:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2
        # order 3
        gamma_3 = self.gamma_3_proxy_flat(T=T)
        price_3 = 0.125 * S
        if order == 3:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

        raise ValueError("Invalid order specified for VIX futures price approximation.")

    def price_vix_approx_mixed(
        self, T, K, lbd, eta_2, opt_payoff, order=3, n_quad: int = 50, eps=1e-3
    ) -> float:
        """
        Price a VIX option in the mixed case using the weak approximation.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        K : float
            Strike of the VIX option.
        lbd : float
            Mixing parameter between the two regimes.
        eta_2 : float
            Volatility of volatility parameter for the second exponential.
        opt_payoff : callable
            Payoff function of the option, e.g., lambda x: np.maximum(x - K, 0)
            for a call.
        order : int, optional
            Order of the approximation expansion (default is 3).
        n_quad : int, optional
            Number of quadrature points for numerical integration (default is 50).
        eps : float, optional
            Finite-difference parameter for numerical integration (default is 1e-3).

        Returns
        -------
        float
            Approximated price of the VIX option using the mixed method.
        """
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        rbergomi_eta_2 = self.__class__(
            s0=self.s0, xi0=self.xi0, H=self.H, eta=eta_2, rho=self.rho
        )

        fvix2 = self.fut_vix2(T)
        eta_1 = self.eta
        meanp_1 = self.mean_proxy_flat(T)
        meanp_2 = rbergomi_eta_2.mean_proxy_flat(T)
        sigp_1 = self.var_proxy_flat(T) ** 0.5
        sigp_2 = rbergomi_eta_2.var_proxy_flat(T) ** 0.5

        # Define the payoff function based on the option type
        def payoff_fut(x):
            return np.sqrt(x)

        def payoff_call(x, K=K):
            return np.maximum(np.sqrt(x) - K, 0.0)

        def payoff_put(x, K=K):
            return np.maximum(K - np.sqrt(x), 0.0)

        if opt_payoff == "fut":
            payoff = payoff_fut
        elif opt_payoff == "call":
            payoff = payoff_call
        elif opt_payoff == "put":
            payoff = payoff_put
        else:
            raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")

        gamma_1_1 = self.gamma_1_proxy(T=T)
        gamma_2_1 = self.gamma_2_proxy(T=T)
        gamma_3_1 = self.gamma_3_proxy(T=T)
        gamma_1_2 = rbergomi_eta_2.gamma_1_proxy(T=T)
        gamma_2_2 = rbergomi_eta_2.gamma_2_proxy(T=T)
        gamma_3_2 = rbergomi_eta_2.gamma_3_proxy(T=T)

        mu0 = meanp_1 / self.eta**2.0

        nodes, weights = utils.gauss_hermite(n_quad)

        exp_1 = np.exp(meanp_1 + sigp_1 * nodes)
        exp_2 = np.exp(meanp_2 + sigp_2 * nodes)
        price_0 = np.sum(weights * payoff(fvix2 * (lbd * exp_1 + (1.0 - lbd) * exp_2)))

        # order 0
        if order == 0:
            return price_0

        def payoff_mixed(x, y, lbd, eta_1, eta_2, mu0):
            term_0 = lbd * np.exp(x + y)
            term_1 = (1 - lbd) * np.exp(
                eta_2 * (eta_1 - eta_2) * (-mu0) + (eta_2 / eta_1) * x
            )
            return payoff(fvix2 * (term_0 + term_1))

        def psi(x, idx=1):
            if idx == 1:

                def f_idx(y):
                    return payoff_mixed(meanp_1 + sigp_1 * x, y, lbd, eta_1, eta_2, mu0)

            else:

                def f_idx(y):
                    return payoff_mixed(
                        meanp_2 + sigp_2 * x, y, 1.0 - lbd, eta_2, eta_1, mu0
                    )

            return (f_idx(eps) - f_idx(-eps)) / (2.0 * eps)

        price_1 = gamma_1_1 * np.sum(
            weights * np.array([psi(x, idx=1) for x in nodes])
        ) + gamma_1_2 * np.sum(weights * np.array([psi(x, idx=2) for x in nodes]))

        if order == 1:
            return price_0 + price_1

        price_2 = (
            gamma_2_1
            * np.sum(weights * np.array([x * psi(x, idx=1) for x in nodes]))
            / sigp_1
            + gamma_2_2
            * np.sum(weights * np.array([x * psi(x, idx=2) for x in nodes]))
            / sigp_2
        )

        if order == 2:
            return price_0 + price_1 + price_2

        price_3 = (
            gamma_3_1
            * np.sum(weights * np.array([(x**2 - 1.0) * psi(x, idx=1) for x in nodes]))
            / sigp_1**2
            + gamma_3_2
            * np.sum(weights * np.array([(x**2 - 1.0) * psi(x, idx=2) for x in nodes]))
            / sigp_2**2
        )

        if order == 3:
            return price_0 + price_1 + price_2 + price_3

        raise ValueError("Invalid order specified for VIX option price approximation.")

    def implied_vol_vix_approx(self, T, k, order=3):
        """
        Approximate the implied volatility of a VIX option at a given log-moneyness
        using the proxy expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM). Can be a scalar or array.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.

        Returns
        -------
        np.ndarray
            Approximated implied volatility for each log-moneyness value.

        Raises
        ------
        ValueError
            If T <= 0.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))
        F = self.price_vix_fut_approx(T=T, order=order)
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self.price_vix_approx(T=T, k=k_i, opttype=opttype_i, order=order)
                for k_i, opttype_i in zip(k, opttype, strict=True)
            ]
        )
        return utils.black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    def implied_vol_vix_approx_mixed(
        self, T, k, order=3, lbd=0.5, eta_2=1.0, n_quad=50, eps=1e-3
    ):
        """
        Compute the implied volatility of a VIX option using a mixed
        approximation method.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float
            Log-moneyness of the VIX option.
        order : int, optional
            Order of the approximation expansion (default is 3).
        lbd : float, optional
            Mixing parameter between the two regimes (default is 0.5).
        eta_2 : float, optional
            Volatility of volatility parameter for the second exponential
            (default is 1.0).
        n_quad : int, optional
            Number of quadrature points for numerical integration (default is 50).
        eps : float, optional
            Tolerance for numerical integration (default is 1e-3).

        Returns
        -------
        float
            Approximated Black-Scholes implied volatility for the VIX option.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))
        F = self.price_vix_approx_mixed(
            T=T, K=0.0, lbd=lbd, eta_2=eta_2, order=order, opt_payoff="fut"
        )
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self.price_vix_approx_mixed(
                    T=T,
                    K=K_i,
                    lbd=lbd,
                    eta_2=eta_2,
                    opt_payoff="call" if opttype_i == 1 else "put",
                    order=order,
                    n_quad=n_quad,
                    eps=eps,
                )
                for K_i, opttype_i in zip(K, opttype, strict=True)
            ]
        )
        return utils.black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    def mean_proxy(self, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the mean of the VIX proxy (log-variance process) at maturity T.

        The mean is defined as:
        -1/2 * \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        } dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 40.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Mean of the VIX proxy at maturity T.
        """
        if quad_scipy:

            def integrand(t):
                integral = integrate.quad(
                    lambda u: self.xi0(u) * self.kernel(u, t) ** 2,
                    T,
                    T + self.delta_vix,
                )
                return integral[0] / self.delta_vix

            return -0.5 * integrate.quad(integrand, 0, T)[0] / self.fut_vix2(T)

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            mean = -0.5 * T / self.fut_vix2(T)
            mean *= np.sum(
                v_weights[:, None]
                * v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2
            )
            return mean

    def mean_proxy_flat(self, T):
        """
        Compute the mean of the VIX proxy when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity.

        Returns
        -------
        float
            Mean of the VIX proxy at maturity T for flat xi0.
        """
        mean = (
            (T + self.delta_vix) ** (2.0 * self.H + 1.0)
            - self.delta_vix ** (2.0 * self.H + 1.0)
            - T ** (2.0 * self.H + 1.0)
        )
        mean *= -(self.eta**2) / (2.0 * self.delta_vix * (2.0 * self.H + 1.0))
        return mean

    def var_proxy(self, T, n_quad=30, quad_scipy: bool = True):
        r"""
        Compute the variance of the VIX proxy at maturity T.

        The variance is defined as:
        \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        }^2 dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 20.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Variance of the VIX proxy at maturity T.
        """
        if quad_scipy:
            fvix2 = self.fut_vix2(T)

            def integrand(t):
                integral = (
                    integrate.quad(
                        lambda u: self.xi0(u) * self.kernel(u, t),
                        T,
                        T + self.delta_vix,
                    )[0]
                    / fvix2
                )
                return (integral / self.delta_vix) ** 2

            return integrate.quad(integrand, 0, T)[0]
        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            inner = np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            ) / self.fut_vix2(T)
            return T * np.sum(v_weights * inner**2)

    def var_proxy_flat(self, T):
        """
        Compute the variance of the VIX proxy when the forward variance curve xi0 is
        flat.

        Parameters
        ----------
        T : float
            Maturity.

        Returns
        -------
        float
            Variance of the VIX proxy at maturity T for flat xi0.
        """
        var = (
            (T + self.delta_vix) ** (2.0 * self.H + 2.0)
            + T ** (2 * self.H + 2)
            - self.delta_vix ** (2 * self.H + 2)
        ) / (2 * self.H + 2)
        var -= (
            2
            * self.delta_vix ** (self.H + 0.5)
            * T ** (self.H + 1.5)
            * special.hyp2f1(
                -self.H - 0.5, self.H + 1.5, self.H + 2.5, -T / self.delta_vix
            )
            / (self.H + 1.5)
        )
        var *= (
            self.eta * np.sqrt(2.0 * self.H) / (self.delta_vix * (self.H + 0.5))
        ) ** 2

        return var

    def integral_kernel(self, t, T):
        r"""
        Compute the normalized integral of the kernel over the VIX proxy time interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t) du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized kernel integral.
        """
        return (
            integrate.quad(
                lambda u: self.xi0(u) * self.kernel(u, t), T, T + self.delta_vix
            )[0]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def integral_kernel_squared(self, t, T):
        r"""
        Compute the normalized integral of the squared kernel over the VIX proxy time
        interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t)^2 du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized squared kernel integral.
        """
        return (
            integrate.quad(
                lambda u: self.xi0(u) * self.kernel(u, t) ** 2, T, T + self.delta_vix
            )[0]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def gamma_1_proxy(self, T, n_quad=30):
        """
        Compute the first-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            First-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel_squared = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            )
            / fvix2
        )
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = (
            0.125
            * T**2
            * np.sum(
                v_weights[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    ** 2
                    - int_kernel_squared[None, :]
                ),
                axis=1,
            )
            ** 2
        )
        integrand += (
            0.5
            * T
            * np.sum(
                v_weights[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    - int_kernel[None, :]
                )
                ** 2,
                axis=1,
            )
        )
        gamma_1 = (
            np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )

        return gamma_1

    def gamma_1_proxy_flat(self, T):
        """
        Compute the first-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.

        Returns
        -------
        float
            First-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        tmp0 = (
            (T + self.delta_vix) ** (4.0 * self.H + 1.0)
            + self.delta_vix ** (4.0 * self.H + 1.0)
            - T ** (4.0 * self.H + 1.0)
        )
        tmp0 /= (4.0 * self.H + 1.0) * self.delta_vix
        tmp1 = (
            T ** (2 * self.H + 1)
            + self.delta_vix ** (2 * self.H + 1)
            - (T + self.delta_vix) ** (2 * self.H + 1)
        )
        tmp1 /= self.delta_vix * (2 * self.H + 1)
        tmp1 = -(tmp1**2)
        tmp2 = (
            -2
            * T ** (2 * self.H)
            * self.delta_vix ** (2 * self.H)
            * special.hyp2f1(
                -2 * self.H, 2 * self.H + 1, 2 * self.H + 2, -self.delta_vix / T
            )
            / (2.0 * self.H + 1.0)
        )
        gamma_10 = self.eta**4 * (tmp0 + tmp1 + tmp2) / 8.0

        gamma_11 = (
            (T + self.delta_vix) ** (2 * self.H + 1)
            - T ** (2 * self.H + 1)
            - self.delta_vix ** (2 * self.H + 1)
        )
        gamma_11 *= self.eta**2 / (2.0 * (2.0 * self.H + 1.0) * self.delta_vix)
        gamma_11 -= 0.5 * self.var_proxy(T)

        return gamma_10 + gamma_11

    def gamma_2_proxy(self, T, n_quad=30):
        """
        Compute the second-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            Second-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel_squared = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            )
            / fvix2
        )
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = np.sum(
            v_weights[None, :]
            * int_kernel[None, :]
            * (
                self.kernel(
                    u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                )
                - int_kernel[None, :]
            ),
            axis=1,
        )
        integrand *= np.sum(
            v_weights[None, :]
            * (
                self.kernel(
                    u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                )
                ** 2
                - int_kernel_squared[None, :]
            ),
            axis=1,
        )
        gamma_2 = (
            -0.5
            * T**2
            * np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )

        return gamma_2

    def gamma_2_proxy_flat(self, T, quad_scipy: bool = False, n_quad: int = 30):
        """
        Compute the second-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. If False, use Gauss-Legendre
            quadrature. Default is False.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 50.

        Returns
        -------
        float
            Second-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        def g2(t, a):
            tmp = (
                (T * t + self.delta_vix) ** (self.H + 0.5) - (T * t) ** (self.H + 0.5)
            ) * (
                (T + a * self.delta_vix) ** (2.0 * self.H)
                - (a * self.delta_vix) ** (2.0 * self.H)
            )
            tmp *= (T * t + a * self.delta_vix) ** (self.H - 0.5)
            return tmp

        if quad_scipy:

            def inner_integral(a):
                return integrate.quad(lambda t: g2(t, a), 0.0, 1.0)[0]

            integral = integrate.quad(inner_integral, 0.0, 1.0)[0]

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)

            def inner_integral(a):
                return np.sum(v_weights * g2(v_nodes, a))

            integral = np.sum(
                v_weights * np.array([inner_integral(a) for a in v_nodes])
            )

        gamma_2 = (
            -integral * T * self.eta**4 * self.H / (self.delta_vix * (self.H + 0.5))
        )
        gamma_2 += (
            self.eta**2
            * (
                (T + self.delta_vix) ** (2.0 * self.H + 1)
                - T ** (2.0 * self.H + 1.0)
                - self.delta_vix ** (2.0 * self.H + 1.0)
            )
            * self.var_proxy_flat(T)
        ) / (2.0 * (2.0 * self.H + 1.0) * self.delta_vix)

        return gamma_2

    def gamma_3_proxy(self, T, n_quad=30):
        """
        Compute the third-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            Third-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = (
            np.sum(
                v_weights[None, :]
                * int_kernel[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    - int_kernel[None, :]
                ),
                axis=1,
            )
            ** 2
        )
        gamma_3 = (
            0.5
            * T**2
            * np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )
        return gamma_3

    def gamma_3_proxy_flat(self, T, quad_scipy: bool = False, n_quad: int = 30):
        """
        Compute the third-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. If False, use Gauss-Legendre
            quadrature. Default is False.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 50.

        Returns
        -------
        float
            Third-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        def g3(t, a):
            d = self.delta_vix / T
            tmp0 = (
                (a * d) ** (self.H - 0.5)
                * special.hyp2f1(
                    -self.H + 0.5, self.H + 1.5, self.H + 2.5, -1.0 / (a * d)
                )
                / (self.H + 1.5)
            )
            tmp1 = (1 + a * d) ** (self.H + 0.5) * special.hyp2f1(
                -self.H - 0.5,
                self.H + 0.5,
                self.H + 1.5,
                -(1.0 + a * d) / (d - a * d),
            )
            tmp1 -= (a * d) ** (self.H + 0.5) * special.hyp2f1(
                -self.H - 0.5, self.H + 0.5, self.H + 1.5, -(a * d) / (d - a * d)
            )
            tmp1 *= (1.0 - a) ** (self.H + 0.5) * d ** (self.H + 0.5) / (self.H + 0.5)
            tmps = tmp1 - tmp0
            return (
                T ** (4.0 * self.H)
                * ((t + d) ** (self.H + 0.5) - t ** (self.H + 0.5))
                * (t + a * d) ** (self.H - 0.5)
                * tmps
            )

        if quad_scipy:

            def inner_integral(a):
                return integrate.quad(lambda t: g3(t, a), 0.0, 1.0)[0]

            integral = integrate.quad(inner_integral, 0.0, 1.0)[0]

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)

            def inner_integral(a):
                return np.sum(v_weights * g3(v_nodes, a))

            integral = np.sum(
                v_weights * np.array([inner_integral(a) for a in v_nodes])
            )

        gamma_3 = (
            2.0
            * self.eta**4
            * self.H**2
            * T**2
            / (self.delta_vix**2 * (self.H + 0.5) ** 2)
        )
        gamma_3 *= integral
        gamma_3 -= 0.5 * self.var_proxy_flat(T) ** 2

        return gamma_3
