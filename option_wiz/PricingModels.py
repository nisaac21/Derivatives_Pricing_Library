import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

from typing import Literal

from utils import validate_option_type, validate_d_i
from quant_math import gbm_simulation, merton_jump_diff, heston_path
from PayOff import PayOff

from abc import ABC, abstractmethod

### MONTE CARLO THETA NOT WORKING ###


class PricingModel(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def option_price(self) -> None:
        pass

    @abstractmethod
    def delta(self) -> float:
        pass

    @abstractmethod
    def gamma(self) -> float:
        pass

    @abstractmethod
    def vega(self) -> float:
        pass

    @abstractmethod
    def theta(self) -> float:
        pass

    @abstractmethod
    def rho(self) -> float:
        pass

    @abstractmethod
    def greeks(self) -> dict:
        pass


class Simulator(PricingModel):

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float,
                 pay_off: PayOff):

        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.pay_off = pay_off

    def _time_steps(self) -> int:
        """Amount of hours in self.T years"""
        return int(self.T * 252 * 6.5)

    @abstractmethod
    def _get_pricing_params(self, num_sims: int) -> dict:
        pass

    @abstractmethod
    def _model_pricing(self) -> float:
        pass

    def option_price(self, num_sims: int) -> float:
        """Returns simulated option_price

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            option_price : (float) representing the option price"""

        return self._model_pricing(self._get_pricing_params(num_sims))

    def delta(self, num_sims: int) -> float:
        """Returns the Delta value of an option through finite differencing

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""

        delta_S = self.T / self._time_steps()

        params = self._get_pricing_params(num_sims)

        params_delta = params.copy()

        params_delta['S'] += delta_S

        return (self._model_pricing(**params_delta)
                - self._model_pricing(**params)) / delta_S

    def gamma(self, num_sims: int) -> float:
        """Returns the Gamma value of an option through finite differencing

        Parameters
        ----------
            num_sims : (int) number 

        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""

        delta_S = self.T / self._time_steps()

        params = self._get_pricing_params(num_sims)

        params_delta_up = params.copy()
        params_delta_up['S'] += delta_S

        params_delta_down = params.copy()
        params_delta_down['S'] -= delta_S

        return (self._model_pricing(**params_delta_up) -
                2 * self._model_pricing(**params) +
                self._model_pricing(**params_delta_down)) / (delta_S ** 2)

    def vega(self, num_sims: int) -> float:
        """Returns the Vega value of an option through finite_differencing

        Parameters
        ----------
            num_sims : (int) number 

        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""

        delta_sigma = self.T / self._time_steps()

        params = self._get_pricing_params(num_sims)

        params_delta = params.copy()

        params_delta['sigma'] += delta_sigma

        return (self._model_pricing(**params_delta)
                - self._model_pricing(**params)) / delta_sigma

    def theta(self, num_sims: int = 1000) -> float:
        """Returns the Theta value of an option through finite differencing

        Parameters
        ----------
            S : (float) underlying price

            K : (float) strike price

            r : (float) risk-free rate, 0.05 means 5%

            sigma : (float) volatility, 0.05 means 5%

            T : (float) time till maturity in years

            pay_off : (PayOff) class for determining optoin intrinsic value at expiration

            option_type : (str) one of ['call' or 'put'] for desired option type

            num_sims : (int) number of simultaions to run

            random_jump_draws : (np.array) pre-sample random draws for the jump component of the model in size (int(T * 252 * 6.5), num_sims)

        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""

        delta_T = self.T / self._time_steps()  # one extra time step

        params = self._get_pricing_params(num_sims)

        params_delta = params.copy()

        params_delta['T'] += delta_T

        return (self._model_pricing(**params_delta)
                - self._model_pricing(**params)) / (- delta_T)

    def rho(self, num_sims: int) -> float:
        """Returns the Rho value of an option through finite_differencing

        Parameters
        ----------
            num_sims : (int) number 

        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""

        delta_rho = self.T / self._time_steps()

        params = self._get_pricing_params(num_sims)

        params_delta = params.copy()

        params_delta['rho'] += delta_rho

        return (self._model_pricing(**params_delta)
                - self._model_pricing(**params)) / delta_rho

    def greeks(self, num_sims: int) -> dict:

        return {
            'delta': self.delta(num_sims),
            'gamma': self.gamma(num_sims),
            'vega': self.vega(num_sims),
            'theta': self.theta(num_sims),
            'rho': self.rho(num_sims)
        }


class AnalyticFormula():
    """Class for analytically deriving option characteristic"""

    def __init__(self):
        pass

    def _get_d_i(self, S: float, K: float, r: float,
                 sigma: float, T: float, i: Literal[1, 2]) -> float:
        """Returns the d_i components of the balck scholes formula where i is 1 or 2

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            i : (int) either 1 or 2 representing d_1 or d_2 of black scholes formula

        Returns
        -------
            d_i : (float) representing respective component of black scholes formula"""

        validate_d_i(i)

        sigma_sqrt_t = sigma * np.sqrt(T)
        return (np.log(S / K) + (r + np.power(-1, i - 1) * (sigma ** 2) / 2) * T) / sigma_sqrt_t

    def black_scholes_price(self, S: float, K: float,
                            r: float, sigma: float, T: float, option_type: Literal["call", "put"] = 'call') -> float:
        """Calculates the call price of the option using Black-Scholes Formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            option_price : (float) calculated option price"""

        validate_option_type(option_type)

        d_1 = self._get_d_i(S, K, r, sigma, T, 1)

        d_2 = self._get_d_i(S, K, r, sigma, T, 2)

        if option_type == 'call':
            return S * norm.cdf(d_1) - K * np.exp(- r * T) * norm.cdf(d_2)
        else:
            return norm.cdf(-d_2) * K * np.exp(- r * T) - norm.cdf(- d_1) * S

    def delta(self, S: float, K: float,
              r: float, sigma: float, T: float, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Delta value of an option through analytic formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""
        validate_option_type(option_type)

        option_change = 0 if option_type == 'call' else 1

        d_1 = self._get_d_i(S, K, r, sigma, T, 1)

        return norm.cdf(d_1) - option_change

    def gamma(self, S: float, K: float,
              r: float, sigma: float, T: float,) -> float:
        """Returns the Gamma value of an option through analytic formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""

        d_1 = self._get_d_i(S, K, r, sigma, T, 1)

        return norm.pdf(d_1) / (S * sigma * np.sqrt(T))

    def vega(self, S: float, K: float,
             r: float, sigma: float, T: float,) -> float:
        """Returns the Vega value of an option through analytic formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""

        d_1 = self._get_d_i(S, K, r, sigma, T, 1)

        return S * norm.pdf(d_1) * np.sqrt(T)

    def theta(self, S: float, K: float,
              r: float, sigma: float, T: float, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Theta value of an option through analytic formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""
        validate_option_type(option_type)

        d_1 = self._get_d_i(S, K, r, sigma, T, 1)
        d_2 = self._get_d_i(S, K, r, sigma, T, 2)

        option_change = 1 if option_type == 'call' else -1

        return ((- S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))) - \
            option_change * r * K * \
            np.exp(- r * T) * norm.cdf(option_change * d_2)

    def rho(self, S: float, K: float,
            r: float, sigma: float, T: float, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Rho value of an option through analytic formula

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""

        validate_option_type(option_type)

        d_2 = self._get_d_i(S, K, r, sigma, T, 2)

        option_change = 1 if option_type == 'call' else -1

        return option_change * K * T * np.exp(- r * T) * norm.cdf(option_change * d_2)

    def implied_volatility(self, S: float, K: float,
                           r: float, T: float, option_price: float, option_type: Literal["call", "put"] = 'call'):
        """Utilizes the newton-raphson algorithm to compute the implied volatility of an option. Assumes 
        black-sholes is the pricing function and uses vega as its relevant derivative.

        For initial guess formula using Brenner and Subrahmanyam (1988) findings:
        https://www.researchgate.net/publication/245065192_A_Simple_Formula_to_Compute_the_Implied_Standard_Deviation

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            option_price : (float) observered price of the option

            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            implied_volatility : (float) where 0.05 means 5% """

        intial_vol = np.sqrt(2 * np.pi / T) * (option_price / S)

        def f(x): return self.black_scholes_price(
            S, K, r, x, T, option_type) - option_price

        def fprime(x): return self.vega(S, K, r, x, T)

        return newton(func=f, fprime=fprime, x0=intial_vol)


class MonteCarlo(Simulator):
    """Class utilizes Monte Carlo techniques to determine option qualities based on jump diffusion """

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float,
                 pay_off: PayOff, lambda_j: float = 0.1,
                 mu_j: float = -0.2, sigma_j: float = 0.3):
        super().__init__(S0, K, r, sigma, T, pay_off)
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def _random_draws(self, num_sims):
        """Returns a standard random sample"""

        time_steps = self._time_steps()

        return np.random.normal(0, 1, size=(time_steps, num_sims))

    def _random_jump_draws(self, num_sims: int, lambda_j: float = 0.1,
                           mu_j: float = -0.2,  sigma_j: float = 0.3):
        """Returns a random sample of the jump component in Merton's jump-diffusion model

        Parameters
        ----------
            T : (float) total time to simulate over

            num_sim : (int) number of simulations to run 

            lambda_j : (float) jump intensity

            mu_j : (float) average jump size 

            sigma_j : (float) jump size volatility

        Returns
        -------
        """

        n_steps = self._time_steps()
        dt = self.T / n_steps
        return np.random.normal(mu_j, sigma_j, size=(n_steps, num_sims)) * np.random.poisson(lambda_j * dt, size=(n_steps, num_sims))

    def _get_pricing_params(self, num_sims):
        return {
            'S0': self.S0,
            'K': self.K,
            'r': self.r,
            'sigma': self.sigma,
            'T': self.T,
            'pay_off': self.pay_off,
            'lambda_j': self.lambda_j,
            'mu_j': self.mu_j,
            'sigma_j': self.sigma_j,
            'num_sims': num_sims,
            'random_draws': self._random_draws(num_sims),
            'random_jump_draws': self._random_jump_draws(num_sims, self.lambda_j, self.mu_j, self.sigma_j)

        }

    def _model_pricing(self, S0: float, K: float, r: float, sigma: float,
                       T: float, pay_off: PayOff, lambda_j: float = 0.1,
                       mu_j: float = -0.2, sigma_j: float = 0.3,
                       jump_diff: bool = True, num_sims: int = 10_000, random_draws: np.array = None, random_jump_draws: np.array = None):
        """Returns the price of a Euopean option using a Monte Carlo Simulation. Note 
        that the price gets more accurate as the simulations increase 

        Parameters
        ----------
            S : (float) underlying price 

            K : (float) strike price 

            r : (float) risk-free rate, 0.05 means 5% 

            sigma : (float) volatility, 0.05 means 5% 

            T : (float) time till maturity in years 

            pay_off : (PayOff) class for determining optoin intrinsic value at expiration  

            num_sims : *OPTIONAL* (int) number of Monte Carlo Simulations to run

            random_draws : *OPTIONAL* (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)

            random_jump_draws : (np.array) pre-sample random draws for the jump component of the model in size (int(T * 252 * 6.5), num_sims)
        Returns
        -------
            option_price : (float) calculated option price"""

        total_trading_hours = self._time_steps()

        if jump_diff:
            sims = merton_jump_diff(S0=S0, mu=r, sigma=sigma, lambda_j=lambda_j,
                                    mu_j=mu_j, sigma_j=sigma_j, T=T, n_steps=total_trading_hours,
                                    random_draws=random_draws, num_sims=num_sims, random_jump_draws=random_jump_draws)
        else:
            # simulating various option paths with Geometric Brownian Motion
            sims = gbm_simulation(S0=S0, mu=r,
                                  n_steps=total_trading_hours, T=T,
                                  sigma=sigma, num_sims=num_sims, random_draws=random_draws)

        # determining all payouts from various paths
        payoffs = pay_off.pay_off(sims)

        # calculating option price
        return np.exp(- r * T) * np.mean(payoffs)


class StochasticVolatility(Simulator):

    def __init__(self, S0: float, K: float, r: float, T: float,
                 sigma: float, corr: float, epsilon: float,
                 kappa: float, theta: float, pay_off: PayOff,):

        super().__init__(S0, K, r, sigma, T, pay_off)

        self.corr = corr
        self.epsilon = epsilon
        self.kappa = kappa
        self._theta = theta

    def _random_draws(self, T, num_sims, corr):
        n_steps = self._time_steps(T)
        return np.random.multivariate_normal(mu=[0, 0],
                                             cov=np.array(
                                                 [[1, corr], [corr, 1]]),
                                             size=(num_sims, n_steps))

    def _get_pricing_params(self, num_sims: int) -> dict:
        return {
            'S0': self.S0,
            'r': self.r,
            'T': self.T,
            'sigma': self.sigma,
            'corr': self.corr,
            'epsilon': self.epsilon,
            'kappa': self.kappa,
            'theta': self._theta,
            'pay_off': self.pay_off,
            'num_sims': num_sims,
            'random_draws': self._random_draws(self.T, num_sims, self.corr)
        }

    def _model_pricing(self, S0: float, r: float, T: float,
                       sigma: float, corr: float, epsilon: float,
                       kappa: float, theta: float, pay_off: PayOff, num_sims: int,
                       random_draws: np.array = None) -> float:
        """Returns the price of a Euopean option using a Monte Carlo Simulation based on
        Heston's stochastic volaility model 

        Parameters
        ----------
            S0 : (float) underlying price 

            r : (float) risk-free rate, 0.05 means 5% 

            T : (float) time till maturity in years 

            sigma : (float) volatility, 0.05 means 5% 

            corr : (float) correlation between asset returns and variance 

            epsilon : (float) variance of volatility distribution

            kappa : (float) rate of mean reversion for volatility process

            theta : (float) long-term mean of variance process 

            pay_off : (PayOff) class for determining optoin intrinsic value at expiration  

            num_sims : *OPTIONAL* (int) number of Monte Carlo Simulations to run

            random_draws : (np.array) pre-sample random draws which must be
                                        np.random.multivariate_normal(mu=[0, 0], 
                                                                    cov=np.array([[1, corr], [corr, 1]]),
                                                                    size=(num_sims, n_steps))

        Returns
        -------
            option_price : (float) calculated option price"""

        total_trading_hours = self._time_steps(T)

        sims = heston_path(S0=S0, mu=r, n_steps=total_trading_hours, T=T, sigma=sigma,
                           corr=corr, epsilon=epsilon, kappa=kappa, theta=theta,
                           num_sims=num_sims, random_draws=random_draws)

        # determining all payouts from various paths
        payoffs = pay_off.pay_off(sims)

        # calculating option price
        return np.exp(- r * T) * np.mean(payoffs)
