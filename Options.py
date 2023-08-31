from typing import Literal
from abc import ABC, abstractmethod


from PricingModels import AnalyticFormula, MonteCarlo
from PayOff import PayOff, PayOffEuropean, PayOffAsianOptionArithmetic, PayOffAsianOptionGeometric
from utils import validate_option_type

ANALYTIC_FORMULA = AnalyticFormula()


class Option(ABC):
    """Abstract Base Class for Option type contracts.

    Parameters
    ----------
    strike_price : (float) representing the strike price of the option (K)

    risk_free_rate : (float) representing the risk-free rate, enter 5% as 0.05 (r)

    maturity_time : (float) representing the time to expiry in years (T)

    underlying_price : (float) representing the price of the underlying asset (S)

    volatility : (float) the volatility of the underlying, enter 5% as 0.05 (sigma)"""

    def __init__(self, strike_price: float, risk_free_rate: float, maturity_time: float,
                 underlying_price: float, volatility: float, option_type: Literal["call", "put"] = 'call') -> None:
        validate_option_type(option_type)
        self.K = strike_price
        self.r = risk_free_rate
        self.T = maturity_time
        self.S = underlying_price
        self.sigma = volatility
        self.option_type = option_type

        pay_off = self._create_payoff()

        self.MONTE_CARLO = MonteCarlo(self, pay_off)

    @abstractmethod
    def _create_payoff(self) -> PayOff:
        pass

    def monte_carlo_pricing(self, jump_intensity: float = None, mean_jump: float = None, jump_vol: float = None, num_sims: int = 10_000) -> float:
        """Returns the price of an option wusing a Monte Carlo Simulation based on 
        Merton's jump-diffusion model. Note that the price gets more accurate as the simulations increase 

        Parameters
        ----------
            jump_intensity : (float) jump intensity

            mean_jump : (float) average jump size 

            jump_vol : (float) jump size volatility

            num_sims : (int) number of Monte Carlo Simulations to run

        Returns
        -------
            option_price : (float) calculated option price"""

        if jump_intensity:
            self.MONTE_CARLO.lambda_j = jump_intensity
        if mean_jump:
            self.MONTE_CARLO.mu_j = mean_jump
        if jump_vol:
            self.MONTE_CARLO.sigma_j = jump_vol

        return self.MONTE_CARLO.option_price(num_sims)

    def get_strike_price(self) -> float:
        return self.K

    def get_risk_free_rate(self) -> float:
        return self.r

    def get_maturity_time(self) -> float:
        return self.T

    def get_underlying_price(self) -> float:
        return self.S

    def get_volatility(self) -> float:
        return self.sigma

    def get_option_type(self) -> str:
        return self.option_type

    def delta(self, num_sims: int) -> float:
        """Returns the Delta value of an option through analytic formula

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""

        return self.MONTE_CARLO.delta(num_sims)

    def gamma(self, num_sims: int) -> float:
        """Returns the Gamma value of an option through analytic formula

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""

        return self.MONTE_CARLO.gamma(num_sims)

    def vega(self, num_sims: int) -> float:
        """Returns the Vega value of an option through analytic formula

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""

        return self.MONTE_CARLO.vega(num_sims)

    def theta(self, num_sims) -> float:
        """Returns the Theta value of an option through analytic formula

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""

        return self.MONTE_CARLO.vega(num_sims)

    def rho(self, num_sims: int) -> float:
        """Returns the Rho value of an option through analytic formula

        Parameters
        ----------
            num_sims : (int) number of simulations to run

        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""

        return self.MONTE_CARLO.rho(num_sims)

    def option_greeks(self) -> dict:
        """Returns the Delta, Gamma, Vega, Theta, and Rho values of an option through analytic formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            {
                delta : (float) representing the option price's sensitivity to underlying price
                gamma : (float) representing the option delta's sensitivity to underlying price
                vega : (float) representing the option price's sensitivity to volatility
                theta : (float) representing the option price's sensitivity to time passed, AKA time value
                rho : (float) representing the option price's sensitivity to interest rate changes
            }"""

        return {
            'delta': self.delta(self.option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(self.option_type),
            'rho': self.rho(self.option_type)
        }


class EuropeanOption(Option):
    """Class for carrying out computations on a European Option

    Parameters
    ----------
    strike_price : (float) representing the strike price of the option (K)

    risk_free_rate : (float) representing the risk-free rate, enter 5% as 0.05 (r)

    maturity_time : (float) representing the time to expiry in years (T)

    underlying_price : (float) representing the price of the underlying asset (S)

    volatility : (float) the volatility of the underlying, enter 5% as 0.05 (sigma)"""

    def __init__(self, strike_price: float, risk_free_rate: float, maturity_time: float,
                 underlying_price: float, volatility: float, option_type: Literal["call", "put"] = 'call') -> None:

        super().__init__(strike_price, risk_free_rate, maturity_time,
                         underlying_price, volatility, option_type)

        self.pay_off = PayOffEuropean(self.K, option_type)

        self.MONTE_CARLO = MonteCarlo(self, self.pay_off)

    def black_scholes_price(self) -> float:
        """Calculates the call price of the option using Black-Scholes Formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            option_price : (float) calculated option price"""

        return ANALYTIC_FORMULA.black_scholes_price(
            S=self.S,
            K=self.K,
            r=self.r,
            sigma=self.sigma,
            T=self.T,
            option_type=self.option_type
        )

    def delta(self) -> float:
        """Returns the Delta value of an option through analytic formula

        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""

        return ANALYTIC_FORMULA.delta(self.S, self.K, self.r, self.sigma, self.T, self.option_type)

    def gamma(self) -> float:
        """Returns the Gamma value of an option through analytic formula

        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""

        return ANALYTIC_FORMULA.gamma(self.S, self.K, self.r, self.sigma, self.T)

    def vega(self) -> float:
        """Returns the Vega value of an option through analytic formula

        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""

        return ANALYTIC_FORMULA.vega(self.S, self.K, self.r, self.sigma, self.T)

    def theta(self) -> float:
        """Returns the Theta value of an option through analytic formula

        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""

        return ANALYTIC_FORMULA.theta(self.S, self.K, self.r, self.sigma, self.T, self.option_type)

    def rho(self) -> float:
        """Returns the Rho value of an option through analytic formula

        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""

        return ANALYTIC_FORMULA.rho(self.S, self.K, self.r, self.sigma, self.T, self.option_type)

    def option_greeks(self) -> dict:
        """Returns the Delta, Gamma, Vega, Theta, and Rho values of an option through analytic formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            {
                delta : (float) representing the option price's sensitivity to underlying price
                gamma : (float) representing the option delta's sensitivity to underlying price
                vega : (float) representing the option price's sensitivity to volatility
                theta : (float) representing the option price's sensitivity to time passed, AKA time value
                rho : (float) representing the option price's sensitivity to interest rate changes
            }"""

        return {
            'delta': self.delta(self.option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(self.option_type),
            'rho': self.rho(self.option_type)
        }


class AsianOption(Option):

    """Abstract Base Class for Option type contracts.

    Parameters
    ----------
    strike_price : (float) representing the strike price of the option (K)

    risk_free_rate : (float) representing the risk-free rate, enter 5% as 0.05 (r)

    maturity_time : (float) representing the time to expiry in years (T)

    underlying_price : (float) representing the price of the underlying asset (S)

    volatility : (float) the volatility of the underlying, enter 5% as 0.05 (sigma)

    arith_avg : (boolean) true if arithmetic averaging, otherwise geometric averaging"""

    def __init__(self, strike_price: float, risk_free_rate: float, maturity_time: float,
                 underlying_price: float, volatility: float, arith_avg: bool,
                 option_type: Literal["call", "put"] = 'call') -> None:

        self.arith_avg = arith_avg
        super().__init__(strike_price, risk_free_rate,
                         maturity_time, underlying_price, volatility,
                         option_type)

    def _create_payoff(self) -> PayOff:
        return PayOffAsianOptionArithmetic(
            self.K, self.option_type) if self.arith_avg else PayOffAsianOptionGeometric(self.K, self.option_type)
