from typing import Literal
from abc import ABC


from PricingModels import AnalyticFormula, MonteCarlo
from PayOff import PayOffEuropean, PayOffAsianOptionArithmetic, PayOffAsianOptionGeometric

ANALYTIC_FORMULA = AnalyticFormula()
MONTE_CARLO = MonteCarlo()


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

        self.K = strike_price
        self.r = risk_free_rate
        self.T = maturity_time
        self.S = underlying_price
        self.sigma = volatility
        self.option_type = option_type

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

    def black_scholes_price(self, option_type: Literal["call", "put"] = 'call') -> float:
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
            option_type=option_type
        )

    def monte_carlo_price(self, option_type: Literal["call", "put"] = 'call', num_sims: int = 1000):
        """Returns the price of a Euopean option using a Monte Carlo Simulation. Note 
        that the price gets more accurate as the simulations increase 

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type 
            num_sims : (int) number of Monte Carlo Simulations to run

        Returns
        -------
            option_price : (float) calculated option price"""

        return MONTE_CARLO.mc_option_price(self.S, self.K, self.r, self.sigma, self.T, self.pay_off, option_type, num_sims)

    def delta(self, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Delta value of an option through analytic formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""

        return ANALYTIC_FORMULA.delta(self.S, self.K, self.r, self.sigma, self.T, option_type)

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

    def theta(self, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Theta value of an option through analytic formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""

        return ANALYTIC_FORMULA.theta(self.S, self.K, self.r, self.sigma, self.T, option_type)

    def rho(self, option_type: Literal["call", "put"] = 'call') -> float:
        """Returns the Rho value of an option through analytic formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type

        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""

        return ANALYTIC_FORMULA.rho(self.S, self.K, self.r, self.sigma, self.T, option_type)

    def option_greeks(self, option_type: Literal["call", "put"] = 'call') -> dict:
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
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
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
                 underlying_price: float, volatility: float, arith_avg: bool) -> None:

        super().__init__(strike_price, risk_free_rate,
                         maturity_time, underlying_price, volatility)

        self.arith_avg = arith_avg

    def monte_carlo_pricing(self, option_type: Literal["call", "put"] = 'call', num_sims: int = 1000) -> float:
        """Returns the price of an Asian option with arithmetic averaging using a Monte Carlo Simulation. Note 
        that the price gets more accurate as the simulations increase 

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type 
            num_sims : (int) number of Monte Carlo Simulations to run

        Returns
        -------
            option_price : (float) calculated option price"""

        pay_off = PayOffAsianOptionArithmetic(
            self.K, option_type) if self.arith_avg else PayOffAsianOptionGeometric(self.K, option_type)

        return MONTE_CARLO.mc_option_price(
            self.S, self.K, self.r, self.sigma, self.T, pay_off, num_sims)
