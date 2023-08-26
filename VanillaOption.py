import numpy as np
from scipy.stats import norm
from typing import Union, Literal
import math

from quant_math import gbm_simulation
from PayOff import PayOffEuropeanCall, PayOffEuropeanPut

def validate_option_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError("Invalid option_type. Allowed values are 'call' or 'put'.")


class VanillaOption():
    """Class for carrying out computations on a Vinalla Option"""

    def __init__(self, strike_price : float, risk_free_rate : float, maturity_time : float, 
                 underlying_price : float, volatility : float) -> None:
        """Parameters
        ----------
        strike_price : (float) representing the strike price of the option (K)
        
        risk_free_rate : (float) representing the risk-free rate, enter 5% as 0.05 (r)
        
        maturity_time : (float) representing the time to expiry in years (T)
        
        underlying_price : (float) representing the price of the underlying asset (S)
        
        volatility : (float) the volatility of the underlying, enter 5% as 0.05 (sigma)"""

        self.K = strike_price
        self.r = risk_free_rate
        self.T = maturity_time
        self.S = underlying_price
        self.sigma = volatility

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

    def _get_d_i(self, i : Literal[1, 2]):
        sigma_sqrt_t = self.sigma * np.sqrt(self.T)
        return (np.log(self.S / self.K) + (self.r + np.power(-1, i - 1) * (self.sigma ** 2) / 2) * self.T) / sigma_sqrt_t

    def black_scholes_price(self, option_type : Literal["call", "put"] = 'call') -> float:
        """Calculates the call price of the option using Black-Scholes Formula

        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            option_price : (float) calculated option price"""

        validate_option_type(option_type)
        
        d_1 = self._get_d_i(1)
        
        d_2 = self._get_d_i(2)

        if option_type == 'call':
            return self.S * norm.cdf(d_1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d_2) 
        else:
            return norm.cdf(-d_2) * self.K * np.exp(- self.r * self.T) - norm.cdf(- d_1) * self.S
    
    def monte_carlo_pricing(self, option_type : Union[Literal["call"], Literal["put"]] = 'call', num_sims=1000):
        """Returns the price of a Euopean option using a Monte Carlo Simulation. Note 
        that the price gets more accurate as the simulations increase 
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type 
            num_sims : (int) number of Monte Carlo Simulations to run
            
        Returns
        -------
            option_price : (float) calculated option price"""
        
        validate_option_type(option_type)
        
        total_trading_hours = int(self.T * 252 * 6.5)
        
        # simulating various option paths with Geometric Brownian Motion
        sims = gbm_simulation(starting_point=self.S, drift_coefficient=self.r, 
                              time_steps=total_trading_hours, total_time=self.T, 
                              variance=self.sigma, num_sims=num_sims)
        
        # determining all payouts from various paths 
        if option_type == 'call':
            payoffs = PayOffEuropeanCall(self.K).pay_off(sims[-1])
        else:
            payoffs = PayOffEuropeanPut(self.K).pay_off(sims[-1])


        # calculating option price
        return np.exp(- self.r * self.T) * np.mean(payoffs)
    
    def delta(self, option_type: Literal["call", "put"] = 'call' ) -> float:
        """Returns the Delta value of an option through analytic formula
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""
        validate_option_type(option_type)
        
        option_change = 1 if option_type == 'call' else 0

        d_1 = self._get_d_i(1)

        return norm.cdf(d_1) - option_change
    
    def gamma(self) -> float:
        """Returns the Gamma value of an option through analytic formula
            
        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""
        d_1 = self._get_d_i(1)
        
        return norm.pdf(1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float: 
        """Returns the Vega value of an option through analytic formula
            
        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""
        
        d_1 = self._get_d_i(1)

        return self.S * self.pdf(d_1) * np.sqrt(self.T)
    
    def theta(self, option_type: Literal["call", "put"] = 'call' ) -> float: 
        """Returns the Theta value of an option through analytic formula
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""
        validate_option_type(option_type)

        d_1 = self._get_d_i(1)
        d_2 = self._get_d_i(2)

        option_change = 1 if option_type == 'call' else -1

        return ((- self.S * norm.pdf(d_1) * self.sigma) / (2 * np.sqrt(self.T))) - \
            option_change * self.r * self.K * np.exp(- self.r * self.T) * norm.cdf(option_change * d_2)
    
    def rho(self, option_type: Literal["call", "put"] = 'call' ) -> float: 
        """Returns the Rho value of an option through analytic formula
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""
        
        validate_option_type(option_type)

        d_2 = self._get_d_i(2)

        option_change = 1 if option_type == 'call' else -1

        return option_change * self.K(self.T) * np.exp(- self.r * self.T) * norm.cdf(option_change * d_2)
    
    def option_greeks(self, option_type: Literal["call", "put"] = 'call' ) -> dict: 
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
            'delta' : self.delta(option_type),
            'gamma' : self.gamma(),
            'vega' : self.vega(),
            'theta' : self.theta(option_type),
            'rho' : self.rho(option_type)
        }
    
    def delta_fdm(self, option_type: Literal["call", "put"] = 'call', delta_S : int = 2*31) -> float: 
        """Returns the Delta values of an option utilizing the finite difference method 
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
        
        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""
        
        validate_option_type(option_type)
        
        if option == 'call':
            return self.black_scholes_price()
    
    def gamma(self) -> float:
        """Returns the Gamma value of an option through analytic formula
            
        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""
        d_1 = self._get_d_i(1)
        
        return norm.pdf(1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float: 
        """Returns the Vega value of an option through analytic formula
            
        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""
        
        d_1 = self._get_d_i(1)

        return self.S * self.pdf(d_1) * np.sqrt(self.T)
    
    def theta(self, option_type: Literal["call", "put"] = 'call' ) -> float: 
        """Returns the Theta value of an option through analytic formula
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""
        validate_option_type(option_type)

        d_1 = self._get_d_i(1)
        d_2 = self._get_d_i(2)

        option_change = 1 if option_type == 'call' else -1

        return ((- self.S * norm.pdf(d_1) * self.sigma) / (2 * np.sqrt(self.T))) - \
            option_change * self.r * self.K * np.exp(- self.r * self.T) * norm.cdf(option_change * d_2)
    
    def rho(self, option_type: Literal["call", "put"] = 'call' ) -> float: 
        """Returns the Rho value of an option through analytic formula
        
        Parameters
        ----------
            option_type : (str) one of ['call' or 'put'] for desired option type
            
        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""
        
        validate_option_type(option_type)

        d_2 = self._get_d_i(2)

        option_change = 1 if option_type == 'call' else -1

        return option_change * self.K(self.T) * np.exp(- self.r * self.T) * norm.cdf(option_change * d_2)


    
if __name__ == '__main__':

    option = VanillaOption(100, 0.05, 1, 90, 0.2)

    print(f"Black Scholes Price is {option.black_scholes_price(option_type='put')}")
    print(f"Monte Carlo Price is {option.monte_carlo_pricing(option_type='put', num_sims=100_000)}")