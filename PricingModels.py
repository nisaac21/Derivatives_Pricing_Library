import numpy as np
from scipy.stats import norm 
from typing import Literal

from utils import validate_option_type, validate_d_i
from quant_math import gbm_simulation
from PayOff import PayOff, PayOffEuropean

### MONTE CARLO THETA NOT WORKING ###

class AnalyticFormula():
    """Class for analytically deriving option characteristic"""

    def __init__(self):
        pass

    def _get_d_i(self, S : float, K : float, r : float, 
                 sigma : float, T : float, i : Literal[1, 2]) -> float:
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

    def black_scholes_price(self, S : float, K : float, 
                     r : float, sigma : float, T : float, option_type : Literal["call", "put"] = 'call') -> float:
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
        
    def delta(self, S : float, K : float, 
                     r : float, sigma : float, T : float, option_type: Literal["call", "put"] = 'call' ) -> float:
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
    
    def gamma(self, S : float, K : float, 
                     r : float, sigma : float, T : float,) -> float:
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
    
    def vega(self, S : float, K : float, 
                     r : float, sigma : float, T : float,) -> float: 
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
    
    def theta(self, S : float, K : float, 
                     r : float, sigma : float, T : float, option_type: Literal["call", "put"] = 'call' ) -> float: 
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
            option_change * r * K * np.exp(- r * T) * norm.cdf(option_change * d_2)
    
    def rho(self, S : float, K : float, 
                     r : float, sigma : float, T : float, option_type: Literal["call", "put"] = 'call' ) -> float: 
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
        

class MonteCarlo():
    """Class utilizes Monte Carlo techniques to determine option qualities"""

    def __init__(self):
        pass

    def mc_option_price(self, S : float, K : float, r : float, sigma : float, 
                        T : float, pay_off : PayOff, num_sims : int = 1000, random_draws : np.array = None):
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
        Returns
        -------
            option_price : (float) calculated option price"""

        total_trading_hours = self._time_steps(T)
        
        # simulating various option paths with Geometric Brownian Motion
        sims = gbm_simulation(starting_point=S, drift_coefficient=r, 
                              time_steps=total_trading_hours, total_time=T, 
                              variance=sigma, num_sims=num_sims, random_draws=random_draws)
        
        # determining all payouts from various paths 
        payoffs = pay_off.pay_off(sims)


        # calculating option price
        return np.exp(- r * T) * np.mean(payoffs)
    
    def _time_steps(self, T):
        return int(T * 252 * 6.5)
    
    def _random_draws(self, T, num_sims):
        """Returns a standard random sample"""

        time_steps = self._time_steps(T)

        return np.random.normal(0, 1, size=(time_steps, num_sims ))
    
    def delta(self, S : float, K : float, r : float, sigma : float, T : float, 
              option_type: Literal["call", "put"] = 'call', num_sims : int = 1000,
              random_draws : np.array = None) -> float:
        """Returns the Delta value of an option through finite differencing and monte_carlo simulations
        
        Parameters
        ----------
            S : (float) underlying price 
            
            K : (float) strike price 
            
            r : (float) risk-free rate, 0.05 means 5% 
            
            sigma : (float) volatility, 0.05 means 5% 
            
            T : (float) time till maturity in years
            
            option_type : (str) one of ['call' or 'put'] for desired option type

            num_sims : (int) number of simulations to run 

            random_draws : (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)
            
        Returns
        -------
            delta : (float) representing the option price's sensitivity to underlying price"""
        
        validate_option_type(option_type)

        delta_S = T / self._time_steps(T)

        if random_draws is None: random_draws = self._random_draws(T, num_sims)
        
        return (self.mc_option_price(S + delta_S, K, r, sigma, T, option_type=option_type, num_sims=num_sims, random_draws=random_draws) \
                - self.mc_option_price(S, K, r, sigma, T, option_type=option_type, num_sims=num_sims, random_draws=random_draws)) / delta_S
    
    def gamma(self, S : float, K : float, r : float, sigma : float, T : float , 
              option_type: Literal["call", "put"] = 'call', num_sims : int = 1000,
              random_draws : np.array = None) -> float:
        """Returns the Gamma value of an option through finite differencing

        Parameters
        ----------
            S : (float) underlying price 
            
            K : (float) strike price 
            
            r : (float) risk-free rate, 0.05 means 5% 
            
            sigma : (float) volatility, 0.05 means 5% 
            
            T : (float) time till maturity in years   
            
            option_type : (str) one of ['call' or 'put'] for desired option type

            num_sims : (int) number of simulations to run 

            random_draws : (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)

        Returns
        -------
            gamma : (float) representing the option delta's sensitivity to underlying price"""
        
        validate_option_type(option_type)

        if random_draws is None: random_draws = self._random_draws(T, num_sims)

        delta_S = T / self._time_steps(T)

        return (self.mc_option_price(S + delta_S, K, r, sigma, T, option_type, num_sims, random_draws) - \
            2 * self.mc_option_price(S, K, r, sigma, T, option_type, num_sims, random_draws) + \
                self.mc_option_price(S - delta_S, K, r, sigma, T, option_type, num_sims, random_draws)) / (delta_S ** 2)
    
    def vega(self, S : float, K : float, r : float, sigma : float, T : float, 
             option_type: Literal["call", "put"] = 'call', num_sims : int = 1000,
             random_draws : np.array = None) -> float: 
        """Returns the Vega value of an option through finite_differencing

        Parameters
        ----------
            S : (float) underlying price 
            
            K : (float) strike price 
            
            r : (float) risk-free rate, 0.05 means 5% 
            
            sigma : (float) volatility, 0.05 means 5% 
            
            T : (float) time till maturity in years 
            
            option_type : (str) one of ['call' or 'put'] for desired option type

            num_sims : (int) number of simulations to run

            random_draws : (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)
            
        Returns
        -------
            vega : (float) representing the option price's sensitivity to volatility"""
        
        validate_option_type(option_type)

        delta_sigma = 1 / self._time_steps(T)

        if random_draws is None: random_draws = self._random_draws(T, num_sims)
        
        return (self.mc_option_price(S, K, r, sigma + delta_sigma, T, option_type, num_sims, random_draws) \
                - self.mc_option_price(S, K, r, sigma, T, option_type, num_sims, random_draws)) / delta_sigma
    
    def theta(self, S : float, K : float, r : float, sigma : float, T : float, 
              option_type: Literal["call", "put"] = 'call', num_sims : int = 1000,
              random_draws : np.array = None) -> float: 
        """Returns the Theta value of an option through finite differencing 
        
        Parameters
        ----------
            S : (float) underlying price 
            
            K : (float) strike price 
            
            r : (float) risk-free rate, 0.05 means 5% 
            
            sigma : (float) volatility, 0.05 means 5% 
            
            T : (float) time till maturity in years 
            
            option_type : (str) one of ['call' or 'put'] for desired option type

            num_sims : (int) number of simultaions to run

            random_draws : (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)
            
        Returns
        -------
            theta : (float) representing the option price's sensitivity to time passed, AKA time value"""
        
        validate_option_type(option_type)

        delta_T = T / self._time_steps(T) # one extra time step 

        if random_draws is None: random_draws = self._random_draws(T + delta_T, num_sims) 

        return (self.mc_option_price(S, K, r, sigma, T  + delta_T, option_type, num_sims, random_draws) \
                - self.mc_option_price(S, K, r, sigma, T, option_type, num_sims, random_draws[:-2, :])) / delta_T #
    
    def rho(self, S : float, K : float, r : float, sigma : float, T : float, 
            option_type: Literal["call", "put"] = 'call', num_sims : int = 1000,
            random_draws : np.array = None) -> float: 
        """Returns the Rho value of an option through finite_differencing formula
        
        Parameters
        ----------
            S : (float) underlying price 
            
            K : (float) strike price 
            
            r : (float) risk-free rate, 0.05 means 5% 
            
            sigma : (float) volatility, 0.05 means 5% 
            
            T : (float) time till maturity in years 
            
            option_type : (str) one of ['call' or 'put'] for desired option type
            
            num_sims : (int) number of simulations to run

            random_draws : (np.array) randomly sample paths from standard normal in size (int(T * 252 * 6.5), num_sims)
            
        Returns
        -------
            rho : (float) representing the option price's sensitivity to interest rate changes"""
        
        validate_option_type(option_type)

        if random_draws is None: random_draws = self._random_draws(T, num_sims)

        delta_r = T / self._time_steps(T)

        return (self.mc_option_price(S, K, r + delta_r, sigma, T, option_type, num_sims, random_draws) \
                - self.mc_option_price(S, K, r, sigma, T, option_type, num_sims, random_draws)) / delta_r


if __name__ == '__main__':
    black_scholes = AnalyticFormula()
    monte_carlo = MonteCarlo()

    print(f"analytic price: {black_scholes.black_scholes_price(100, 90, 0.05, 0.2, 1, 'call')}")
    print(f"mc price: {monte_carlo.mc_option_price(100, 90, 0.05, 0.2, 1, PayOffEuropean(90, 'call'), 10000)}")
