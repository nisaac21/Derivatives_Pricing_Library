from abc import ABC, abstractmethod
import numpy as np

class PayOff(ABC):

    @abstractmethod
    def pay_off(self, spot_price : float) -> float:
        """Returns the pay off price at expiry for call option"""
        pass

class PayOffEuropeanCall(PayOff):
    """Pay off for European call options"""

    def __init__(self, strike_price):
        self.K = strike_price
    

    def pay_off(self, spot_price: float) -> float:
        return np.maximum(spot_price - self.K, 0)
    
class PayOffEuropeanPut(PayOff):
    """Pay off for European put options"""

    def __init__(self, strike_price):
        self.K = strike_price
    
    def pay_off(self, spot_price: float) -> float:
        return np.maximum(self.K - spot_price, 0)
    
class PayOffDigitalCall(PayOff):
    """Pay off for Digital Call"""

    def __init__(self, strike_price, coupon):
        self.K = strike_price
        self.C = coupon
    

    def pay_off(self, spot_price: float) -> float:
        if spot_price <= self.K:
            return self.C
        else:
            return 0
    
class PayOffDigitalPut(PayOff):
    """Pay off for Digital Put"""

    def __init__(self, strike_price, coupon):
        self.K = strike_price
        self.C = coupon
    

    def pay_off(self, spot_price: float) -> float:
        if spot_price >= self.K:
            return self.C
        else:
            return 0

class PayOffDoubleDigital(PayOff):
    """Pay off for a Double Digital Option"""

    def __init__(self, upper_strike_price, lower_strike_price, coupon):
        self.U = upper_strike_price
        self.D = lower_strike_price
        self.C = coupon
    

    def pay_off(self, spot_price: float) -> float:
        if spot_price >= self.D and spot_price <= self.U:
            return self.C
        else:
            return 0