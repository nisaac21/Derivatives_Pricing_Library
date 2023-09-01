from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence
import numbers
from scipy.stats import gmean

from utils import validate_option_type
from option_wiz.quant_math import gbm_simulation


class PayOff(ABC):

    @abstractmethod
    def __init__(self, strike_price: float, option_type: str) -> None:
        validate_option_type(option_type)
        self.K = strike_price
        self.option_type = option_type

    @abstractmethod
    def pay_off(self, spot_prices: np.array) -> float:
        """Returns the pay off price at expiry for call option"""
        pass


class PayOffEuropean(PayOff):
    """Pay off for European call options"""

    def __init__(self, strike_price, option_type: str):
        super().__init__(strike_price, option_type)

    def pay_off(self, spot_prices: np.array) -> float:
        if self.option_type == 'call':
            return np.maximum(spot_prices[-1] - self.K, 0)
        elif self.option_type == 'put':
            return np.maximum(self.K - spot_prices[-1], 0)


class PayOffDigital(PayOff):
    """Pay off for Digital """

    def __init__(self, strike_price: float, option_type: str, coupon: float):
        super().__init__(strike_price, option_type)
        self.C = coupon

    def pay_off(self, spot_prices: np.array) -> float:

        if self.option_type == 'call':
            return np.where(spot_prices[-1] >= self.K, self.C, 0)
        elif self.option_type == 'put':
            return np.where(spot_prices[-1] <= self.K, self.C, 0)


class PayOffDoubleDigital(PayOff):
    """Pay off for a Double Digital Option"""

    def __init__(self, upper_strike_price, lower_strike_price, coupon):
        self.U = upper_strike_price
        self.D = lower_strike_price
        self.C = coupon

    def pay_off(self, spot_prices: np.array) -> float:
        if self.option_type == 'call':
            return np.where(spot_prices[-1] > self.D and spot_prices[-1] < self.U,
                            self.C, 0)
        elif self.option_type == 'put':
            return np.where(spot_prices[-1] < self.D or spot_prices[-1] > self.U,
                            self.C, 0)


class PayOffAsianOption(PayOff):

    @abstractmethod
    def __init__(self, strike_price: float, option_type: str) -> None:
        super().__init__(strike_price, option_type)

    @abstractmethod
    def _get_mean(self, path):
        pass

    def pay_off(self, spot_prices: np.array) -> float:
        mean = self._get_mean(spot_prices)
        return PayOffEuropean(mean, self.option_type).pay_off(spot_prices)


class PayOffAsianOptionArithmetic(PayOffAsianOption):

    def __init__(self, strike_price: float, option_type: str,) -> None:
        super().__init__(strike_price, option_type)

    def _get_mean(self, path):
        return np.mean(path)


class PayOffAsianOptionGeometric(PayOffAsianOption):

    def __init__(self, strike_price: float, option_type: str) -> None:
        super().__init__(strike_price, option_type)

    def _get_mean(self, path) -> float:
        return gmean(path)
