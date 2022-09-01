import pandas as pd
import numpy as np
import sklearn.linear_model
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import LinearRegression


class FeatureFactory:

    def __init__(self, y, idx):
        self.y = y
        self.index = idx
        self.input_df = pd.DataFrame(data={"y": self.y}).set_index(self.index)

        #TODO Check idx is datetime and same length as y

    def create_time_trend_features(self, order):
        dp = DeterministicProcess(index=self.index,
                                  constant=True,
                                  order=order,
                                  seasonal=False,
                                  drop=True)

        print(f"Creating time trend features of order: {order}")

        return dp.in_sample()

    def create_lag_features(self, lag_steps=1):
        return self.input_df.shift(lag_steps)

    def create_fourier_features(self, freq, fourier_order):
        """Returns a series of sin and cosine terms of order (order) to represent seasonal variability"""

        fourier_terms = CalendarFourier(freq=freq, order=fourier_order)
        dp = DeterministicProcess(index=self.index,
                                  constant=True,
                                  order=0,
                                  seasonal=True,
                                  additional_terms=[fourier_terms],
                                  drop=True)

        print(f"Creating seasonal features for Fourier freq: {freq} and Fourier order: {fourier_order}")

        return dp.in_sample()

    def create_temperature_features(self):
        pass

    def combine_features(self, features: list):
        return pd.concat(features, axis=1, ignore_index=False)















