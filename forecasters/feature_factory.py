import pandas as pd
import numpy as np
import sklearn.linear_model
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import LinearRegression


class FeatureFactory:

    def __init__(self, df):
        self.input_df = df
        self.y = self.input_df.y
        self.index = self.input_df.index

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
        print(f"Creating lag features of lag: {lag_steps}")
        return self.input_df.y.shift(lag_steps).rename(f"ylag_{lag_steps}")

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

    def create_statistical_features(self, mean_window, median_window, stdev_window):
        """Need to be careful of look-ahead leakage. Use the right edge to compute rolling stats
        Lag the y-value by 1 so as to not include the value we want to predict in the rolling stat"""

        y_lag = self.y.shift(1)
        mean_x = y_lag.rolling(mean_window, min_periods=3, center=False).mean().rename(f"mean_{mean_window}")
        median_x = y_lag.rolling(median_window, min_periods=3, center=False).median().rename(f"median_{median_window}")
        stdev_x = y_lag.rolling(stdev_window, min_periods=3, center=False).std().rename(f"stdev_{stdev_window}")

        print(f"Creating rolling window statistical features of mean_{mean_window}, "
              f"median_{median_window}, stdev{stdev_window}")

        return pd.concat([mean_x, median_x, stdev_x], axis=1, ignore_index=False)




    def create_temperature_features(self):
        pass

    def create_temperature_forecast_features(self):
        """Could potentially be used as a leading indicator"""
        pass

    def create_holiday_features(self):
        pass

    def combine_features(self, features: list):
        return pd.concat(features, axis=1, ignore_index=False)















