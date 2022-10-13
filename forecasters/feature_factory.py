import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier


class FeatureFactory:
    """ Class to create time-based features from an input df which includes a timeseries index"""

    def __init__(self, df):
        """Parameters
        --------------
        df : dataframe
            A dataframe object which includes the following as a minimum:
                - a timeseries index
                - at least one feature column
                - a target column labeled 'y'
        """

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

    def create_statistical_features(self, mean_window, median_window, stdev_window, lag=7):
        """Creates statistical features if mean, median and stdev
        TO avoid look-ahead leakage, uses the right edge to compute rolling stats
        Lags the y-value by 7 days so as to not include the values we want to predict in the rolling stats

        Parameters
        ----------
        mean_window : int
            Time window over which to calc the statistical feature, e.g. 7 returns a rolling mean over 7 days
        median_window : int
            As above
        stdev_window : int
            As above
        lag : int
            Number of steps to lag the data on which the stats are created. Default = 7.
        """

        y_lag = self.y.shift(lag)
        mean_x = y_lag.rolling(mean_window, min_periods=3, center=False).mean().rename(f"mean_{mean_window}")
        median_x = y_lag.rolling(median_window, min_periods=3, center=False).median().rename(f"median_{median_window}")
        stdev_x = y_lag.rolling(stdev_window, min_periods=3, center=False).std().rename(f"stdev_{stdev_window}")

        print(f"Creating rolling window statistical features of mean_{mean_window}, "
              f"median_{median_window}, stdev{stdev_window}")

        return pd.concat([mean_x, median_x, stdev_x], axis=1, ignore_index=False)

    def create_temperature_features(self, temperature_cols):
        """Creates temperature features.
        Requires temperature data to be present in df used to instantiate the class"""

        print(f"Creating temperature feature for {temperature_cols}")
        return self.input_df.loc[:, temperature_cols]

    def create_daylength_features(self, sunrise_col, sunset_col):
        """Creates daylength (hours between sunrise and sunset) features.
        Requires sunrise and sunset data to be present in df used to instantiate the class"""

        print(f"Creating daylength feature using cols: {sunrise_col, sunset_col}")
        time_diff = pd.to_datetime(self.input_df[sunset_col]) - pd.to_datetime(self.input_df[sunrise_col])
        return (time_diff/pd.to_timedelta(1, unit="h")).rename("daylength")

    def create_temperature_forecast_features(self):
        """Could potentially be used as a leading indicator"""
        pass

    def create_holiday_features(self):
        pass

    def combine_features(self, features: list):
        return pd.concat(features, axis=1, ignore_index=False)















