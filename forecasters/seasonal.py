import pandas as pd
import numpy as np
import sklearn.linear_model
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import LinearRegression


class SeasonalForecast:

    def __init__(self, y, idx):
        self.y = y
        self.index = idx

        #TODO Check idx is datetime and same length as y

    def create_features(self, freq, fourier_order):
        """Returns a series of sin and cosine terms of order (order) to represent seasonal variability"""

        fourier_terms = CalendarFourier(freq=freq, order=fourier_order)
        dp = DeterministicProcess(
            index=self.index,
            constant=True,  # dummy for bias/y-intercept
            order=1,  # order 1 - linear trend
            seasonal=True,
            additional_terms=[fourier_terms],
            drop=True)

        print(f"Creating seasonal features for freq: {freq} and order: {fourier_order}")

        return dp.in_sample()

    def lin_regression_fit(self, freq, order):
        """Return the fitted lin model and the features X"""

        X = self.create_features(freq, order)
        return LinearRegression().fit(X, self.y), X

    def lin_regression_predict(self, freq, order):
        """Return the predicted y values"""

        model, X = self.lin_regression_fit(freq, order)
        # y_pred
        return pd.Series(model.predict(X),
                         index=X.index)






