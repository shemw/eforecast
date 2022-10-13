import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from forecasters import boosted_hybrid as bh


class ValidateForecaster:
    """Validates a forecast model against real data using multiple forecast windows.
    Loops over a number of train/test periods, contiguous in time, and stores error metrics for each run"""

    def __init__(self, X_1, X_2, y, naive_forecast, xgb_tuning_params={}):
        """Parameters
        --------------
        X1 : dataframe
            A timeseries feature set for model 1 (e.g. for linear regression)
        X2 : dataframe
            A timeseries feature set for model 2 (e.g. XGBoost)
        y : dataframe
            A single column timeseries of target values to be predicted
        naive_forecast : dataframe
            A single column timeseries of predictions (e.g. target values lagged by 7 days)
        xgb_tuning_params : dict
            Optional dict of XGBoost tuning parameters, see XGBoost docs for allowable params, defaults to empty
        """

        self.X_1 = X_1
        self.X_2 = X_2
        self.y = y
        self.naive_forecast = naive_forecast
        self.xgb_tuning_params = xgb_tuning_params

    def run(self, num_samples=10, train_len=365, test_len=7, error_metric=metrics.mean_absolute_error,
            plotting=True):
        """Run the validation
        Parameters
        ------------
        num_samples : int
            number of forecasts to be attempted across the training set, note these are not necessarily
            independent as window shifts by one step fore each sample, Default=10
        train_len : int
            number of days to use to train model. Default=365
        test_len : int
            number of days to forecast ahead. Default=7
        error_metric : sklearn.metric
            a valid metric from sklearn.metric. Default=mean_absolute_error
        plotting : Bool
            If True, plot 5 randomly selected train test windows. Default=True"""

        max_num_samples = self.get_max_num_samples(train_len,test_len)
        if num_samples > max_num_samples:
            raise ValueError(f"max_num_samples is {max_num_samples} but {num_samples} were requested")

        print(f"Running validation for \n"
              f"train_len: {train_len} \n"
              f"test_len: {test_len} \n"
              f"num_samples: {num_samples} \n"
              f"error_metric: {error_metric.__name__}")

        if plotting:
            iterations_to_plot = np.random.choice(np.arange(num_samples), size=min(num_samples, 4), replace=False)
        else:
            iterations_to_plot = []

        # set up error calcs
        error_train_boost = []
        error_train = []
        error_train_naive = []
        error_test_boost = []
        error_test = []
        error_test_naive = []

        samples_df = self.get_train_test_split_indices(num_samples=num_samples, train_len=train_len, test_len=test_len)

        # Instantiate boosted model class
        model = bh.BoostedHybrid(model_1=LinearRegression(),
                                 model_2=XGBRegressor(**self.xgb_tuning_params))

        # Loop over all train/test samples
        for iteration, idx in tqdm(enumerate(samples_df.index)):

            X_1_train = self.X_1.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx], :]
            X_2_train = self.X_2.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx], :]
            y_train = self.y.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx]]
            naive_forecast_train = self.naive_forecast.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx]]

            X_1_test = self.X_1.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx], :]
            X_2_test = self.X_2.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx], :]
            y_test = self.y.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx]]
            naive_forecast_test = self.naive_forecast.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx]]

            # Fit the model on training data and get self predict
            model.fit(X_1_train, X_2_train, y_train)
            y_fit = model.predict(X_1_train, X_2_train)
            y_fit = y_fit.clip(0.0)

            # Apply model to test data and get forecast
            y_forecast = model.predict(X_1_test, X_2_test)
            y_forecast = y_forecast.clip(0.0)

            # Get errors for the self predict period
            get_error(error_train, error_metric, y_train, y_fit.y_pred)
            get_error(error_train_boost, error_metric, y_train, y_fit.y_pred_boosted)
            get_error(error_train_naive, error_metric, y_train, naive_forecast_train)

            # Get errors for the forecast period
            get_error(error_test, error_metric, y_test, y_forecast.y_pred)
            get_error(error_test_boost, error_metric, y_test, y_forecast.y_pred_boosted)
            get_error(error_test_naive, error_metric, y_test, naive_forecast_test)

            if iteration in iterations_to_plot:
                plot_samples(y_test, y_forecast, naive_forecast_test, y_train, y_fit, iteration)

        error_df = pd.DataFrame(data={f"{error_metric.__name__}_train_xgb": error_train_boost,
                                      f"{error_metric.__name__}_train_reg": error_train,
                                      f"{error_metric.__name__}_train_naive_forecast": error_train_naive,
                                      f"{error_metric.__name__}_test_xgb": error_test_boost,
                                      f"{error_metric.__name__}_test_reg": error_test,
                                      f"{error_metric.__name__}_test_naive_forecast": error_test_naive})

        return pd.concat([samples_df, error_df], axis=1, ignore_index=False)

    def get_train_test_split_indices(self, num_samples, train_len, test_len):
        train_start = []
        train_end = []
        test_start = []
        test_end = []

        idx = self.y.index

        for i in range(num_samples):
            train_start.append(idx[i])
            train_end.append(idx[i + train_len - 1])
            test_start.append(idx[i + train_len])
            test_end.append(idx[i + train_len + test_len - 1])

        return pd.DataFrame(data={"train_start": train_start, "train_end": train_end,
                                  "test_start": test_start, "test_end": test_end})

    def get_max_num_samples(self, train_len, test_len):
        data_len = len(self.y)
        window_len = train_len + test_len
        return data_len - window_len


def get_error(list_to_append, error_metric, actual, predicted):

    list_to_append.append(error_metric(actual, predicted))


def plot_samples(y_test, y_forecast, naive_forecast, y_train, y_fit, iteration):

    # Plot train period
    ax = y_train.plot(alpha=1, figsize=(14, 5.5), label="Actual")
    ax = y_fit.y_pred.plot(ax=ax, linewidth=1, label="Reg_Predict", color="r", linestyle=":")
    ax = y_fit.y_pred_boosted.plot(ax=ax, linewidth=1, label="XGB_Predict", color="g", linestyle="--")
    ax.set_title(f"Train Validation for sample {iteration}")
    ax.set_ylabel("Energy (kWh)")
    plt.legend()
    plt.show()

    # Plot test/forecast period
    ax = y_test.plot(alpha=0.5, figsize=(14, 5.5), label="Actual", marker="+")
    ax = y_forecast.y_pred.plot(ax=ax, linewidth=2, label="Reg_Predict", color="r", linestyle=":", marker="*")
    ax = y_forecast.y_pred_boosted.plot(ax=ax, linewidth=2, label="XGB_Predict", color="g", linestyle="--", marker="o")
    ax = naive_forecast.plot(ax=ax, linewidth=2, label="Naive_Forecast", color="grey", linestyle="-.", marker="^")
    ax.set_title(f"Forecast Validation for sample {iteration}")
    ax.set_ylabel("Energy (kWh)")
    plt.legend()
    plt.show()




