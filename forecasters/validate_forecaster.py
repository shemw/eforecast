import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from forecasters import boosted_hybrid as bh



class ValidateForecaster:
    """Loops over a number of train/test periods, contiguous in time, and stores error metrics for each run"""

    def __init__(self, X_1, X_2, y, xgb_tuning_params={}):
        self.X_1 = X_1
        self.X_2 = X_2
        self.y = y
        self.xgb_tuning_params = xgb_tuning_params

    def run(self, num_samples=10, train_len=365, test_len=7, error_metric=metrics.mean_absolute_error,
            plotting=True):

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
        error_test_boost = []
        error_test = []

        samples_df = self.get_train_test_split_indices(num_samples=num_samples, train_len=train_len, test_len=test_len)

        # Instantiate boosted model class
        model = bh.BoostedHybrid(model_1=LinearRegression(),
                                 model_2=XGBRegressor(**self.xgb_tuning_params))

        # Loop over all train/test samples
        for iteration, idx in tqdm(enumerate(samples_df.index)):

            X_1_train = self.X_1.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx], :]
            X_2_train = self.X_2.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx], :]
            y_train = self.y.loc[samples_df["train_start"][idx]: samples_df["train_end"][idx]]

            X_1_test = self.X_1.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx], :]
            X_2_test = self.X_2.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx], :]
            y_test = self.y.loc[samples_df["test_start"][idx]: samples_df["test_end"][idx]]

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

            # Get errors for the forecast period
            get_error(error_test, error_metric, y_test, y_forecast.y_pred)
            get_error(error_test_boost, error_metric, y_test, y_forecast.y_pred_boosted)

            if iteration in iterations_to_plot:
                plot_samples(y_test, y_forecast, y_train, y_fit, iteration)

        error_df = pd.DataFrame(data={f"{error_metric.__name__}_train_xgb": error_train_boost,
                                      f"{error_metric.__name__}_train_reg": error_train,
                                      f"{error_metric.__name__}_test_xgb": error_test_boost,
                                      f"{error_metric.__name__}_test_reg": error_test})

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


def plot_samples(y_test, y_forecast, y_train, y_fit, iteration):

    ax = y_train.plot(alpha=1, figsize=(14, 5.5), label="Actual")
    ax = y_fit.y_pred.plot(ax=ax, linewidth=1, label="Reg_Predict", color="r", linestyle=":")
    ax = y_fit.y_pred_boosted.plot(ax=ax, linewidth=1, label="XGB_Predict", color="g", linestyle="--")
    ax.set_title(f"Train Validation for sample {iteration}")
    ax.set_ylabel("Energy (kWh)")
    plt.legend()
    plt.show()

    ax = y_test.plot(alpha=0.5, figsize=(14, 5.5), label="Actual", marker="+")
    ax = y_forecast.y_pred.plot(ax=ax, linewidth=2, label="Reg_Predict", color="r", linestyle=":", marker="*")
    ax = y_forecast.y_pred_boosted.plot(ax=ax, linewidth=2, label="XGB_Predict", color="g", linestyle="--", marker="o")
    ax.set_title(f"Forecast Validation for sample {iteration}")
    ax.set_ylabel("Energy (kWh)")
    plt.legend()
    plt.show()




