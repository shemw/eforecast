import pandas as pd

class BoostedHybrid:
    """Fit a simple lin regression model (model_1), subtract this from the targets and fit a boosted ML
    model (model_2) to predict the residuals."""

    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store col names from fit method

    # Define fit method for boosted hybrid

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_fit = pd.DataFrame(self.model_1.predict(X_1),
                             index=X_1.index,
                             columns=["y_fit"])

        y_resid = y.y - y_fit.y_fit

        # y_resid = y_resid.stack().squeeze()  # wide to long

        # fit model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # save col names for predict method
        #self.y_columns = y.columns

    def predict(self, X_1, X_2):
        # Define predict method for boosted hybrid

        y_pred = pd.DataFrame(self.model_1.predict(X_1),
                              index=X_1.index,
                              columns=["y_pred"])

        y_pred_residuals = pd.DataFrame(self.model_2.predict(X_2),
                                        index=X_1.index,
                                        columns=["y_pred_residuals"])

        # y_pred = y_pred.stack().squeeze()  # wide to long

        # add model_2 predictions
        # TODOD sort out this indexing in a better way
        y_pred_boosted = (y_pred.y_pred + y_pred_residuals.y_pred_residuals).rename("y_pred_boosted")

        y_predictions = pd.concat([y_pred, y_pred_boosted], axis=1, ignore_index=False)

        return y_predictions
