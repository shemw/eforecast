# Example test suite

from unittest import TestCase
from unittest.mock import patch
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np

from forecasters import feature_factory as ff


class FeatureFactoryTest(TestCase):

    def setUp(self):

        sunrise = ['2022-01-01T08:00:00', '2022-01-01T08:30:00', '2022-01-01T08:30:00', '2022-01-01T08:30:00',
                   '2022-01-01T08:30:00']
        sunset = ['2022-01-01T16:00:00', '2022-01-01T15:30:00', '2022-01-01T08:30:00', '2022-01-01T08:30:00',
                  '2022-01-01T08:30:00']
        y = [2, 4, 6, 17, 20]
        self.idx = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-04'])
        self.df_input = pd.DataFrame(data={"y": y, "sunrise": sunrise, "sunset": sunset}, index=self.idx)

        self.factory = ff.FeatureFactory(self.df_input)

    def test__create_daylength_features__returns_daylength_hrs(self):

        expected_df = pd.DataFrame(data={'daylength': [8.0, 7.0, 0.0, 0.0, 0.0]}, index=self.idx)

        actual_df = self.factory.create_daylength_features("sunrise", "sunset").to_frame()
        assert_frame_equal(expected_df, actual_df)

    def test__create_statistical_features__returns_correct_values_with_lag_1(self):

        x_mean = [np.nan, np.nan, np.nan, 4.0, 9.0]
        x_median = [np.nan, np.nan, np.nan, 4.0, 6.0]
        x_std = [np.nan, np.nan, np.nan, 2.0, 7.0]

        expected_df = pd.DataFrame(data={'mean_3': x_mean, 'median_3': x_median, 'stdev_3': x_std},
                                   index=self.idx)

        actual_df = self.factory.create_statistical_features(mean_window=3,
                                                             median_window=3,
                                                             stdev_window=3,
                                                             lag=1)
        assert_frame_equal(expected_df, actual_df)

    def test__create_lag_features__returns_correct_lag(self):

        expected_df = pd.DataFrame(data={'ylag_2': [np.nan, np.nan, 2.0, 4.0, 6.0]},
                                   index=self.idx)

        actual_df = self.factory.create_lag_features(lag_steps=2).to_frame()
        assert_frame_equal(expected_df, actual_df)

#TODO add further tests for all forecaster modules