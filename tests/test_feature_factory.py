
from unittest import TestCase
from unittest.mock import patch
from pandas.testing import assert_frame_equal
import pandas as pd

from forecasters import feature_factory as ff


class RunnerTest(TestCase):

    def setUp(self):

        sunrise = ['2022-01-01T08:00:00', '2022-01-01T08:30:00']
        sunset = ['2022-01-01T16:00:00', '2022-01-01T15:30:00']
        y = [1, 2]
        self.idx = pd.to_datetime(['2022-01-01', '2022-01-01T'])
        self.df_input = pd.DataFrame(data={"y": y, "sunrise": sunrise, "sunset": sunset}, index=self.idx)

        self.factory = ff.FeatureFactory(self.df_input)

    def test_create_daylength_features_returns_daylength_hrs(self):

        expected_df = pd.DataFrame(data={'daylength': [8.0, 7.0]}, index=self.idx)

        actual_df = self.factory.create_daylength_features("sunrise", "sunset").to_frame()
        assert_frame_equal(expected_df, actual_df)
