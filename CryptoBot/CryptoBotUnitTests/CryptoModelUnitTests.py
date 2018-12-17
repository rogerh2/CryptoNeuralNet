import unittest
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from CryptoBot.CryptoForecast import CryptoModel
from CryptoBot.CryptoBot_Shared_Functions import get_current_tz
import keras
from datetime import datetime
from datetime import timedelta

class CryptoModelTestCase(unittest.TestCase):
    @classmethod  # This is a decorator that denotes a class method
    def setUpClass(self):
        # setUpClass is a class method that runs before the class is setup
        super(CryptoModelTestCase, self).setUpClass()
        self.fmt = '%Y-%m-%d %H:%M:%S'
        date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00'
        minute_diff = 2999
        self.ref_len = 3001
        date_from = (datetime.now() - timedelta(minutes=minute_diff)).strftime('%Y-%m-%d %H:%M:') + '00'

        self.test_obj = CryptoModel(date_from, date_to, 'ETH')
        self.test_obj.create_formatted_data_obj()

    # --Tests for Model Functions--

    def test_can_object_train_model_and_return_history(self):
        hist = self.test_obj.model_actions('train', show_plots=False)
        self.assertEqual(type(hist), keras.callbacks.History)

    def test_can_object_create_clean_test_and_reference_output_data(self):
        _ = self.test_obj.model_actions('train', show_plots=False)
        data = self.test_obj.model_actions('test', show_plots=False)
        prediction = data['predicted']
        test_output = data['actual']

        self.assertEqual(len(prediction), len(test_output))

    # --Tests for Data Creation/Manipulation--

    def test_can_object_create_recent_dates_and_data_for_predictions(self):
        self.test_obj.create_formatted_data_obj(hourly_time_offset=1)
        date_to = datetime.strptime(self.test_obj.data_obj.date_to, self.fmt).timestamp()
        date_from = datetime.strptime(self.test_obj.data_obj.date_from, self.fmt).timestamp()
        self.assertEqual(date_to-3600, date_from)

    def test_can_object_update_data_obj(self):
        fmt = '%Y-%m-%d %H:%M:'
        new_date_to = datetime.now().strftime(fmt) + '00'
        date_to = (datetime.now() - timedelta(minutes=30)).strftime(fmt) + '00'
        date_from = (datetime.now() - timedelta(minutes=90)).strftime(fmt) + '00'

        test_obj = CryptoModel(date_from, date_to, 'ETH')
        test_obj.create_formatted_data_obj()
        test_obj.update_formatted_data(date_to=new_date_to)

        # This tests whether the time stamps from the updated dataobj matches the target
        ref_ts_1 = datetime.strptime(test_obj.data_obj.date_to, self.fmt).timestamp()
        ref_ts_2 = datetime.strptime(new_date_to, self.fmt).timestamp()
        ref_ts_3 = test_obj.data_obj.raw_data.values[-1][0].tz_localize('EST').timestamp()


        self.assertEqual(ref_ts_1, ref_ts_2, 'Incorrect date_to in new data_obj')
        self.assertEqual(ref_ts_1, ref_ts_3, 'Incorrect end date in new data_obj raw data')
        self.assertTrue(np.sum(test_obj.data_obj.raw_data.duplicated('date').values) == 0, 'New data contains duplicated dates')
        self.assertTrue(len(test_obj.data_obj.raw_data.date.values) == 91, 'New data added an incorrect number of columns')
        # TODO add test to ensure updated data frames don't skip indices





if __name__ == '__main__':
    unittest.main()
