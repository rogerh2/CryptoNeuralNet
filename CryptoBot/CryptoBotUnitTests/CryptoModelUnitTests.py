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




if __name__ == '__main__':
    unittest.main()
