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

    # --Tests for Setting Up Data--

    def test_can_create_input_array_of_correct_shape_for_training(self):
        test_offset = 30
        pred_data = self.test_obj.create_formatted_data('test', forecast_offset=test_offset)
        input_arr = pred_data['input']
        self.assertSequenceEqual(input_arr.shape, (self.ref_len-test_offset, len(self.test_obj.data_obj.raw_data.columns) - 1, 1))

    def test_can_create_output_array_with_correct_values_for_training(self):
        test_offset = 30
        pred_qaul='high'
        pred_data = self.test_obj.create_formatted_data('train', forecast_offset=test_offset, predicted_quality=pred_qaul)
        output_vec = pred_data['output']
        np.testing.assert_array_equal(output_vec, self.test_obj.data_obj.raw_data['ETH_' + pred_qaul].values[test_offset::])

    def test_do_input_and_output_shapes_agree_for_training(self):
        test_offset = 30
        pred_data = self.test_obj.create_formatted_data('test', forecast_offset=test_offset)
        output_vec = pred_data['output']
        input_arr = pred_data['input']
        self.assertEqual(len(output_vec), input_arr.shape[0])

    def test_can_split_for_train_and_test(self):
        test_offset = 30
        pred_data = self.test_obj.create_formatted_data('train/test', forecast_offset=test_offset, train_test_split=0.3)

        test_output_vec = pred_data['output']
        test_input_arr = pred_data['input']
        train_output_vec = pred_data['training output']
        train_input_arr = pred_data['training input']

        self.assertEqual(len(test_output_vec), test_input_arr.shape[0], 'test input and output not consistent')
        self.assertEqual(len(train_output_vec), train_input_arr.shape[0], 'train input and output not consistent')

        self.assertEqual(len(test_output_vec), 892, 'test input and output not correct length')
        self.assertEqual(len(train_output_vec), 2079, 'train input and output not correct length')

    def test_can_create_input_array_of_correct_shape_for_prediction(self):
        pred_data = self.test_obj.create_formatted_data('forecast')
        input_arr = pred_data['input']
        self.assertSequenceEqual(input_arr.shape, (self.ref_len, len(self.test_obj.data_obj.raw_data.columns) - 1, 1))

    # --Tests for Model Functions--

    def test_can_object_train_model_and_return_history(self):
        pred_data = self.test_obj.create_formatted_data('test', forecast_offset=30)
        input_arr = pred_data['input']
        output_arr = pred_data['output']
        hist = self.test_obj.train_model(input_arr, output_arr)
        self.assertEqual(type(hist), keras.callbacks.History)

    #TODO add test for testing model



if __name__ == '__main__':
    unittest.main()
