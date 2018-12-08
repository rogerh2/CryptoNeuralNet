import unittest
import os
from pathlib import Path
from CryptoBot.CryptoForecast import FormattedData
from CryptoBot.CryptoBot_Shared_Functions import get_current_tz
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np


class DataFormatterTestCase(unittest.TestCase):

    ref_file_name = 'stand in'

    @classmethod #This is a decorator that denotes a class method
    def setUpClass(self):
        # setUpClass is a class method that runs before the class is setup
        super(DataFormatterTestCase, self).setUpClass()
        self.fmt = '%Y-%m-%d %H:%M:%S'
        date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00'
        minute_diff = 2999
        date_from = (datetime.now() - timedelta(minutes=minute_diff)).strftime('%Y-%m-%d %H:%M:') + '00'

        # ref len comes from the fact that an extra minute is added at the end of every scraped segment and the current
        # limite on the length of minute segments is 2000 (this is controlled by the CryptoCompare API and is subject
        # to change!!!)
        self.ref_len = int(minute_diff + np.floor(minute_diff/2000) + 1)


        self.test_obj = FormattedData(date_from, date_to, 'ETH', sym_list=['BTC', 'LTC'], time_units='min')
        self.test_obj.scrape_data()
        self.test_obj.merge_raw_data_frames()
        self.sentiment_col, self.count_col = self.test_obj.collect_news_counts_and_sentiments()

        tz = get_current_tz()
        ref_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/minbymin__ticker_ETH_aux_BTC,LTC__from_' \
                        + self.test_obj.date_from + tz + '_to_' + self.test_obj.date_to + tz + '.pickle'
        self.ref_file_name = ref_file_name.replace(' ', '_')


    # --Tests for Formatting News Data--

    def test_is_sentiment_is_the_correct_len(self):
        self.assertEqual(len(self.sentiment_col), self.ref_len)

    def test_is_count_is_the_correct_len(self):
        self.assertEqual(len(self.sentiment_col), len(self.count_col))

    def test_sentiment_are_data(self):
        self.assertFalse(np.isnan(self.sentiment_col).any())

    def test_count_are_data(self):
        self.assertFalse(np.isnan(self.count_col).any())

    def test_is_raw_news_data_scraped_with_enough_offset(self):
        # The raw news data should be scraped at times earlier than the price data because news is considered relevant
        # for multiple hours. If it is not scraped the initial counts and sentiments will be zero
        initial_news_ts = self.test_obj.raw_news[0]['published_on']
        from_ts = datetime.strptime(self.test_obj.date_from, self.fmt).timestamp()

        self.assertGreater(initial_news_ts, from_ts - self.test_obj.news_hourly_offset*3600)
        self.assertLess(initial_news_ts, from_ts)

    # --Tests for Creating Raw Data--

    def test_can_formatter_create_correct_price_data_columns_for_raw_data(self):
        for sym in ['ETH', 'BTC', 'LTC']:
            for att in ['high', 'low', 'open', 'close', 'volumeto', 'volumefrom']:
                self.assertIn(sym + '_' + att, self.test_obj.raw_data.columns)

        self.assertIn('Sentiment', self.test_obj.raw_data.columns)
        self.assertIn('Count', self.test_obj.raw_data.columns)

    def test_can_formatter_create_correct_number_of_rows_for_raw_data(self):
        self.assertEqual(len(self.test_obj.raw_data.ETH_high), self.ref_len)

    def test_raw_data_columns_do_not_contain_nan(self):
        for data_col in self.test_obj.raw_data.columns:
            data = self.test_obj.raw_data[data_col].values
            self.assertFalse(any(pd.isnull(data)))

    # --Tests for Formatting Data for Neural Net--

    def test_can_create_input_array_of_correct_shape_for_training(self):
        test_offset = 30
        pred_data = self.test_obj.format_data('test', forecast_offset=test_offset)
        input_arr = pred_data['input']
        self.assertSequenceEqual(input_arr.shape, (self.ref_len-test_offset, len(self.test_obj.raw_data.columns) - 1, 1))

    def test_can_create_output_array_with_correct_values_for_training(self):
        test_offset = 30
        pred_qaul='high'
        pred_data = self.test_obj.format_data('train', forecast_offset=test_offset, predicted_quality=pred_qaul)
        output_vec = pred_data['output']
        np.testing.assert_array_equal(output_vec, self.test_obj.raw_data['ETH_' + pred_qaul].values[test_offset::])

    def test_do_input_and_output_shapes_agree_for_training(self):
        test_offset = 30
        pred_data = self.test_obj.format_data('test', forecast_offset=test_offset)
        output_vec = pred_data['output']
        input_arr = pred_data['input']
        self.assertEqual(len(output_vec), input_arr.shape[0])

    def test_can_split_for_train_and_test(self):
        test_offset = 30
        pred_data = self.test_obj.format_data('train/test', forecast_offset=test_offset, train_test_split=0.3)

        test_output_vec = pred_data['output']
        test_input_arr = pred_data['input']
        train_output_vec = pred_data['training output']
        train_input_arr = pred_data['training input']

        self.assertEqual(len(test_output_vec), test_input_arr.shape[0], 'test input and output not consistent')
        self.assertEqual(len(train_output_vec), train_input_arr.shape[0], 'train input and output not consistent')

        self.assertEqual(len(test_output_vec), 892, 'test input and output not correct length')
        self.assertEqual(len(train_output_vec), 2079, 'train input and output not correct length')

    def test_can_create_input_array_of_correct_shape_for_prediction(self):
        pred_data = self.test_obj.format_data('forecast')
        input_arr = pred_data['input']
        self.assertSequenceEqual(input_arr.shape, (self.ref_len, len(self.test_obj.raw_data.columns) - 1, 1))

    # --Tests for Saving Historical Data--

    def test_does_save_raw_data_with_correct_file_name(self):

        self.test_obj.save_raw_data()
        self.assertTrue(Path(self.ref_file_name).is_file())

    #TODO add test to ensure file is saved properly

    @classmethod
    def tearDownClass(cls):
        #This deletes test files
        if Path(cls.ref_file_name).is_file():
            os.remove(cls.ref_file_name)





if __name__ == '__main__':
    unittest.main()
