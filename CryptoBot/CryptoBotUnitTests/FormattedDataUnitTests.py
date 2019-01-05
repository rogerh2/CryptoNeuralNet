import unittest
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from CryptoBot.CryptoForecast import FormattedData
from CryptoBot.CryptoBot_Shared_Functions import get_current_tz
from datetime import datetime
from datetime import timedelta



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

    def test_can_create_datetime_arr_of_correct_shape_for_plots(self):
        test_offset = 30
        pred_data = self.test_obj.format_data('train', forecast_offset=test_offset)
        label_vec = pred_data['x labels']
        output_vec = pred_data['output']
        self.assertEqual(len(label_vec),len(output_vec))

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

    def test_does_save_correct_raw_data(self):
        self.test_obj.save_raw_data()
        with open(self.ref_file_name, 'rb') as ref_file:
            saved_table = pickle.load(ref_file)

        pd.testing.assert_frame_equal(saved_table, self.test_obj.raw_data)

    # --Tests for Formatting Historical Order Book Data--

    def test_does_convert_list_of_strs_to_timestamps(self):
        fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
        time_strs = ['2018-12-30T23:24:03.242Z', '2018-12-30T23:27:37.905Z', '2018-12-30T23:29:09.619Z',
                     '2018-12-30T23:30:56.929Z', '2018-12-30T23:31:52.7Z', '2018-12-30T23:33:28.299Z',
                     '2018-12-30T23:33:58.814Z', '2018-12-30T23:34:20.894Z', '2018-12-30T23:35:10.975Z',
                     '2018-12-30T23:35:43.04Z']
        true_answer = np.array([1546212243.242, 1546212457.905, 1546212549.619, 1546212656.929, 1546212712.7, 1546212808.299, 1546212838.814, 1546212860.894, 1546212910.975, 1546212943.04])
        ans = self.test_obj.str_list_to_timestamp(time_strs)

        for i in range(0, len(true_answer)):
            self.assertEqual(ans[i], true_answer[i])

    def test_does_normalize_fills_according_to_first_bid(self):
        test_order_book = pd.DataFrame({'ts': [1, 2, 3, 4, 5, 6], '3': [1, 2, 3, 4, 5, 6], '6': [2, 4, 6, 8, 10, 12]})
        test_fills = pd.DataFrame({'time': [2.5, 2.75, 3.5, 5.5], 'price': [2, 4, 9, 20]})
        true_ans = np.array([2, 1, 3, 5, 4])
        ans, _ = self.test_obj.normalize_fill_array(test_order_book, test_fills)

        np.testing.assert_array_equal(ans, true_ans)

    def test_does_normalize_order_book_row_according_to_frist_bid(self):
        test_order_book = pd.DataFrame({'ts': [1, 2, 3, 4, 5, 6], '0': [4, 8, 12, 16, 20, 12], '1': [4, 8, 12, 16, 20, 12], '2': [4, 8, 12, 16, 20, 12], '3': [2, 4, 6, 8, 10, 6], '4': [4, 8, 12, 16, 20, 12], '5': [4, 8, 12, 16, 20, 12], '6': [4, 8, 12, 16, 20, 12]})
        test_fills = pd.DataFrame({'time': [2.5, 2.75, 3.5, 5.5], 'price': [2, 4, 9, 20]})
        true_ans = np.array([2, 2, 2, 2, 2])
        _, full_ans = self.test_obj.normalize_fill_array(test_order_book, test_fills)
        ans = full_ans[::, 6]

        np.testing.assert_array_equal(ans, true_ans)

    def tearDown(self):
        #This deletes test files
        if Path(self.ref_file_name).is_file():
            os.remove(self.ref_file_name)





if __name__ == '__main__':
    unittest.main()
