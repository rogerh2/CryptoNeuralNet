import unittest
from CryptoBot.CryptoForecast import FormattedData
from CryptoBot.CryptoBot_Shared_Functions import convert_time_to_uct
from datetime import datetime
from datetime import timedelta
import pickle
import numpy as np


class DataFormatterTestCase(unittest.TestCase):

    def setUp(self):
        self.fmt = '%Y-%m-%d %H:%M:%S'
        date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00'
        date_from = (datetime.now() - timedelta(minutes=2999)).strftime('%Y-%m-%d %H:%M:') + '00'
        self.test_obj = FormattedData(date_from, date_to, 'ETH', sym_list=['BTC', 'LTC'], time_units='min', news_hourly_offset=5)
        self.test_obj.scrape_data()

    def test_can_formatter_create_correct_number_of_input_data_columns(self):
        #self.assertEqual(len(self.test_obj.raw_data.columns), 20)
        for sym in ['ETH', 'BTC', 'LTC']:
            for att in ['high', 'low', 'open', 'close', 'volumeto', 'volumefrom']:
                self.assertIn(sym + '_' + att, self.test_obj.raw_data.columns)

if __name__ == '__main__':
    unittest.main()
