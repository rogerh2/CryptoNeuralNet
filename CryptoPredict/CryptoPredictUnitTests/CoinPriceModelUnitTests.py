import unittest
import pandas as pd
import numpy as np
from CryptoPredict.CryptoPredict import CoinPriceModel


class MyTestCase(unittest.TestCase):

    def setUp(self):
        temp = "2018-05-05 00:00:00 EST"
        minute_len = 30
        prediction_ticker = 'ETH'
        bitinfo_list = ['eth']
        minute_model = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/test_model.h5'


        self.cp = CoinPriceModel(temp, temp, days=minute_len, prediction_ticker=prediction_ticker,
                                 bitinfo_list=bitinfo_list, time_units='minutes', model_path=minute_model,
                                 need_data_obj=False)

    def test_model_produce_prediction_frame_without_duplicates(self):
        full_minute_prediction, full_minute_price = self.cp.predict(time_units='minutes', show_plots=False)
        duplicated_price_sum = np.sum(full_minute_price.index.duplicated())

        self.assertEqual(duplicated_price_sum, 0)


if __name__ == '__main__':
    unittest.main()
