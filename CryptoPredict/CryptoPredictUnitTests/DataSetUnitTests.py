import unittest
from CryptoPredict.CryptoPredict import DataSet
import pickle
import numpy as np


class TestDataSet(unittest.TestCase):
    def setUp(self):
        date_from = "2018-05-15 09:00:00 EST"
        date_to = "2018-05-16 09:00:00 EST"
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'hours'

        self.data_obj = DataSet(date_from=date_from, date_to=date_to, days=6, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units)
        self.data_obj.create_arrays()

    def test_create_price_prediction_columns_creates_correct_number_of_columns_and(self):
        self.assertEqual(len(self.data_obj.final_table.columns), 9)

    def test_create_price_prediction_columns_creates_correct_number_of_rows(self):
        self.assertEqual(len(self.data_obj.final_table.index), 25)

    def test_dataset_creates_correct_predictions(self):
        self.assertEqual(self.data_obj.fin_table['ETH_open'][7], self.data_obj.output_array[1])

    def test_create_buysell_prediction_columns(self):
        date_from = "2018-05-13 00:00:00 EST"
        date_to = "2018-05-17 00:15:00 EST"
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'minutes'
        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_price_from_2018-05-13_00:00:00_EST_to_2018-05-17_00:15:00_EST.pickle'

        with open(pickle_path, 'rb') as ds_file:
            saved_table = pickle.load(ds_file)

        data_obj = DataSet(date_from=date_from, date_to=date_to, days=5, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, fin_table=saved_table)
        data_obj.create_arrays(model_type='buy&sell')
        num_buy_ones = np.sum([x == 1 for x in data_obj.final_table['Buy'].values])
        num_buy_zeros = np.sum([x == 0 for x in data_obj.final_table['Buy'].values])

        num_sell_ones = np.sum([x == 1 for x in data_obj.final_table['Sell'].values])
        num_sell_zeros = np.sum([x == 0 for x in data_obj.final_table['Sell'].values])

        #The array should only have ones and zeros
        self.assertEqual(num_buy_ones + num_buy_zeros, len(data_obj.final_table['Buy'].values))
        self.assertEqual(num_sell_ones + num_sell_zeros, len(data_obj.final_table['Sell'].values))

        #It should buy and sell an equal number of times
        self.assertEqual(num_sell_ones, num_buy_ones)


if __name__ == '__main__':
    unittest.main()
