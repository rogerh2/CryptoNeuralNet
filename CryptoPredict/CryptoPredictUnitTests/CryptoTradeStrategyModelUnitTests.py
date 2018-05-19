import unittest
from CryptoPredict.CryptoPredict import CryptoTradeStrategyModel


class TestCryptoTradeStrategyModel(unittest.TestCase):

    def setUp(self):
        date_from = "2018-05-13 00:00:00 EST"
        date_to = "2018-05-17 00:15:00 EST"
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'minutes'
        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_price_from_2018-05-13_00:00:00_EST_to_2018-05-17_00:15:00_EST.pickle'

        self.strategy_model = CryptoTradeStrategyModel(date_from, date_to, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, data_set_path=pickle_path)

    def test_strategy_model_returns_correct_columns(self):
        price_columns, prediction_columns = self.strategy_model.create_test_price_columns()

        self.assertEqual(price_columns.shape, prediction_columns.shape)
        self.assertEqual(price_columns.shape[1], 10)

        self.assertFalse(all([ v == 0 for v in price_columns[0, ::]]))
        self.assertFalse(all([v == 0 for v in prediction_columns[1, ::]]))


if __name__ == '__main__':
    unittest.main()
