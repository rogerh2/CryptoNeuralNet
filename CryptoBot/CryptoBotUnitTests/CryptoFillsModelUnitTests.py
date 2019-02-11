import unittest
import pandas as pd
import numpy as np
import CryptoBot.CryptoForecast as cf


class CryptoFillsModelTestCase(unittest.TestCase):
    data_obj = cf.FormattedCoinbaseProData(historical_order_books_path=None, historical_fills_path=None)

    def test_does_create_formatted_input_data_with_one_order_book_and_no_fills(self):
        # This is needed for live runs and backtests
        historical_order_books_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/' \
                                      'UnitTestData/SYM_historical_order_books_20entries.csv'
        historical_order_books = pd.read_csv(historical_order_books_path)
        order_books = historical_order_books.iloc[-1::]
        self.data_obj.historical_order_books = order_books.reset_index(drop=True)

        data_dict = self.data_obj.format_data('forecast')
        data = data_dict['input']

        self.assertFalse(np.isnan(data).any())
        self.assertEqual(len(data[0, ::]), 120)


if __name__ == '__main__':
    unittest.main()
