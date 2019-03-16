import unittest
import numpy as np
from CryptoBot.CryptoStrategies import Strategy
from CryptoBot.BackTest import BackTestBot


class BackTestBotTestCase(unittest.TestCase):

    def setUp(self):
        strategy = Strategy()
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/back_test_model.h5'
        historical_order_books_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/SYM_historical_order_books_20entries.csv'
        historical_fills_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/SYM_fills_20entries.csv'
        self.bot = BackTestBot(model_path, strategy)
        self.bot.load_model_data(historical_order_books_path, historical_fills_path, 1)

    def test_does_get_order_book_and_holds_only_30(self):
        start_ind = 0
        stop_ind = 33
        ref_start_ind = stop_ind - 30 + start_ind

        for t in range(start_ind, stop_ind):
            self.bot.portfolio.exchange.time = t
            _ = self.bot.get_order_book()

        bot_order_books = self.bot.order_books
        true_order_books = self.bot.portfolio.exchange.order_books

        bot_values = bot_order_books['0'].values
        true_values = true_order_books['0'].values[ref_start_ind:stop_ind]


        np.testing.assert_array_equal(bot_values, true_values)

    def test_does_place_order(self):
        price = 161.30 # top bid price is 161.33 for the first time
        size = 100/price
        side = 'bids'

        self.bot.place_order(price, side, size)
        order = self.bot.portfolio.exchange.orders['bids'][0]

        self.assertEqual(order['size'], size, 'incorrect size')
        self.assertEqual(order['price'], price, 'incorrect price')
        self.assertFalse(order['filled'], 'placed filled order')

    def test_does_get_correct_portfolio_value_with_just_usd(self):
        ref_val = 93
        test_value = {'USD': ref_val, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
        self.bot.portfolio.value = test_value

        val = self.bot.get_full_portfolio_value()
        self.assertEqual(ref_val, val)

    def test_does_get_correct_portfolio_value_with_just_sym(self):
        ref_val = 93
        bid_price = self.bot.portfolio.exchange.order_books['0'].values[0]
        ask_price = self.bot.portfolio.exchange.order_books['60'].values[0]
        price = (bid_price + ask_price)/2
        test_value = {'USD': 0, 'SYM': ref_val/price, 'USD Hold': 0, 'SYM Hold': 0}
        self.bot.portfolio.value = test_value
        val = self.bot.get_full_portfolio_value()

        self.assertAlmostEqual(ref_val, val, 2)







if __name__ == '__main__':
    unittest.main()
