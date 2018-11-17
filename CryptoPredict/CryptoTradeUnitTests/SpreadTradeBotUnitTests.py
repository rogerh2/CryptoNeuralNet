import unittest
from ToyScripts.bellmantest import OptimalTradeStrategyV5
from CryptoPredict.CryptoTrade import SpreadTradeBot
from CryptoPredict.CryptoPredict import DataSet
import numpy as np
import pandas as pd
import pickle


class SpreadTradeBotUnitTests(unittest.TestCase):
    def setUp(self):
        self.order_book = {'sequence': 4948871033, 'bids': [['226.87', '75.54126341', 11], ['226.85', '20', 1], ['226.8', '1', 1], ['226.72', '20', 1], ['226.63', '6', 1], ['226.43', '6', 1], ['226.41', '6', 1], ['226.26', '1', 1], ['226.16', '6', 1], ['226.14', '19', 2], ['226.05', '43.959', 1], ['226.04', '20.35309838', 2], ['226', '217.69911504', 2], ['225.88', '30', 1], ['225.85', '68.689', 1], ['225.84', '76.5625', 1], ['225.71', '6', 1], ['225.69', '14.24321', 1], ['225.6', '0.0299', 1], ['225.49', '26.6', 1], ['225.41', '10.822522', 1], ['225.39', '300', 1], ['225.3', '16', 1], ['225.29', '40', 1], ['225.21', '20', 1], ['225.15', '12', 1], ['225.09', '23', 1], ['225.06', '0.03417071', 1], ['225', '0.125892', 1], ['224.99', '1', 1], ['224.94', '11.65', 1], ['224.92', '4.666565', 1], ['224.9', '116.911595', 2], ['224.87', '44', 1], ['224.7', '3.41038', 1], ['224.43', '15.740974', 1], ['224.4', '8.50931', 1], ['224.39', '162', 1], ['224.23', '0.074', 1], ['224.21', '10', 1], ['224.2', '10', 1], ['224.08', '0.0758', 1], ['224.01', '2', 1], ['224', '24.44', 2], ['223.97', '0.06385481', 1], ['223.93', '0.074', 1], ['223.87', '10.042316', 1], ['223.86', '8.64175', 1], ['223.83', '2.44475', 1], ['223.8', '18.504556', 1]], 'asks': [['226.88', '2.43232095', 2], ['226.89', '181.16266694', 3], ['226.9', '0.018', 1], ['227.04', '0.09935', 1], ['227.05', '36.2464076', 1], ['227.16', '1.100562', 1], ['227.23', '0.16953535', 1], ['227.24', '6', 1], ['227.25', '58.42', 1], ['227.29', '1', 1], ['227.34', '0.7687008', 1], ['227.45', '66.573', 2], ['227.59', '4.95221237', 2], ['227.66', '58.1', 2], ['227.69', '11.83057', 2], ['227.72', '0.1', 1], ['227.82', '9.51082', 1], ['227.85', '0.1', 1], ['227.86', '19.6', 1], ['227.9', '0.01', 1], ['228', '40.04', 9], ['228.04', '0.7', 1], ['228.25', '15', 1], ['228.28', '0.02', 1], ['228.29', '1.095087', 1], ['228.35', '0.294', 1], ['228.41', '0.29496565', 2], ['228.45', '0.503', 1], ['228.48', '0.05291237', 1], ['228.51', '30', 1], ['228.55', '29.06070408', 1], ['228.61', '0.038', 1], ['228.64', '7.01', 2], ['228.67', '0.01', 1], ['228.7', '0.6999976', 1], ['228.73', '0.03801335', 1], ['228.9', '1', 1], ['228.98', '2.22491116', 2], ['229', '4.072', 6], ['229.2', '21', 1], ['229.3', '12', 1], ['229.31', '0.012', 1], ['229.32', '5', 1], ['229.33', '0.0867145', 1], ['229.36', '1', 1], ['229.43', '1.089638', 1], ['229.44', '66.91250252', 1], ['229.52', '54.81', 1], ['229.53', '254.09773', 1], ['229.76', '0.05444364', 1]]}

        minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Live_Tested_Models/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_2epochs1534230422.515854.h5'

        self.spread_bot = SpreadTradeBot(minute_model=minute_path, api_key='redacted',
                                    secret_key='redacted',
                                    passphrase='redacted', is_sandbox_api=True, minute_len=30)

        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-11-06_21:30:00_EST.pickle'
        date_from = '2018-08-11 08:46:00 EST'
        date_to = '2018-08-16 08:00:00 EST'
        time_units = 'minutes'

        with open(pickle_path, 'rb') as ds_file:
            saved_table = pickle.load(ds_file)

        self.data_obj = DataSet(date_from=date_from, date_to=date_to, prediction_length=30, bitinfo_list=['eth'], prediction_ticker='ETH', time_units=time_units, fin_table=saved_table, aggregate=1)
        self.spread_bot.cp.data_obj = self.data_obj
        self.prediction, self.test_output = self.spread_bot.cp.test_model(did_train=False, show_plots=False)

    def test_does_price_loop_return_correct_bids(self):
        min_price = 226.42
        max_price = 226.83
        prices = self.order_book['bids']
        num_bids = 5

        price_step = np.round((max_price - min_price) / num_bids, 2)

        test_bid_array = np.arange(min_price, max_price, price_step)

        bid_array = self.spread_bot.price_loop(prices, max_price, min_price, num_bids, 'buy')

        np.testing.assert_array_equal(bid_array, test_bid_array)

    def test_does_price_loop_return_correct_asks(self):
        min_price = 226.8
        max_price = 227.05
        prices = self.order_book['asks']
        num_bids = 5

        price_step = np.round((max_price - 226.88) / num_bids, 2)

        test_ask_array = np.arange(226.88, max_price, price_step)

        ask_array = self.spread_bot.price_loop(prices, max_price, min_price, num_bids, 'sell')

        np.testing.assert_array_equal(ask_array, test_ask_array)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_small_amount_to_buy(self):
        # This test ensures the find_trade_size_and_number method returns sizes that allow most of the available balance
        # to be used when given a size that can be easily used with the minnimum price ($10) and spread
        err = 0.4
        available = 73.49
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'buy')

        self.assertEqual(num_orders, 7)
        self.assertEqual(size, 10.49)
        np.testing.assert_almost_equal(available, num_orders*size, 0.01)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_large_amount_to_buy(self):
        # This test ensures the find_trade_size_and_number method returns sizes that allow most of the available balance
        # to be used when given a size that can be used with more than the minnimum price ($10) and spread
        err = 0.4
        available = 803.49
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'buy')

        self.assertEqual(num_orders, 18)
        self.assertEqual(size, 44.63)
        np.testing.assert_almost_equal(available, num_orders * size, 0.01)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_too_much_to_buy(self):
        # This test ensures the find_trade_size_and_number method returns sizes that do not go over the max price when
        # given too much to buy ($5000) and spread
        err = 0.4
        available = 101000
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'buy')

        self.assertEqual(num_orders, 20)
        self.assertEqual(size, 4999.99)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_small_amount_to_sell(self):
        # This test ensures the find_trade_size_and_number method returns sizes that allow most of the available balance
        # to be used when given a size that can be easily used with the minnimum price (E 0.05) and spread
        err = 0.4
        available = 0.14596839
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'sell')

        self.assertEqual(num_orders, 14)
        self.assertEqual(size, 0.01042630)
        np.testing.assert_almost_equal(available, num_orders * size, 0.01)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_large_amount_to_sell(self):
        # This test ensures the find_trade_size_and_number method returns sizes that allow most of the available balance
        # to be used when given a size that can be used with more than the minnimum price (E 0.05) and spread
        err = 0.4
        available = 2.14789241
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'sell')

        self.assertEqual(num_orders, 20)
        self.assertAlmostEqual(size, 0.10739462, 7)
        np.testing.assert_almost_equal(available, num_orders * size, 0.01)

    def test_does_find_trade_size_and_number_return_correct_values_when_given_too_much_to_sell(self):
        # This test ensures the find_trade_size_and_number method returns sizes that do not go over the max price when
        # given too much Ethereum to sell (E1 * $5000/$200 in this case) and spread
        err = 0.4
        available = 27.98986221
        current_price = 200
        size, num_orders = self.spread_bot.find_trade_size_and_number(err, available, current_price, 'sell')

        self.assertEqual(num_orders, 20)
        self.assertAlmostEqual(size, 1.3994931, 7)
        np.testing.assert_almost_equal(available, num_orders * size, 0.01)

    def test_does_bot_make_correct_predictions(self):
        prediction = self.prediction
        test_output = self.test_output
        start_ind = 968 + 3804
        stop_ind = 1068 + 3804
        backtest_padding = 50

        strategy_obj_backtest = OptimalTradeStrategyV5(
            prediction[(start_ind - backtest_padding):(stop_ind + backtest_padding + 30), 0],
            test_output[(start_ind - backtest_padding):(stop_ind + backtest_padding), 0])
        strategy_obj_backtest.find_optimal_trade_strategy()
        buy_bool = strategy_obj_backtest.buy_array
        sell_bool = strategy_obj_backtest.sell_array

        bot_buy_bool = np.zeros(len(prediction))
        bot_sell_bool = np.zeros(len(prediction))

        for i in range(start_ind, stop_ind):
            jump_bool = 0
            self.spread_bot.price = test_output[(i - 60):(i), 0]
            self.spread_bot.prediction = prediction[(i - 60):(i + 30), 0]
            err, fit_coeff, fit_offset, const_diff, fuzziness = self.spread_bot.find_fit_info()
            sell_a, sell_b = self.spread_bot.find_expected_value(err, False, const_diff, fit_coeff, fuzziness, fit_offset)
            buy_a, buy_b = self.spread_bot.find_expected_value(err, True, const_diff, fit_coeff, fuzziness, fit_offset)
            test_buy = strategy_obj_backtest.find_expected_value_over_many_trades(i + backtest_padding - start_ind, err,
                                                                                  True, const_diff, fit_coeff,
                                                                                  fuzziness, fit_offset)
            test_sell = strategy_obj_backtest.find_expected_value_over_many_trades(i + backtest_padding - start_ind, err,
                                                                                  False, const_diff, fit_coeff,
                                                                                  fuzziness, fit_offset)

            print(i-start_ind)
            self.assertEqual(test_buy, buy_a != -1)
            self.assertEqual(test_sell, sell_a != -1)


            if buy_a != sell_a:
                bot_buy_bool[i] = buy_a > 0
                bot_sell_bool[i] = sell_a > 0

        bot_buy_bool = bot_buy_bool > 0
        bot_sell_bool = bot_sell_bool > 0

        np.testing.assert_array_equal(buy_bool[backtest_padding:-(backtest_padding + 1)],
                                      bot_buy_bool[start_ind:stop_ind], 'incorrect buys')
        np.testing.assert_array_equal(sell_bool[backtest_padding:-(backtest_padding + 1)],
                                      bot_sell_bool[start_ind:stop_ind], 'incorrect sells')

    def test_can_CoinPriceModel_make_predictions(self):
        self.spread_bot.spread_bot_predict()

        nan_sum = np.sum(np.isnan(self.prediction))

        self.assertEqual(nan_sum, 0)


if __name__ == '__main__':
    unittest.main()
