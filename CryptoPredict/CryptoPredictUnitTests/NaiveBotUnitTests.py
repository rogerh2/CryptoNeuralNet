import unittest
from CryptoPredict.CryptoPredict import NaiveTradingBot, DataSet
from ToyScripts.bellmantest import OptimalTradeStrategy
import pickle
import numpy as np
import pandas as pd
from ToyScripts.bellmantest import findoptimaltradestrategystochastic
import matplotlib.pyplot as plt


class NaiveBotUnitTests(unittest.TestCase):
    #TODO rewrite in to work like OptimalTradeStrategyTests
    def setUp(self):
        plt.ion()
        hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_62epochs_30neuron1527097308.228338.h5'
        minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Current_Best_Model/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_80neurons_3epochs1532511217.103676.h5'

        self.naive_bot = NaiveTradingBot(hourly_model=hour_path, minute_model=minute_path,
                                    api_key='redacted',
                                    secret_key='redacted',
                                    passphrase='redacted', is_sandbox_api=True, minute_len=30)

        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-07-24_10:17:00_UTC_to_2018-07-24_22:17:00_UTC.pickle'
        date_from = '2018-07-24 10:17:00 UTC'
        date_to = '2018-07-24 22:17:00 UTC'
        time_units = 'minutes'

        with open(pickle_path, 'rb') as ds_file:
            saved_table = pickle.load(ds_file)

        self.data_obj = DataSet(date_from=date_from, date_to=date_to, prediction_length=30, bitinfo_list=['eth'], prediction_ticker='ETH', time_units=time_units, fin_table=saved_table, aggregate=1)
        self.naive_bot.minute_cp.data_obj = self.data_obj
        self.prediction, self.test_output = self.naive_bot.minute_cp.test_model(did_train=False, show_plots=False)

    def test_prepare_data_for_plotting_handles_no_trades(self):

        trade_inds = []

        inds, values = self.naive_bot.prepare_data_for_plotting(self.prediction, self.test_output, trade_inds)

        self.assertEqual(len(inds), len(values))
        self.assertNotEqual(len(inds), 0)

    def test_prepare_data_for_plotting_handles_no_future_trades(self):

        prediction = self.prediction[30:120]
        prices = self.test_output[0:90]

        trade_arr = np.zeros(120)
        trade_inds = np.array([10, 37, 85])
        trade_arr[trade_inds] = 1
        trade_arr = trade_arr > 0

        inds, values = self.naive_bot.prepare_data_for_plotting(prediction, prices, trade_arr)

        self.assertEqual(len(inds), len(values))
        np.testing.assert_array_equal(values, prices[trade_inds])

    def test_prepare_data_for_plotting_handles_no_past_trades(self):

        prediction = self.prediction[30:120]
        prices = self.test_output[0:90]

        trade_arr = np.zeros(120)
        trade_inds = np.array([100, 105, 113])
        trade_arr[trade_inds] = 1
        trade_arr = trade_arr > 0

        inds, values = self.naive_bot.prepare_data_for_plotting(prediction, prices, trade_arr)

        self.assertEqual(len(inds), len(values))
        np.testing.assert_array_equal(values, prediction[trade_inds-30])

    def test_prepare_data_for_plotting_handles_past_and_future_trades(self):

        prediction = self.prediction[30:120]
        prices = self.test_output[0:90]

        trade_arr = np.zeros(120)
        trade_inds = np.array([10, 37, 85, 100, 105, 113])
        trade_arr[trade_inds] = 1
        trade_arr = trade_arr > 0

        inds, values = self.naive_bot.prepare_data_for_plotting(prediction, prices, trade_arr)

        self.assertEqual(len(inds), len(values))
        np.testing.assert_array_equal(values, np.concatenate((prices[trade_inds[0:3]], prediction[trade_inds[3::]-30])))

    def test_jump_in_minute_price_prediction_returns_same_buybool_as_backtest(self):
        prediction = self.prediction
        test_output = self.test_output
        start_ind = 500
        stop_ind = 600

        strategy_obj_backtest = OptimalTradeStrategy(prediction[(start_ind - 60):(stop_ind + 90), 0],
                                                     test_output[(start_ind - 60):(stop_ind + 60), 0])
        strategy_obj_backtest.find_optimal_trade_strategy()
        buy_bool = strategy_obj_backtest.buy_array

        bot_buy_bool = np.zeros(len(prediction))

        for i in range(start_ind, stop_ind):
            self.naive_bot.minute_prediction = pd.DataFrame(data=prediction[(i - 60):(i + 30)])
            self.naive_bot.minute_price = pd.DataFrame(data=test_output[(i - 90):(i)])
            jump_bool, jump_ind = self.naive_bot.is_peak_in_minute_price_prediction(1, show_plots=False)

            if jump_ind:
                if jump_ind < 2:
                    bot_buy_bool[i] = jump_bool

        bot_buy_bool = bot_buy_bool > 0

        np.testing.assert_array_equal(buy_bool[60:-61], bot_buy_bool[start_ind:stop_ind])

    def test_jump_in_minute_price_prediction_returns_same_sellbool_as_backtest(self):
        prediction = self.prediction
        test_output = self.test_output
        start_ind = 500
        stop_ind = 600

        strategy_obj_backtest = OptimalTradeStrategy(prediction[(start_ind - 60):(stop_ind + 90), 0],
                                                     test_output[(start_ind - 60):(stop_ind + 60), 0])
        strategy_obj_backtest.find_optimal_trade_strategy()
        sell_bool = strategy_obj_backtest.sell_array

        bot_sell_bool = np.zeros(len(prediction))

        for i in range(start_ind, stop_ind):
            self.naive_bot.minute_prediction = pd.DataFrame(data=prediction[(i-60):(i+30)])
            self.naive_bot.minute_price = pd.DataFrame(data=test_output[(i-90):(i)])
            jump_bool, jump_ind = self.naive_bot.is_peak_in_minute_price_prediction(-1, show_plots=False)

            if jump_ind:
                if jump_ind < 2:
                    bot_sell_bool[i] = jump_bool

        bot_sell_bool = bot_sell_bool > 0

        np.testing.assert_array_equal(sell_bool[60:-61], bot_sell_bool[start_ind:stop_ind])






if __name__ == '__main__':
    unittest.main()