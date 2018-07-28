import unittest
from CryptoPredict.CryptoPredict import NaiveTradingBot, DataSet
from ToyScripts.bellmantest import OptimalTradeStrategy
import pickle
import numpy as np
import pandas as pd
from ToyScripts.bellmantest import findoptimaltradestrategystochastic
import matplotlib.pyplot as plt


class TestOptimalTradeStrategy(unittest.TestCase):

    def setUp(self):
        plt.ion()
        hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_62epochs_30neuron1527097308.228338.h5'
        minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Current_Best_Model/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_80neurons_3epochs1532511217.103676.h5'

        naive_bot = NaiveTradingBot(hourly_model=hour_path, minute_model=minute_path,
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
        naive_bot.minute_cp.data_obj = self.data_obj
        self.prediction, self.test_output = naive_bot.minute_cp.test_model(did_train=False, show_plots=False)

    def test_returns_same_sellbool_as_backtest(self):
        prediction = self.prediction
        test_output = self.test_output
        start_ind = 550
        stop_ind = 600

        sell_bool, buy_bool = findoptimaltradestrategystochastic(prediction[(start_ind - 60):(stop_ind + 60), 0],
                                                                 test_output[(start_ind - 60):(stop_ind + 60), 0], 40,
                                                                 show_plots=False)
        bot_sell_bool = np.zeros(stop_ind - start_ind)

        for i in range(start_ind, stop_ind):
            strategy_obj = OptimalTradeStrategy(self.prediction[(i-330):(i+30), 0], self.test_output[(i-330):i, 0])
            strategy_obj.find_optimal_trade_strategy()

            if strategy_obj.sell_array[-2] == 1:
                bot_sell_bool[i - start_ind] = 1

        plt.figure()
        plt.plot(test_output[start_ind:stop_ind])
        plt.plot(np.nonzero(bot_sell_bool)[0], test_output[start_ind:stop_ind][np.nonzero(bot_sell_bool)[0]], 'rx')
        plt.title('Simulated Live Bot Sells')
        plt.figure()

        plt.plot(test_output[start_ind:stop_ind])
        plt.plot(np.nonzero(sell_bool[60:-60])[0], test_output[start_ind:stop_ind][np.nonzero(sell_bool[60:-60])[0]],
                 'rx')
        plt.title('Backtested Bot Sells')
        plt.show()
        plt.pause(30)


        bot_sell_bool = bot_sell_bool > 0

        self.assertEqual(np.sum(sell_bool[60:-60]), np.sum(bot_sell_bool))
        np.testing.assert_array_equal(sell_bool[60:-60], bot_sell_bool)


if __name__ == '__main__':
    unittest.main()
