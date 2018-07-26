import unittest
from CryptoPredict.CryptoPredict import NaiveTradingBot, DataSet
import pickle


class NaiveBotUnitTests(unittest.TestCase):
    def setUp(self):
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

        self.data_obj = DataSet(date_from=date_from, date_to=date_to, prediction_length=self.prediction_length, bitinfo_list=['eth'], prediction_ticker='ETH', time_units=time_units, fin_table=saved_table, aggregate=1)

    def test_prepare_data_for_plotting_handles_no_trades(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
