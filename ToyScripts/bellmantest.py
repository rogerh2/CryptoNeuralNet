import numpy as np
import matplotlib.pyplot as plt
import pickle
from CryptoPredict.CryptoPredict import find_trade_strategy_value
from  CryptoPredict.CryptoPredict import CoinPriceModel


def findoptimaltradestrategy(data, show_plots=False, min_price_jump = 1.0001):
    buy_array = np.zeros(len(data))
    sell_array = np.zeros(len(data))
    all_times = np.arange(0,len(data))
    data_len = len(data)
    price_is_rising = None
    for i in range(data_len):
        ind = data_len - i -1
        current_price = data[ind]
        prior_price = data[ind-1]
        bool_price_test = current_price > prior_price

        if price_is_rising is None:
            price_is_rising = bool_price_test
            last_inflection_price = current_price
        else:
            if bool_price_test != price_is_rising: # != acts as an xor gate
                if price_is_rising:
                    if current_price*min_price_jump < last_inflection_price:
                        buy_array[ind] = 1
                else:
                    if current_price/min_price_jump > last_inflection_price:
                        sell_array[ind] = 1
                last_inflection_price = current_price
                price_is_rising = bool_price_test

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]
    if show_plots:
        returns = find_trade_strategy_value(buy_bool, sell_bool, data)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('Return of '+str(returns)+'%')
        plt.show()
    else:
        return sell_bool, buy_bool

def findoptimaltradestrategystochastic(prediction, data, offset, absolute_output, show_plots=True): #Cannot be copie pasted, this is a test
    buy_array = np.zeros(len(data))
    sell_array = np.zeros(len(data))
    all_times = np.arange(0, len(data))
    data_len = len(data)
    price_is_rising = None

    prediction = prediction - np.min(prediction)
    prediction = prediction/np.max(prediction)
    data = data - np.min(data)
    data = data/np.max(data)
    fuzziness = 5

    for i in range(fuzziness, data_len-offset):
        ind = data_len - i - 1
        past_predictions = prediction[(ind-offset):(ind)]
        past_data = data[(ind-offset):(ind)]

        #Find error
        current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
        current_err = current_fit[1]

        #Find trades
        current_price = prediction[ind]
        prior_price = prediction[ind - 1]
        bool_price_test = current_price > prior_price


        if price_is_rising is None:
            price_is_rising = bool_price_test
            last_inflection_price = current_price
        else:
            if bool_price_test != price_is_rising:  # != acts as an xor gate
                if price_is_rising:
                    current_price = np.max(prediction[(ind - fuzziness):(ind + fuzziness)])
                    if (current_price+current_err) < last_inflection_price:
                        buy_array[ind] = 1
                        last_inflection_price = current_price+current_err
                    else:
                        last_inflection_price = current_price+current_err
                else:
                    current_price = np.min(prediction[(ind - fuzziness):(ind + fuzziness)])
                    if (current_price-current_err) > last_inflection_price:
                        sell_array[ind] = 1
                        last_inflection_price = current_price-current_err
                    else:
                        last_inflection_price = current_price-current_err

                price_is_rising = bool_price_test

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]
    if show_plots:
        returns = find_trade_strategy_value(buy_bool, sell_bool, absolute_output)
        plt.plot(all_times[sell_bool], absolute_output[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], absolute_output[buy_bool], 'gx')
        plt.plot(absolute_output)
        plt.title('Return of ' + str(returns) + '%')
        plt.show()

if __name__ == '__main__':
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-07-02_09:46:00_EST.pickle'
    #pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-07-05_20:29:00_EST_to_2018-07-05_21:35:00_EST.pickle'
    with open(pickle_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)

    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_4epochs1530856066.874304.h5'

    date_from = '2018-07-05 21:35:00 EST'
    date_to = '2018-07-06 01:22:00 EST'
    bitinfo_list = ['eth']
    cp = CoinPriceModel(date_from, date_to, days=30, prediction_ticker='ETH',
                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=model_path, need_data_obj=True,
                        data_set_path=pickle_path)
    #cp.test_model(did_train=False) #currently at 12.994%
    zerod_prediction, test_output = cp.test_model(did_train=False, show_plots=False)
    absolute_output = test_output[::, 0]
    zerod_output = absolute_output - np.mean(test_output[::, 0])
    zerod_output = zerod_output.reshape(1, len(zerod_output))
    zerod_output = zerod_output.T
    findoptimaltradestrategystochastic(zerod_prediction[::, 0], zerod_output[::, 0], 60, absolute_output, show_plots=True)

    price = saved_table.ETH_high.values
    findoptimaltradestrategy(price, show_plots=True)

