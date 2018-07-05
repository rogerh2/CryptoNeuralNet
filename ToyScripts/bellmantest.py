import numpy as np
import matplotlib.pyplot as plt
import pickle
from CryptoPredict.CryptoPredict import find_trade_strategy_value


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
        gain = find_trade_strategy_value(buy_bool, sell_bool, data)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('$'+str(gain))
        plt.show()
    else:
        return sell_bool, buy_bool

if __name__ == '__main__':
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-07-02_09:46:00_EST.pickle'
    with open(pickle_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)


    price = saved_table.ETH_high.values
    findoptimaltradestrategy(price, show_plots=True)

