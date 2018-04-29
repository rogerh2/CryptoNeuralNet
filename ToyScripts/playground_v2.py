from CryptoPredict.CryptoPredict import DataSet
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


test_data = DataSet("2018-04-12 21:00:00 EST", "2018-04-25 12:00:00 EST", days=1, google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['btc', 'eth', 'ltc'], time_units='hours')

def get_nth_hr_block(test_data, start_time, n=24, time_unit='hours'):
    if time_unit == 'days':
        time_delta = timedelta(days=n)
    elif time_unit == 'hours':
        time_delta = timedelta(hours=n)
    elif time_unit == 'minutes':
        time_delta = timedelta(minutes=n)

    data_block = test_data[(test_data.index >= start_time) & (test_data.index <= start_time+timedelta)]
    return data_block

def buy_trade_loop(x_point, y_point, m, max_iterations, iterable, delta, delta_end_point):
    if x_point > (delta_end_point - delta):
        n = 1
        for n in range(m, max_iterations):
            y_point = np.partition(iterable, n)[n]
            x_point = iterable.nsmallest(max_iterations-1).index[-1]
            if x_point < (delta_end_point - delta):
                break
        return x_point, y_point, n
    return x_point, y_point, 1

def sell_trade_loop(x_point, y_point, max_iterations, iterable, delta, delta_start_point):
    if x_point < (delta_start_point + delta):
        n = 1
        for n in range(1, max_iterations):
            y_point = np.partition(iterable, n)[n]
            x_point = iterable.nsmallest(max_iterations-1).index[-1]
            if x_point > (delta_start_point + delta):
                break
        return x_point, y_point, n
    return x_point, y_point, 1


def find_trades(data_block, trade_data_frame=None, n=12, m=4, max_iterations=5, time_unit='hours'):
    buy_y_point = np.min(data_block)
    buy_x_point = data_block.idxmin()

    if time_unit == 'days':
        time_delta = timedelta(days=n)
        buy_sell_delta = timedelta(days=m)
    elif time_unit == 'hours':
        time_delta = timedelta(hours=n)
        buy_sell_delta = timedelta(hours=m)
    else:
        time_delta = timedelta(minutes=n)
        buy_sell_delta = timedelta(minutes=m)

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, 1, max_iterations, data_block.LTC_open, time_delta, data_block.index[0] + time_delta)
    if buy_x_point > (np.max(data_block.index) - time_delta):
        return False

    data_after_buy = data_block[data_block.index > buy_x_point]
    sell_y_point = np.max(data_after_buy)
    sell_x_point = data_block.idxmax

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, n_buy, max_iterations, data_after_buy,
                                                     buy_sell_delta, sell_x_point)
    sell_x_point, sell_y_point, n_sell = sell_trade_loop(sell_x_point, sell_y_point, max_iterations, data_after_buy, buy_sell_delta, buy_x_point)

    #TODO add functionality to return the sell and buy moves in find_trades as a pandas dataframe

#TODO create function to find all trades and return them in pandas dataframe (not just two moves like find_trades)
