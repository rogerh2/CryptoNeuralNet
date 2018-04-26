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

#TODO finish find_trades
def find_trades(data_block, trade_data_frame=None, min_iterations=5, n=24, m=4, time_unit='hours'):
    buy_y_point = np.min(data_block)
    buy_x_point = data_block.idxmin()

    if time_unit == 'days':
        time_delta = timedelta(days=m)
    elif time_unit == 'hours':
        time_delta = timedelta(hours=m)
    elif time_unit == 'minutes':
        time_delta = timedelta(minutes=m)

    if buy_x_point > (np.max(data_block.index) - time_delta):
        for n in range(1, min_iterations):
            buy_y_point = np.partition(data_block.LTC_open, n)[n]
            buy_x_point = data_block.LTC_open.nsmallest(4).index[-1]
            if buy_x_point < (np.max(data_block.index) - time_delta):
                break
    if buy_x_point > (np.max(data_block.index) - time_delta):
        return False

    data_after_buy = data_block[data_block.index > buy_x_point]
    sell_y_point = np.max(data_block)
    sell_x_point = data_block.idxmax