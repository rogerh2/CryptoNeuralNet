from CryptoPredict.CryptoPredict import DataSet
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


def get_nth_hr_block(test_data, start_time, n=48, time_unit='hours'):
    if time_unit == 'days':
        time_delta = timedelta(days=n)
    elif time_unit == 'hours':
        time_delta = timedelta(hours=n)
    elif time_unit == 'minutes':
        time_delta = timedelta(minutes=n)

    start_date_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S %Z')
    data_block = test_data[(test_data.index >= start_date_time) & (test_data.index <= (start_date_time + time_delta))]
    return data_block

def buy_trade_loop(x_point, y_point, m, max_iterations, iterable, delta, delta_end_point):
    if x_point > (delta_end_point - delta):
        n = 1
        for n in range(m, max_iterations):
            y_point = np.partition(iterable.values.T, n)[::, n][0]
            x_point = iterable.nsmallest(max_iterations-1, columns='LTC_open').index[-1]
            if x_point < (delta_end_point - delta):
                break
        return x_point, y_point, n
    return x_point, y_point, 1

def sell_trade_loop(x_point, y_point, max_iterations, iterable, delta, delta_start_point):
    if x_point < (delta_start_point + delta):
        n = 1
        for n in range(1, max_iterations):
            y_point = np.partition(iterable.values.T, -n)[::, -n][0]
            x_point = iterable.nlargest(max_iterations-1, columns='LTC_open').index[-1]
            if x_point > (delta_start_point + delta):
                break
        return x_point, y_point, n
    return x_point, y_point, 1


def find_trades(data_block, trade_data_frame=None, n=24, m=4, max_iterations=5, time_unit='hours'):
    buy_y_point = np.min(data_block.values)
    buy_x_point = data_block.idxmin()[0]

    if time_unit == 'days':
        time_delta = pd.Timedelta(days=m)
        buy_sell_delta = pd.Timedelta(days=m)
    elif time_unit == 'hours':
        time_delta = pd.Timedelta(hours=m)
        buy_sell_delta = pd.Timedelta(hours=m)
    else:
        time_delta = pd.Timedelta(minutes=m)
        buy_sell_delta = pd.Timedelta(minutes=m)

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, 1, max_iterations, data_block, time_delta, data_block.index[-1])
    if buy_x_point > (np.max(data_block.index) - time_delta):
        return False

    data_after_buy = data_block[data_block.index > buy_x_point]
    sell_y_point = np.max(data_after_buy.values)
    sell_x_point = data_after_buy.idxmax()[0]

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, n_buy, max_iterations, data_after_buy,
                                                     buy_sell_delta, sell_x_point)
    sell_x_point, sell_y_point, n_sell = sell_trade_loop(sell_x_point, sell_y_point, max_iterations, data_after_buy, buy_sell_delta, buy_x_point)
    df = pd.DataFrame([[buy_x_point, buy_y_point, sell_x_point, sell_y_point]], columns=['Buy X', 'Buy Y', 'Sell X', 'Sell Y'])
    if trade_data_frame:
        final_data_frame = trade_data_frame.appened(df, ignore_index=True)
    else:
        final_data_frame = df
    return final_data_frame

#TODO create function to find all trades and return them in pandas dataframe (not just two moves like find_trades)

test_data = DataSet("2018-04-29 18:00:00 EST", "2018-04-30 18:00:00 EST", days=1, google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['btc', 'eth', 'ltc'], time_units='minutes')
test_data.create_arrays(type='price')
df = test_data.final_table.LTC_open

data_block = get_nth_hr_block(df, "2018-04-30 12:00:00 EST", n=120, time_unit='minutes')
data_frame_block = data_block.to_frame()
final_data_frame = find_trades(data_frame_block, n=60, m=10, max_iterations=5, time_unit='minutes')
print(final_data_frame)
#data_frame_block.plot()
#plt.show()