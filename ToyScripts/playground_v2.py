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


def find_trades(data_block, trade_data_frame=None, n=4, max_iterations=5, time_unit='hours'):
    buy_y_point = np.min(data_block.values)
    buy_x_point = data_block.idxmin()[0]

    if time_unit == 'days':
        time_delta = pd.Timedelta(days=n)
        buy_sell_delta = pd.Timedelta(days=n)
    elif time_unit == 'hours':
        time_delta = pd.Timedelta(hours=n)
        buy_sell_delta = pd.Timedelta(hours=n)
    else:
        time_delta = pd.Timedelta(minutes=n)
        buy_sell_delta = pd.Timedelta(minutes=n)

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, 1, max_iterations, data_block, time_delta, data_block.index[-1])
    if buy_x_point > (np.max(data_block.index) - time_delta):
        return False, 0, 0

    data_after_buy = data_block[data_block.index > buy_x_point]
    sell_y_point = np.max(data_after_buy.values)
    sell_x_point = data_after_buy.idxmax()[0]

    buy_x_point, buy_y_point, n_buy = buy_trade_loop(buy_x_point, buy_y_point, n_buy, max_iterations, data_after_buy,
                                                     buy_sell_delta, sell_x_point)
    sell_x_point, sell_y_point, n_sell = sell_trade_loop(sell_x_point, sell_y_point, max_iterations, data_after_buy, buy_sell_delta, buy_x_point)
    df = pd.DataFrame([[buy_x_point, buy_y_point, sell_x_point, sell_y_point]], columns=['Buy_X', 'Buy_Y', 'Sell_X', 'Sell_Y'])
    if trade_data_frame is not None:
        final_data_frame = trade_data_frame.append(df, ignore_index=True)
    else:
        final_data_frame = df
    return final_data_frame, buy_y_point, sell_y_point

def timedelta_strunits(n, time_unit):
    if time_unit == 'days':
        time_delta = timedelta(days=n)
    elif time_unit == 'hours':
        time_delta = timedelta(hours=n)
    else:
        time_delta = timedelta(minutes=n)

    return time_delta

def find_all_trades(data_frame, time_unit='hours', n_0=48, m=6, max_iterations=5):

    start_time_str = datetime.strftime(data_frame.index[0], '%Y-%m-%d %H:%M:%S')
    start_time = data_frame.index[0]

    end_time = data_frame.index[-1]
    block_trades = None
    n = n_0
    time_delta_0 = timedelta_strunits(n, time_unit)
    search_range = start_time + time_delta_0

    while search_range < (end_time - time_delta_0):
        data_block = get_nth_hr_block(data_frame, start_time_str + ' EST', n=n, time_unit=time_unit)
        data_frame_block = data_block.to_frame()
        current_block_trades, current_buy, current_sell = find_trades(data_frame_block, trade_data_frame=block_trades, n=m, max_iterations=max_iterations, time_unit=time_unit)
        if (current_sell - current_buy) < 0.005*current_buy or type(current_block_trades) is not pd.DataFrame:
            n = n + n_0
            time_delta = timedelta_strunits(n, time_unit)
            search_range = search_range + time_delta
        else:
            time_delta = timedelta_strunits(n_0, time_unit)
            start_time_str = datetime.strftime(search_range, '%Y-%m-%d %H:%M:%S')
            search_range = search_range + time_delta
            block_trades = current_block_trades

    return  block_trades

def create_single_prediction_column(price_data_frame, trade_data_frame, time_units='hours'):

    if time_units == 'days':
        norm_constant = 24*3600
    elif time_units == 'hours':
        norm_constant = 3600
    else:
        norm_constant = 60

    loop_len = price_data_frame.count()
    trade_ind = 0
    trade_time = trade_data_frame[trade_ind]
    time_arr = None
    for price_ind in range(0, loop_len):
        price_time = price_data_frame.index[price_ind]
        if price_time > trade_time:
            trade_ind += 1
            if trade_ind >= trade_data_frame.count():
                return time_arr
            trade_time = trade_data_frame[trade_ind]

        time_to_trade = (trade_time - price_time).total_seconds()/norm_constant

        if time_arr is None:
            time_arr = np.array([time_to_trade])
        else:
            time_arr = np.vstack((time_arr, np.array([time_to_trade])))

def  create_buy_sell_prediction_columns(price_data_frame, buy_sell_data_frame, time_units='hours'):
    buy_column = create_single_prediction_column(price_data_frame, buy_sell_data_frame.Buy_X, time_units)
    sell_column = create_single_prediction_column(price_data_frame, buy_sell_data_frame.Sell_X, time_units)
    prediction_columns = np.hstack((buy_column, sell_column[0:len(buy_column)]))
    return prediction_columns

test_data = DataSet("2018-04-30 21:00:00 EST", "2018-05-01 20:30:00 EST", days=1, google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['btc', 'eth', 'ltc'], time_units='minutes')
test_data.create_arrays(model_type='price')
df = test_data.final_table.LTC_open

# data_block = get_nth_hr_block(df, "2018-04-30 12:00:00 EST", n=120, time_unit='minutes')
# data_frame_block = data_block.to_frame()
# final_data_frame = find_trades(data_frame_block, n=10, max_iterations=5, time_unit='minutes')

final_data_frame = find_all_trades(df, time_unit='minutes', n_0=60, m=15, max_iterations=5)
arr = create_buy_sell_prediction_columns(df, final_data_frame, time_units='minutes')

print(arr)
#data_frame_block.plot()
#plt.show()