import requests
from requests.auth import AuthBase
import re
from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
import pytz
import pickle
import base64
import hashlib
import hmac
from cbpro import PublicClient
from tzlocal import get_localzone
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler
from CryptoBot.CryptoBot_Shared_Functions import progress_printer


def str_list_to_timestamp(datetime_str_list):
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    utc = pytz.UTC
    for i in range(0, len(datetime_str_list)):
        current_date_str = datetime_str_list[i]
        if '.' not in current_date_str:
            new_date_str = current_date_str[0:-1] + '.000Z'
            datetime_str_list[i] = new_date_str


    localized_datetime_objects = [utc.localize(datetime.strptime(str, fmt)) for str in datetime_str_list]
    time_stamps = np.array([dt.timestamp() for dt in localized_datetime_objects])

    return time_stamps

def average_orderbook_features(start_col, full_row):
    i = start_col
    full_values = np.array([])

    while i < (len(full_row.columns)):
        col_title = str(i)
        full_values = np.append(full_values, full_row[col_title])
        i += 3

    ans = np.mean(full_values)

    return ans

def price_at_max_order_size(full_row):
    i = 1
    full_values = np.array([])

    while i < (len(full_row.columns)):
        col_title = str(i)
        full_values = np.append(full_values, full_row[col_title])
        i += 3

    max_size_ind = np.argmax(full_values)
    ans = full_row[str(3*(max_size_ind))]

    return ans

def normalize_order_book_row(price_base_value, full_row):
    i = 0

    normalized_row = full_row.values
    size_base_value = average_orderbook_features(1, full_row)
    num_orders_base_value = average_orderbook_features(2, full_row)


    while i < (len(full_row.columns)):
        price_col_title = str(i)
        size_col_title = str(i + 1)
        num_orders_col_title = str(i + 2)

        order_price = full_row[price_col_title]
        order_size = full_row[size_col_title]
        order_num = full_row[num_orders_col_title]

        normalized_order_price = order_price / price_base_value
        normalized_row[0, i] = normalized_order_price
        normalized_size = order_size / size_base_value
        normalized_row[0, i + 1] = normalized_size
        normalized_order_num = order_num / num_orders_base_value
        normalized_row[0, i + 2] = normalized_order_num
        i += 3

    #normalized_row = np.delete(normalized_row, [0])

    return  normalized_row

def normalize_fill_array_and_order_book(historical_order_books, historical_fills):
    # This function takes advantage of the Markovian nature of crypto prices and normalizes the fills by the current
    # top bid. This is intended to make the predictions homogeneous no matter what the current price is
    # Note: current setup has prices at every third entry, should change to have identifying headers

    order_book = historical_order_books
    fills = historical_fills

    fill_ind = 0
    order_book_ts_vals = order_book.ts.values
    order_book_top_bid_vals = order_book['0'].values # The fills are normalized off the top bid

    fill_ts_vals = str_list_to_timestamp(fills.time.values)
    fill_price_vals = fills.price.values


    current_fill_ts = fill_ts_vals[fill_ind]
    current_fill = fill_price_vals[fill_ind]
    normalized_fills = np.array([])

    for order_book_ind in range(0, len(order_book_ts_vals)):

        progress_printer(len(order_book_ts_vals), order_book_ind)

        ts = order_book_ts_vals[order_book_ind]
        current_bid = order_book_top_bid_vals[order_book_ind]

        while (ts > current_fill_ts) or (np.abs(current_fill - current_bid) < 1):
            fill_ind += 1
            if fill_ind == len(fill_price_vals):
                # If there are more order book states after the last fill than this stops early
                return normalized_fills, normalized_order_book
            current_fill_ts = fill_ts_vals[fill_ind]
            current_fill = fill_price_vals[fill_ind]

        current_order_book_row = order_book[order_book.index == order_book_ind]
        current_order_book_row = current_order_book_row.drop(['ts'], axis=1)
        fill_base_val = price_at_max_order_size(current_order_book_row)
        price_base_val = average_orderbook_features(0, current_order_book_row)

        current_normalized_fill = current_fill/fill_base_val
        normalized_fills = np.append(normalized_fills, current_normalized_fill)


        normalized_order_book_row = normalize_order_book_row(price_base_val, current_order_book_row)

        if order_book_ind == 0:
            normalized_order_book = normalized_order_book_row
        else:
            normalized_order_book = np.vstack((normalized_order_book, normalized_order_book_row))

    return normalized_fills, normalized_order_book

def plot_correlation(historical_order_books, historical_fills):
    #output_vec, temp_input_arr = normalize_fill_array_and_order_book(historical_order_books, historical_fills)
    with open('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/current_test_books.pickle',
              'rb') as ds_file:
        temp_input_arr = pickle.load(ds_file)

    with open('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/current_test_fills.pickle',
              'rb') as ds_file:
        output_vec = pickle.load(ds_file)

    output_mask = np.diff(output_vec) > 0.001
    output_mask = np.append(np.array([True]), output_mask)

    x = temp_input_arr[output_mask, 0]
    y = output_vec[output_mask]
    plt.plot(x, y, 'b.')
    plt.title('Correlation')
    plt.figure()
    plt.plot(y)
    plt.title('Normalized Fills')
    plt.figure()
    plt.plot(x)
    plt.title('Normalized First Buys')
    plt.show()

if __name__ == '__main__':

    historical_order_books_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/BTC_historical_order_books_20entries.csv'
    historical_fills_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/BTC_fills_20entries.csv'

    historical_order_books = pd.read_csv(historical_order_books_path)
    historical_fills = pd.read_csv(historical_fills_path)

    plot_correlation(historical_order_books, historical_fills)
