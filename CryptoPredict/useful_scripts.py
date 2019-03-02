from CryptoPredict import increase_saved_dataset_length
from datetime import datetime
import numpy as np
import pandas as pd
import os
import re

def update_old_dataset():
    date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00 EST'
    pickle_path = input('Enter path to old data here: ')
    increase_saved_dataset_length(pickle_path, date_to)

def create_arrays_for_price_probability_scatter(prices, hours):
    prices = prices[-hours*60::]
    x = np.arange(np.min(prices), np.max(prices), 0.01)
    y = np.array([])

    for i in range(0, len(x)):
        probability_of_return = np.sum(prices > x[i])/len(prices)
        y = np.append(y, probability_of_return)

    return x, y

def concat_csv_files(file_paths_list, new_file_path):
    print('For ' + new_file_path)
    final_df = None
    file_num = 1
    for file_path in file_paths_list:
        print('Now loading file ' + str(file_num) + ' of ' + str(len(file_paths_list)))
        current_df = pd.read_csv(file_path)
        file_num += 1
        if final_df is None:
            final_df = current_df
        else:
            next_df = current_df
            next_df.index = np.arange(0, len(current_df.index)) + np.max(final_df.index.values) + 1
            final_df = final_df.append(next_df)

    final_df.to_csv(new_file_path)


if __name__ == "__main__":
    all_files = os.listdir('/Users/rjh2nd/Dropbox (Personal)/crypto/AppDataStorage')
    regex_fill = re.compile('fills*')
    csv_fills_list_local = list(filter(regex_fill.search, all_files))
    csv_fills_list = ['/Users/rjh2nd/Dropbox (Personal)/crypto/AppDataStorage//'+name for name in csv_fills_list_local]

    regex_order_book = re.compile('book*')
    csv_order_books_list_local = list(filter(regex_order_book.search, all_files))
    csv_order_books_list = ['/Users/rjh2nd/Dropbox (Personal)/crypto/AppDataStorage//'+name for name in csv_order_books_list_local]


    concat_csv_files(csv_fills_list,
                     '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_fills_granular.csv')
    concat_csv_files(csv_order_books_list,
                     '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_historical_order_books_granular.csv')