import csv
import os
import sys
import traceback
from cbpro import PublicClient
from time import time
from time import sleep
from pathlib import Path
from itertools import islice


def print_err_msg(section_text, e, err_counter):
        sleep(5) #Most errors are connection related, so a short time out is warrented
        err_counter += 1
        print('failed to' + section_text + ' due to error: ' + str(e))
        print('number of consecutive errors: ' + str(err_counter))
        print(traceback.format_exc())

        return err_counter

def get_single_order_book_row(sym, pub_client):
    product_id = sym.upper() + '-USD'
    order_book = pub_client.get_product_order_book(product_id, level=2)
    sleep(0.5)
    ts = str(time())
    if not ('bids' in order_book.keys()):
        print('Get ' + sym.upper() + ' order book error, the returned dict is: ' + order_book)
        return None, None
    bids = order_book['bids']
    asks = order_book['asks']
    num_order_book_entries = 20 # How far up the order book to scrape
    num_cols = 3*2*num_order_book_entries

    bid_row = []
    ask_row = []

    for i in range(0, num_order_book_entries):
        bid_row = bid_row + bids[i]
        ask_row = ask_row + asks[i]

    new_row = [ts] + bid_row + ask_row
    header_names = ['ts'] + [str(x) for x in range(0, num_cols)]

    return new_row, header_names


def save_single_row(file_name, new_row, header_names):
    csv_historical_order_books = Path(file_name)
    is_csv_file = csv_historical_order_books.is_file()

    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        if is_csv_file:
            writer.writerow(new_row)
        else:
            writer.writerow(header_names)
            writer.writerow(new_row)

def scrape_and_save_order_books(sym_list, file_name_base='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/', unique_id=''):

    err_counter = 0
    old_row_dict = dict((sym, [0]) for sym in sym_list)
    old_fill_dict = dict((sym, ['N/A']) for sym in sym_list)
    sym = 'N/A'
    public_client = PublicClient()

    while err_counter < 10:

        try:
            for sym in sym_list:
                #TODO get rid of need to repeat processes for every loop
                file_name = file_name_base + sym + '_historical_order_books' + unique_id + '.csv'
                old_row = old_row_dict[sym]
                new_row, header_names = get_single_order_book_row(sym, public_client)
                if new_row is None:
                    # For errors
                    continue
                old_row_dict[sym] = new_row

                if old_row == new_row:
                    # Don't waste time saving repeated rows
                    continue

                save_single_row(file_name, new_row, header_names)

            err_counter = 0
        except Exception as e:
            err_counter = print_err_msg('scrape and save ' + sym + ' order_book', e, err_counter)

        try:
            for sym in sym_list:
                prod_id = sym.upper() + '-USD'
                recent_fill = list(islice(public_client.get_product_trades(product_id=prod_id), 1))[0]
                sleep(0.5)
                if type(recent_fill) == str:
                    print('Error: recent_fill is a str, recent_fill = ' + recent_fill)
                    continue
                recent_fill_side = recent_fill['side']
                last_fill_side = old_fill_dict[sym]
                old_fill_dict[sym] = recent_fill_side

                if last_fill_side == recent_fill_side:
                    continue

                fill_file_name = file_name_base + sym + '_fills' + unique_id + '.csv'

                save_single_row(fill_file_name, list(recent_fill.values()), list(recent_fill.keys()))

        except Exception as e:
            err_counter = print_err_msg(' get last fill for ' + sym, e, err_counter)


if __name__ == '__main__':
    sym_list = ['BCH', 'BTC', 'ETC', 'ETH', 'LTC', 'ZRX']
    print('Begin scraping data')
    scrape_and_save_order_books(sym_list, unique_id='_20entries_1')