import csv
import os
import sys
import traceback
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
from cbpro import PublicClient
from time import time
from time import sleep
from pathlib import Path
from itertools import islice



def save_file_to_dropbox(data_path, file_path, access_token):
    dbx = dropbox.Dropbox(access_token)

    with open(data_path, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        try:
            dbx.files_upload(f.read(), file_path, mode=WriteMode('overwrite'))
            print(file_path + ' uploaded!')
        except ApiError as err:
            # This checks for the specific error where a user doesn't have
            # enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().reason.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()

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

def scrape_and_save_order_books(sym_list, file_name_base='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/', unique_id='', run_time=24*3600):

    err_counter = 0
    old_row_dict = dict((sym, [0]) for sym in sym_list)
    old_fill_dict = dict((sym, ['N/A']) for sym in sym_list)
    sym = 'N/A'
    public_client = PublicClient()
    start_time = time()
    elapsed_time = 0
    file_name = None
    local_file_path = None
    fill_file_name = None
    local_fill_file_path = None

    while (err_counter < 10) and (elapsed_time < run_time):

        try:
            for sym in sym_list:
                #TODO get rid of need to repeat processes for every loop
                file_name = sym + '_historical_order_books' + unique_id + '.csv'
                local_file_path = file_name_base + file_name
                old_row = old_row_dict[sym]
                new_row, header_names = get_single_order_book_row(sym, public_client)
                if new_row is None:
                    # For errors
                    continue
                old_row_dict[sym] = new_row

                if old_row == new_row:
                    # Don't waste time saving repeated rows
                    continue

                save_single_row(local_file_path, new_row, header_names)

            err_counter = 0
            current_time = time()
            elapsed_time = current_time - start_time
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

                fill_file_name = sym + '_fills' + unique_id + '.csv'
                local_fill_file_path = file_name_base + fill_file_name

                save_single_row(local_fill_file_path, list(recent_fill.values()), list(recent_fill.keys()))

        except Exception as e:
            err_counter = print_err_msg(' get last fill for ' + sym, e, err_counter)

    return file_name, local_file_path, fill_file_name, local_fill_file_path


def continuous_scrape(access_token, sym_list, file_name_base):
    # TODO make this work with multiple symbols (right now only pushes one symbol to dropbox)
    while True:
        file_name, local_file_path, fill_file_name, local_fill_file_path = scrape_and_save_order_books(sym_list,
                                    file_name_base=file_name_base,
                                    unique_id=str(time()), run_time=24*3600)

        dbx_file_path = '/crypto/AppDataStorage/' + file_name
        save_file_to_dropbox(local_file_path, dbx_file_path, access_token)

        dbx_fill_file_path = '/crypto/AppDataStorage/' + fill_file_name
        save_file_to_dropbox(local_fill_file_path, dbx_fill_file_path, access_token)

        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        else:
            print("The file does not exist")

        if os.path.exists(local_fill_file_path):
            os.remove(local_fill_file_path)
        else:
            print("The fill file does not exist")



if __name__ == '__main__':
    sym_list = ['ETH']
    if len(sys.argv) > 1:
        dbx_key = sys.argv[1]

    else:
        dbx_key = input('What is the token key?')

    print('Begin scraping data')
    continuous_scrape(dbx_key, sym_list, './')