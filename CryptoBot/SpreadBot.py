# Format for fills
# [{
#     "time": "2014-11-07T22:19:28.578544Z",
#     "trade_id": 74,
#     "price": "10.00000000",
#     "size": "0.01000000",
#     "side": "buy"
# }, {
#     "time": "2014-11-07T01:08:43.642366Z",
#     "trade_id": 73,
#     "price": "100.00000000",
#     "size": "0.01000000",
#     "side": "sell"
# }]

import matplotlib
matplotlib.use('Agg')
import sys
import cbpro
import pandas as pd
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt
from operator import itemgetter
from itertools import islice
from itertools import compress
from time import sleep
from time import time
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoBot.CryptoBot_Shared_Functions import current_est_time
from CryptoBot.CryptoBot_Shared_Functions import print_err_msg
from CryptoBot.CryptoBot_Shared_Functions import str_list_to_timestamp
import re

# SETTINGS_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/spread_bot_settings.txt'
# SAVED_DATA_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/Test' + str(current_est_time().date()).replace('-', '')
SETTINGS_FILE_PATH = r'./spread_bot_settings.txt'
SAVED_DATA_FILE_PATH = r'./Test' + str(current_est_time().date()).replace('-', '')
MIN_SPREAD = 1.005

if not os.path.exists(SAVED_DATA_FILE_PATH):
    os.mkdir(SAVED_DATA_FILE_PATH)
# else:
#     override_saved_data = input('Override data in current saved data folder? (yes/no)' )
#     if override_saved_data != 'yes': # TODO change to allow inclusion of a new file name (also print old one)
#         raise ValueError('Folder for saved plots already taken')


EXCHANGE_CONSTANTS = {'BTC':{'resolution':2, 'base order min':0.001, 'base resolution':8},
                      'ETH':{'resolution':2, 'base order min':0.01, 'base resolution':8},
                      'XRP':{'resolution':4, 'base order min':1, 'base resolution':0},
                      'LTC':{'resolution':2, 'base order min':0.1, 'base resolution':8},
                      'BCH':{'resolution':2, 'base order min':0.01, 'base resolution':8},
                      'EOS':{'resolution':3, 'base order min':0.1, 'base resolution':1},
                      'XLM':{'resolution':6, 'base order min':1, 'base resolution':0},
                      'ETC':{'resolution':3, 'base order min':0.1, 'base resolution':8},
                      'LINK':{'resolution':5, 'base order min':1, 'base resolution':0},
                      'REP':{'resolution':2, 'base order min':0.1, 'base resolution':6},
                      'ZRX':{'resolution':6, 'base order min':1, 'base resolution':5},
                      'XTZ':{'resolution':3, 'base order min':1, 'base resolution':0}
                      }

QUOTE_ORDER_MIN = 10

class Product:
    orders = {'buy': {}, 'sell': {}}
    order_book = None

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False, auth_client=None, pub_client=None):

        self.product_id = prediction_ticker.upper() + '-USD'
        self.usd_decimal_num = EXCHANGE_CONSTANTS[prediction_ticker]['resolution']
        self.usd_res = 10**(-self.usd_decimal_num)
        self.quote_order_min = QUOTE_ORDER_MIN
        self.base_order_min = EXCHANGE_CONSTANTS[prediction_ticker]['base order min']
        self.base_decimal_num = EXCHANGE_CONSTANTS[prediction_ticker]['base resolution']
        self.filt_fills = None
        self.all_fills = None
        self.filtered_fill_times = None
        self.fill_time = 0

        if auth_client is None:
            if is_sandbox_api:
                self.api_base = 'https://api-public.sandbox.pro.coinbase.com'
                self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)
                self.pub_client = cbpro.PublicClient(api_url=self.api_base)
            else:
                self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase)
                self.pub_client = cbpro.PublicClient()
        else:
            self.auth_client = auth_client
            self.pub_client = pub_client

    def get_current_book(self):
        order_book = self.pub_client.get_product_order_book(self.product_id, level=2)
        sleep(0.5)
        ts = time()
        if not ('bids' in order_book.keys()):
            print('Get order book error, the returned dict is: ' + str(order_book))
        else:
            self.order_book = order_book
            order_book['time'] = ts

    def get_top_order(self, side):
        self.get_current_book()
        if not side in ['asks', 'bids']:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = float(self.order_book[side][0][0])
        return top_order

    def get_recent_fills(self, fill_number=1000, return_time=False):
        if (time() - self.fill_time) > 5:
            # If fills have not been scraped recently (last 3s) then scrape
            recent_fills = None
            for i in range(0, 10):
                recent_fills = list(islice(self.pub_client.get_product_trades(product_id=self.product_id), fill_number))
                sleep(0.5)
                if 'message' in recent_fills:
                    sleep(1)
                else:
                    break
            recent_fills.reverse() # arrange from newest to oldest

            # Ensure that only current fills are looked at for calculating statistics
            fill_ts_ls_r = np.array(str_list_to_timestamp([fill['time'] for fill in recent_fills]))
            fill_ts_diff = np.abs(fill_ts_ls_r - np.max(fill_ts_ls_r))
            current_fill_mask = fill_ts_diff < 7200
            recent_fills = list(compress(recent_fills, current_fill_mask))

            # Aggreagate changes over moves in a single direction
            fill_mask = [recent_fills[i]['side']!=recent_fills[i-1]['side'] for i in range(0, len(recent_fills))]
            filtered_recent_fills = list(compress(recent_fills, fill_mask))
            filt_fill_ts_ls_r = list(compress(fill_ts_ls_r, fill_mask))
            self.filt_fills = filtered_recent_fills
            self.filtered_fill_times = filt_fill_ts_ls_r
            self.all_fills = recent_fills
            self.fill_time = time()
        else:
            # If fills have been scraped recently (within the last 30s) then use saved values
            filtered_recent_fills = self.filt_fills
            recent_fills = self.all_fills
            filt_fill_ts_ls_r = self.filtered_fill_times

        if return_time:
            return filtered_recent_fills, filt_fill_ts_ls_r
        else:
            return filtered_recent_fills, recent_fills

    def get_num_price_momentum_switches_per_time(self, t_interval=5*60):
        fills, fill_ts_ls = self.get_recent_fills(return_time=True)
        num_trades_per_t = []
        num_trades = 0
        ts0 = fill_ts_ls[0]
        t = 0

        for ts in fill_ts_ls:
            t = ts - ts0
            if t >= t_interval:
                num_trades_per_t.append(num_trades)
                ts0 = ts
                num_trades = 0
            else:
                num_trades += 1

        if t > (0.1 * t_interval):
            num_trades_per_t.append((t_interval/t) * num_trades)

        avg_num_trades = np.mean(np.array(num_trades_per_t))
        if (avg_num_trades is None) or np.isnan(avg_num_trades):
            avg_num_trades = (t_interval/t) * num_trades

        return avg_num_trades

    def adjust_fill_data(self, mu, std, fill_diff_ratio):
        avg_num_trades = self.get_num_price_momentum_switches_per_time()
        weighted_mu = avg_num_trades * mu
        weighted_std = np.sqrt(avg_num_trades) * std
        if avg_num_trades < 1:
            last_fill = fill_diff_ratio[-1]
        else:
            last_fill = np.sum(fill_diff_ratio[-int(avg_num_trades)::])

        return weighted_mu, weighted_std, last_fill

    def get_mean_and_std_of_price_changes(self):
        fills, _ = self.get_recent_fills()
        fill_arr = np.array([float(fill['price']) for fill in fills])
        fill_diff = np.diff(fill_arr)
        fill_diff_mask = np.abs(fill_diff) > self.usd_res #ignore small bounces between the minimum resolution
        fill_diff_ratio = np.append(0, fill_diff) / fill_arr
        fill_diff_ratio = fill_diff_ratio[1::][fill_diff_mask]
        std = np.std(fill_diff_ratio)
        mu = np.mean(fill_diff_ratio)

        # Adjust mu and std to account for number of trades over time
        weighted_mu, weighted_std, last_fill = self.adjust_fill_data(mu, std, fill_diff_ratio)

        return weighted_mu, weighted_std, last_fill

    def get_mean_and_std_of_fill_sizes(self, side):
        _, fills = self.get_recent_fills()
        size_ls = []
        current_size = 0
        # This loop aggregates over moves in a single direction to match the price based variables
        for i in range(1, len(fills)):
            current_fill = fills[i]
            last_fill = fills[i-1]
            size = float(last_fill['size'])
            current_size += size
            if current_fill['side'] != last_fill['side']:
                size_ls.append(current_size * (-1)**(current_fill['side'] == side)) # size is negative for sizes in the opposing direction
                current_size=0

        size_arr = np.array(size_ls)
        size_arr = size_arr[np.abs(size_arr) > 0]
        std = np.std(size_arr)
        mu = np.mean(size_arr)

        # Adjust mu and std to account for number of trades over time
        weighted_mu, weighted_std, last_fill = self.adjust_fill_data(mu, std, size_arr)

        return weighted_mu, weighted_std, last_fill




    def place_order(self, price, side, size, coeff=1, post_only=True):
        # TODO ensure it never rounds up to the point that the order is larger than available

        if not side in ['buy', 'sell']:
            raise ValueError(side + ' is not a valid orderbook side')
        if price * size < self.quote_order_min:
            print(num2str(price * size, self.usd_decimal_num) + ' is smaller than the minimum quote size')
            return None
        if size < self.base_order_min:
            print(num2str(size, 6) + ' is smaller than the minimum base size')
            return None

        new_order_id = None
        price_str = num2str(price, self.usd_decimal_num)
        size_str = num2str(coeff * size, self.base_decimal_num)

        order_info = self.auth_client.place_limit_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only)
        sleep(0.5)

        if type(order_info) == dict:
            if "price" in order_info.keys():
                new_order_id = order_info["id"]
        if new_order_id is None:
            print(order_info)

        return new_order_id

class Wallet:
    offset_value = None # Offset value is subtracted from the amount in usd to ensure only the desired amount of money is traded
    last_buy_price = None
    last_sell_price = None
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, api_key, secret_key, passphrase, sym='ETH', is_sandbox_api=False, auth_client=None, pub_client=None):
        self.product = Product(api_key, secret_key, passphrase, prediction_ticker=sym, is_sandbox_api=is_sandbox_api, auth_client=auth_client, pub_client=pub_client)
        self.ticker = sym
        self.value = {'USD': 15, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}

    def get_wallet_values(self, currency, data):
        # Data should come from self.auth_client.get_accounts()
        ind = [acc["currency"] == currency for acc in data]
        wallet = data[ind.index(True)]
        balance = wallet["balance"]
        hold_balance = wallet["hold"]

        return balance, hold_balance

    def update_value(self, data=None):
        if data is None:
            data = self.product.auth_client.get_accounts()
        sleep(0.5)
        usd_balance, usd_hold_balance = self.get_wallet_values('USD', data)
        sym_balance, sym_hold_balance = self.get_wallet_values(self.ticker, data)
        usd_float_balance = float(usd_balance)
        if self.offset_value is None:
            if usd_float_balance > self.value['USD']:
                self.offset_value = usd_float_balance - self.value['USD']
            else:
                self.offset_value = 0
                print('starting value too large, defaulting to full portfolio value')

        self.value['USD'] = usd_float_balance - self.offset_value
        self.value['USD Hold'] = float(usd_hold_balance)
        self.value['SYM'] = float(sym_balance)
        self.value['SYM Hold'] = float(sym_hold_balance)

    def get_amnt_available(self, side):
        if side == 'sell':
            sym = 'SYM'
        elif side == 'buy':
            sym = 'USD'
        else:
            raise ValueError('side value set to ' + side + ', side must be either "sell" or "buy"')
        available = float(self.value[sym]) - float(self.value[sym + ' Hold'])
        return available

class LiveRunSettings:

    settings = {'portfolio value offset':None, 'limit buy':None, 'limit sell':None, 'spread':1.01, 'std':2}

    def __init__(self, settings_file_path):
        self.fname = settings_file_path
        with open(settings_file_path) as f:
            self.contents = f.readlines()
        self.reg_ex = re.compile(r'(?<=:)([0-9]*\.[0-9]+|[0-9]+)')

    def read_setting_from_file(self, setting_name):
        setting_value = None

        for content in self.contents:
            if setting_name in content:
                setting_str = self.reg_ex.search(content.replace(' ', ''))
                if setting_str is not None:
                    setting_value = float(setting_str.group(0)) # TODO investigate how to get this line to work on python 3.5 (for AWS)

        return setting_value

    def update_settings(self):
        with open(self.fname) as f:
            self.contents = f.readlines()
        for setting_name in self.settings.keys():
            setting_value = self.read_setting_from_file(setting_name)
            self.settings[setting_name] = setting_value

    def write_setting_to_file(self, setting_name, setting_val):
        self.update_settings()
        self.settings[setting_name] = setting_val
        write_str = ''
        for key in self.settings.keys():
            write_str = write_str + key + ': ' + str(self.settings[key]) + '\n'
        with open(self.fname, 'w') as f:
            f.write(write_str)

class CombinedPortfolio:

    def __init__(self, api_key, secret_key, passphrase, sym_list, offset_value=70, is_sandbox=False):
        self.wallets = {}

        if is_sandbox:
            api_base = 'https://api-public.sandbox.pro.coinbase.com'
            auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=api_base)
            sleep(0.5)
            pub_client = cbpro.PublicClient(api_url=api_base)
        else:
            auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase)
            sleep(0.5)
            pub_client = cbpro.PublicClient()

        self.auth = auth_client

        for symbol in sym_list:
            self.wallets[symbol] = Wallet(api_key, secret_key, passphrase, sym=symbol, auth_client=auth_client, pub_client=pub_client)
            self.wallets[symbol].offset_value = offset_value

        self.symbols = sym_list

    def update_offset_value(self, new_offset):
        for sym in self.symbols:
            self.wallets[sym].offset_value = new_offset

    def get_common_wallet(self):
        # Useful to use common functions from all wallets regadless of symbol
        wallet = self.wallets[self.symbols[0]]
        return wallet

    def get_full_portfolio_value(self):
        # print('getting portfolio value')
        full_value = 0
        # get the wallet data once to reduce API calls
        wallet = self.get_common_wallet()
        full_wallet_data = wallet.product.auth_client.get_accounts()

        # update the last recorded price and add to get full value
        for sym in self.wallets.keys():
            # print('getting value for ' + sym)
            wallet = self.wallets[sym]
            current_price = {'asks':None, 'bids':None}

            for side in current_price.keys():
                top_order = wallet.product.get_top_order(side)
                current_price[side] = top_order

            wallet.update_value(data=full_wallet_data)
            price = np.mean([current_price['asks'], current_price['bids']])

            sym = wallet.value['SYM']
            full_value += sym*price

        usd = wallet.value['USD']
        full_value += usd

        return full_value

    def get_usd_available(self):
        wallet = self.get_common_wallet()
        usd_available = wallet.get_amnt_available('buy')
        return usd_available

    def get_usd_held(self):
        wallet = self.get_common_wallet()
        usd_hold = wallet.value['USD Hold']
        return usd_hold

    def remove_order(self, id):
        self.auth.cancel_order(id)
        sleep(0.5)

    def update_value(self):
        wallet = self.get_common_wallet()
        # Get the data for all wallets to reduce api calls
        wallets_data = wallet.product.auth_client.get_accounts()

        for sym in self.symbols:
            self.wallets[sym].update_value(data=wallets_data)

class Bot:

    settings = LiveRunSettings(SETTINGS_FILE_PATH)
    spread = 1.01

    def __init__(self, api_key, secret_key, passphrase, syms=('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'EOS', 'XLM', 'ETC', 'LINK', 'REP', 'ZRX', 'XTZ'), is_sandbox_api=False):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        current_offset = self.settings.read_setting_from_file('portfolio value offset')
        self.portfolio = CombinedPortfolio(api_key, secret_key, passphrase, syms, is_sandbox=is_sandbox_api, offset_value=current_offset)
        self.symbols = syms
        self.current_price = {}
        self.spread_price_limits = {}

        for sym in syms:
            self.current_price[sym] = {'asks': None, 'bids': None}
            self.spread_price_limits[sym] = {'sell': None, 'buy': None}

    def update_current_prices(self):
        for side in ['asks', 'bids']:
            for sym in self.symbols:
                top_order = self.portfolio.wallets[sym].product.get_top_order(side)
                self.current_price[sym][side] = top_order

    def place_order(self, price, side, size, sym, post_only=True):
        order_id = self.portfolio.wallets[sym].product.place_order(price, side, size, post_only=post_only)
        return order_id

    def get_full_portfolio_value(self):

        full_value = self.portfolio.get_full_portfolio_value()

        return full_value

    def update_spread_prices_limits(self, last_price, side, sym):
        self.spread_price_limits[sym][side] = last_price
        self.settings.write_setting_to_file('limit ' + side, self.spread_price_limits[sym][side])

    def cancel_out_of_bound_orders(self, side, price, sym):
        orders = list(self.portfolio.wallets[sym].product.auth_client.get_orders(self.portfolio.wallets[sym].product.product_id))
        sleep(0.5)
        keys_to_delete = []
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1
        num_cancelled_orders = 0

        for order in orders:
            if order['side'] != side:
                continue
            if coeff * float(order['price']) > coeff * (price + coeff * self.portfolio.wallets[sym].product.usd_res):
                keys_to_delete.append(order['id'])

        for id in keys_to_delete:
            num_cancelled_orders += 1
            print('Cancelled ' + num2str(num_cancelled_orders, 1) + ' out of bounds ' + sym + ' ' + side + ' orders')
            self.portfolio.remove_order(id)

    def get_side_depedent_vars(self, side):
        if side == 'buy':
            book_side = 'bids'
            opposing_book_side = 'asks'
            coeff = -1
        else:
            book_side = 'asks'
            opposing_book_side = 'bids'
            coeff = 1

        return book_side, opposing_book_side, coeff

    def determine_price_based_on_std(self, sym, usd_available, side, std_mult=1):

        # initialize variables
        book_side, opposing_book_side, coeff = self.get_side_depedent_vars(side)

        wallet = self.portfolio.wallets[sym]
        mu, std, last_diff = wallet.product.get_mean_and_std_of_price_changes()

        # Calculate the coefficient used to determine the which multiple of the std to use
        std_coeff = std_mult * self.settings.read_setting_from_file('std')

        # determine trade price
        std_offset = coeff * (std_coeff * std) + mu
        order_coeff = 1 + std_offset
        current_price = wallet.product.get_top_order(book_side)
        if (side == 'buy') and ((last_diff-1) > np.abs(std_coeff * std + mu)):
            order_coeff = 1
            current_price = wallet.product.get_top_order(opposing_book_side)
        elif (coeff * last_diff < coeff * std_offset):
            order_coeff -= last_diff
        # else:
        #     order_coeff = 1
        #     current_price = wallet.product.get_top_order(opposing_book_side) + coeff * wallet.product.usd_res

        buy_price = order_coeff * current_price
        size = usd_available / buy_price

        return buy_price, wallet, size, std, mu

    def determine_price_based_on_fill_size(self, sym, usd_available, side, std_mult=1):
        # initialize variables
        book_side, opposing_book_side, coeff = self.get_side_depedent_vars(side)
        wallet = self.portfolio.wallets[sym]
        mu, std, existing_fill_size = wallet.product.get_mean_and_std_of_fill_sizes(side)

        # Calculate the coefficient used to determine the which multiple of the std to use
        std_coeff = std_mult * self.settings.read_setting_from_file('std')

        # determine trade price
        current_price = wallet.product.get_top_order(opposing_book_side)
        book = wallet.product.order_book[book_side]
        max_size = std_coeff * std + mu - existing_fill_size * (existing_fill_size > 0)
        if max_size < 0:
            max_size = std_coeff * std + mu
        current_book_size = 0 # This is the size of order needed to climb to a certain point in the book
        buy_price = None
        size = None
        if max_size > 0:
            for order in book:
                buy_price = float(order[0]) - coeff * wallet.product.usd_res
                current_book_size += float(order[1])
                size = usd_available / buy_price
                if ( current_book_size + size ) > max_size:
                    break
        elif buy_price is None:
            buy_price = current_price + coeff * wallet.product.usd_res
            size = usd_available / buy_price

        return buy_price, size

    def determine_trade_price(self, sym, usd_available, side):
        _, _, coeff = self.get_side_depedent_vars(side)
        price_p, wallet, size_p, std, mu = self.determine_price_based_on_std(sym, usd_available, side)
        price_s, size_s = self.determine_price_based_on_fill_size(sym, usd_available, side)
        if coeff * price_p > coeff * price_s: # Always use the more conservative price
            return price_p, wallet, size_p, std, mu
        else:
            return price_s, wallet, size_s, std, mu

    def cancle_out_of_bound_buy_orders(self, top_syms):
        # Ensure to update portfolio value before running
        for sym in self.symbols:
            # Find price 2 standard deviations below the change (unlikely buy price to hit)
            wallet = self.portfolio.wallets[sym]
            _, _, last_diff = wallet.product.get_mean_and_std_of_price_changes()
            minimum_buy_price, _, _, std, mu = self.determine_trade_price(sym, 11, 'buy')

            # std_coeff = 1 - (3 * std_coeff * std) + mu * (mu < 0) # only factor in the mean when it gives a wider margin (make it harder to cancel an order due to short changes)
            if (np.abs(last_diff) > (std + mu)) and (sym in top_syms):
                # Don't cancel orders due to one big jump, but do cancel orders that are not first choice
                continue
            self.cancel_out_of_bound_orders('buy', minimum_buy_price, sym)

    def sort_currencies(self, usd_available, print_sym):
        # setup
        ranking_dict = {}

        # create dictionary for symbols and relevant data
        for sym in self.symbols:
            if print_sym:
                print(sym)
            buy_price, wallet, size, std, mu = self.determine_trade_price(sym, usd_available, side='buy')
            sell_price, _, _, _, _ = self.determine_trade_price(sym, usd_available, side='sell')
            current_price = wallet.product.get_top_order('bids')
            spread = 1 + ( sell_price - buy_price ) / buy_price
            rank = mu #( sell_price - current_price ) / current_price
            if spread < MIN_SPREAD:
                rank = -1 # eliminate trades with low spreads
            ranking_dict[sym] = (rank, mu, buy_price, wallet, std, spread, size)

        # sort (by mean first then standard deviation)
        sorted_syms = sorted(ranking_dict.items(), key=itemgetter(1), reverse=True)

        return sorted_syms

    def rank_currencies(self, usd_available, print_sym=True, sym_ind=0):
        sorted_syms = self.sort_currencies(usd_available, print_sym)
        return_None = False
        if type(sym_ind) == int:
            top_sym_data = sorted_syms[sym_ind]
            top_sym = top_sym_data[0]
            mu = top_sym_data[1][1]
            buy_price = top_sym_data[1][2]
            wallet = top_sym_data[1][3]
            std = top_sym_data[1][4]
            spread = top_sym_data[1][5]
            size = top_sym_data[1][6]
            if spread < 1.004:
                return_None = True
        else:
            top_sym = []
            spread = []
            buy_price = []
            wallet = []
            std = []
            mu = []
            size = []

            for ind in sym_ind:
                top_sym_data = sorted_syms[ind]
                if top_sym_data[1][5] < MIN_SPREAD:
                    continue
                top_sym.append(top_sym_data[0])
                mu.append(top_sym_data[1][1])
                buy_price.append(top_sym_data[1][2])
                wallet.append(top_sym_data[1][3])
                std.append(top_sym_data[1][4])
                spread.append(top_sym_data[1][5])
                size.append(top_sym_data[1][6])

            if len(spread) == 0:
                return_None = True

        if not return_None:
            return buy_price, wallet, size, top_sym, std, mu, spread
        else:
            return None, None, None, None, None, None, None

    def place_order_for_nth_currency(self, no_viable_trade, top_syms, usd_available, n=0):
        print('Evaluating currencies for best buy')
        if no_viable_trade:
            print('No viable trades found')
        else:
            top_sym = top_syms[n]
            buy_price, wallet, size, std, mu = self.determine_trade_price(top_sym, usd_available, side='buy')
            sell_price, _, _, _, _ = self.determine_trade_price(top_sym, usd_available, side='sell')
            # recalculate prices for most recent figure
            buy_price, _, _, _, _ = self.determine_trade_price(top_sym, usd_available, side='buy')
            sell_price, _, _, _, _ = self.determine_trade_price(top_sym, usd_available, side='sell')
            spread = 1 + (sell_price - buy_price) / buy_price
            # place order and record
            print('\n' + top_sym + ' Chosen as best buy')
            print('placing order\n' + 'price: ' + num2str(buy_price,
                                                          wallet.product.usd_decimal_num) + '\n' + 'size: ' + num2str(
                size, 3) + '\n' + 'std: ' + num2str(std, 6) + '\n' + 'mu: ' + num2str(mu, 6) + '\n')
            order_id = self.place_order(buy_price, 'buy', size, top_sym, post_only=False)
            if order_id is None:
                print('Buy Order rejected\n')
            else:
                print('Buy Order placed!\n')
                self.update_spread_prices_limits(buy_price, 'buy', top_sym)
                if spread < MIN_SPREAD:
                    spread = MIN_SPREAD
                    sell_price = spread * buy_price
                self.settings.write_setting_to_file('spread', spread)
                self.update_spread_prices_limits(sell_price, 'sell', top_sym)

    def place_order_for_top_currencies(self):
        # Ensure to update portfolio value before running
        usd_hold = self.portfolio.get_usd_held()
        usd_available = self.portfolio.get_usd_available()
        buy_prices, wallets, sizes, top_syms, stds, mus, spreads = self.rank_currencies(usd_available, print_sym=False, sym_ind=(0, 1, 2))
        no_viable_trade = False

        # Cancel bad buy orders before continuing
        if top_syms is None:
            no_viable_trade = True
            top_syms = self.symbols  # If no viable trades are found then allow any symbol to remain
        if usd_hold > QUOTE_ORDER_MIN:
            self.cancle_out_of_bound_buy_orders(top_syms=top_syms)
            sleep(0.5)

        # Check available cash after canceling the non_optimal buy orders and place the next order
        full_portfolio_value = self.get_full_portfolio_value()
        num_orders = len(top_syms)
        for ind in range(1, num_orders + 1):
            i = num_orders - ind
            usd_available = self.portfolio.get_usd_available()
            if (usd_available > QUOTE_ORDER_MIN):
                nominal_order_size = full_portfolio_value / num_orders
                if nominal_order_size < QUOTE_ORDER_MIN:
                    # never try to place an order smalle than the minimum
                    nominal_order_size = QUOTE_ORDER_MIN

                if nominal_order_size > (usd_available - QUOTE_ORDER_MIN):
                    # If placing the nominal order leaves an unusable amount of money then skip
                    order_size = usd_available
                else:
                    order_size = nominal_order_size

                self.place_order_for_nth_currency(no_viable_trade, top_syms, order_size, n=i)
            else:
                break

    def place_limit_sells(self):
        # Ensure to update portfolio value before running
        for sym in self.portfolio.wallets.keys():
            # Get relevant variables
            wallet = self.portfolio.wallets[sym]
            limit_price = self.spread_price_limits[sym]['sell']
            bounce_price, _ = self.determine_price_based_on_fill_size(sym, 11, 'sell', std_mult=6)
            e_sell_price, _ = self.determine_price_based_on_fill_size(sym, 11, 'sell', std_mult=4)
            available = wallet.get_amnt_available('sell')
            alt_price = wallet.product.get_top_order('bids')

            if limit_price is None:
                continue

            # Cancel sell orders that have a large resistance
            if (e_sell_price < limit_price) and (limit_price < bounce_price):
                print('Initiating market sell because sell price of ' + num2str(limit_price) + ' out of bounds' + '\n')
                self.update_spread_prices_limits(alt_price, 'sell', sym)
                self.cancel_out_of_bound_orders( 'sell', alt_price, sym)
                available = wallet.get_amnt_available('sell')
                limit_price = alt_price
            elif (limit_price > bounce_price):
                print('Reducing sell price because sell price of ' + num2str(limit_price) + ' out of bounds' + '\n')
                self.update_spread_prices_limits(bounce_price, 'sell', sym)
                self.cancel_out_of_bound_orders('sell', bounce_price, sym)
                available = wallet.get_amnt_available('sell')
                limit_price = bounce_price

            # Filter unnecessary currencies
            if (available < wallet.product.base_order_min):
                continue
            if (available*limit_price < QUOTE_ORDER_MIN):
                print('Cannot sell ' + sym + ' because available is less than minnimum Quote order. Manual sell required')
                continue

            print('Placing ' + sym + ' spread sell order' + '\n' + 'price: ' + num2str(limit_price, wallet.product.usd_decimal_num) + '\n')

            order_id = self.place_order(limit_price, 'sell', available, sym, post_only=False)
            if order_id is None:
                print('Sell Order rejected\n')
            else:
                print('\nSell Order placed')

class PortfolioTracker:

    def __init__(self, portfolio):
        self.portfolio = portfolio
        percentage_data = {'Market': 100, 'Algorithm': 100}
        current_datetime = current_est_time()
        self.returns = pd.DataFrame(data=percentage_data, index=[current_datetime])
        self.initial_price = portfolio.wallets['BTC'].product.get_top_order('bids')
        self.initial_value = portfolio.get_full_portfolio_value()
        self.prediction_ticker = 'BTC'
        absolute_data = {'Portfolio Value:':self.initial_value}
        self.portfolio_value = pd.DataFrame(data=absolute_data, index=[current_datetime])

    def format_price_info(self, returns, price):
        if returns >= 0:
            value_sym = '+'
        else:
            value_sym = '-'

        return_str = value_sym + num2str(returns, 3)
        formated_string = '$' + num2str(price, 2) + ' (' + return_str + ')'

        return formated_string

    def add_new_row(self, df, new_row):
        # Append data to dataframes
        new_df = df.append(new_row)

        # Ensure dataframes are not too long
        diff_from_max_len = len(new_df.index) - 60000
        if diff_from_max_len > 0:
            new_df = new_df.iloc[diff_from_max_len::]

        return new_df

    def update_returns_data(self):
        # Scrape data
        price = self.portfolio.wallets['BTC'].product.get_top_order('bids')
        portfolio_value = self.portfolio.get_full_portfolio_value()

        # Setup calculated values
        current_datetime = current_est_time()
        market_returns = 100 * price / self.initial_price - 100
        portfolio_returns = 100 * portfolio_value / self.initial_value - 100

        # Setup new rows
        data = {'Market': market_returns + 100, 'Algorithm': portfolio_returns + 100}
        absolute_data = {'Portfolio Value:': portfolio_value}
        new_percentage_row = pd.DataFrame(data=data, index=[current_datetime])
        new_portfolio_row = pd.DataFrame(data=absolute_data, index=[current_datetime])

        # Append data to dataframes
        self.returns = self.add_new_row(self.returns, new_percentage_row)
        self.portfolio_value = self.add_new_row(self.portfolio_value, new_portfolio_row)

        return portfolio_returns, market_returns, portfolio_value, price

    def plot_returns(self):
        # Get data
        portfolio_returns, market_returns, portfolio_value, price = self.update_returns_data()
        portfolio_str = self.format_price_info(portfolio_returns, portfolio_value)
        market_str = self.format_price_info(market_returns, price)

        # Plot returns
        self.returns.plot()
        plt.title('Portfolio: ' + portfolio_str + '\n' + self.prediction_ticker + ': ' + market_str)
        plt.xlabel('Date/Time')
        plt.ylabel('% Initial Value')

        plt.savefig(SAVED_DATA_FILE_PATH + r'/returns.png')
        plt.close()

        # Plot portfolio value
        self.portfolio_value.plot()
        plt.title('Total Portfolio Value')
        plt.xlabel('Date/Time')
        plt.ylabel('Portfolio Value ($)')
        plt.savefig(SAVED_DATA_FILE_PATH + r'/value.png')
        plt.close()

        # Save raw data to csv
        self.returns.to_csv(SAVED_DATA_FILE_PATH + r'/returns.csv')
        self.portfolio_value.to_csv(SAVED_DATA_FILE_PATH + r'/value.csv')

        return portfolio_value


def run_bot():
    # -- Secret/changing variable declerations
    if len(sys.argv) > 2:
        # Definition from a shell file
        api_input = sys.argv[1]
        secret_input = sys.argv[2]
        passphrase_input = sys.argv[3]

    else:
        api_input = input('What is the api key? ')
        secret_input = input('What is the secret key? ')
        passphrase_input = input('What is the passphrase? ')

    # Setup initial variables
    print('Initializing bot')
    bot = Bot(api_input, secret_input, passphrase_input)
    bot.portfolio.update_value()
    print('Initializing portfolio tracking')
    portfolio_tracker = PortfolioTracker(bot.portfolio)
    portfolio_value = portfolio_tracker.initial_value
    print('Bot starting value ' + num2str(portfolio_value, 2))

    sleep(1)
    last_check = 0
    last_plot = 0
    plot_period = 60
    check_period = 1
    err_counter = 0

    while (11 < portfolio_value) and (err_counter < 10):
        current_time = datetime.now().timestamp()
        if (current_time > (last_check + check_period)):
            try:
                # Trade
                bot.portfolio.update_value()
                last_check = datetime.now().timestamp()
                bot.place_order_for_top_currencies()
                bot.place_limit_sells()

                # Update Settings
                bot.settings.update_settings()
                new_offset = bot.settings.settings['portfolio value offset']
                bot.portfolio.update_offset_value(new_offset)

                if (current_time > (last_plot + plot_period)):
                    portfolio_value = portfolio_tracker.plot_returns()
                    last_plot = datetime.now().timestamp()
                err_counter = 0

            except Exception as e:
                if err_counter > 1:
                    err_counter = print_err_msg('find new data', e, err_counter)
                err_counter += 1


    print('Loop END')



if __name__ == "__main__":
    run_bot()