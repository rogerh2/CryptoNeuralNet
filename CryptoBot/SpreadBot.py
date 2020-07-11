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
from json.decoder import JSONDecodeError
from datetime import datetime
from matplotlib import pyplot as plt
from operator import itemgetter
from queue import Queue
from itertools import islice
from itertools import compress
from time import sleep, time
from threading import Thread
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoBot.CryptoBot_Shared_Functions import current_est_time
from CryptoBot.CryptoBot_Shared_Functions import print_err_msg
from CryptoBot.CryptoBot_Shared_Functions import offset_current_est_time
from CryptoBot.CryptoBot_Shared_Functions import convert_coinbase_timestr_to_timestamp
from CryptoBot.CryptoBot_Shared_Functions import calculate_spread, safe_get
from CryptoBot.CryptoBot_Shared_Functions import save_file_to_dropbox, private_pause
from CryptoBot.constants import EXCHANGE_CONSTANTS, QUOTE_ORDER_MIN, PRIVATE_SLEEP, MIN_PORTFOLIO_VALUE, TRADE_LEN, PRIVATE_SLEEP_QUEUE, OPEN_ORDERS
from CryptoPredict.CryptoPredict import CryptoCompare
from CryptoBot.Coinbase_API import Wallet, CombinedPortfolio, Websocket
from ODESolvers.PSM import create_multifrequency_propogator_from_data
import re

# SETTINGS_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/spread_bot_settings.txt.txt'
# SAVED_DATA_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/Test' + str(current_est_time().date()).replace('-', '')
def portfolio_file_path_generator():
    file_path_dt_format = '%Y%m%d_%H%M_%Z'
    return r'./' + current_est_time().strftime(file_path_dt_format) + '_coinbasepro'

SYMBOLS = []
for pair in EXCHANGE_CONSTANTS:
    if ('USD' in pair) and (not 'USDC' in pair):
        current_sym = pair.split('-')[0]
        # Do not trade stablecoins
        if current_sym == 'DAI':
            continue
        SYMBOLS.append(current_sym)
SYMBOLS = tuple(SYMBOLS)

SETTINGS_FILE_PATH = r'./spread_bot_settings.txt'
SAVED_DATA_FILE_PATH = portfolio_file_path_generator()
MIN_SPREAD = 1.01 # This is the minnimum spread before a trade can be made
MAX_LIMIT_SPREAD = 1.11 # This is the maximum spread before stop limit orders are utilized
STOP_PROFIT = 0.1 # This is the additional profit needed for the bot to begin using stop limit orders
MIN_PROFIT = 0 # This is the minnimum value (net profit) to get per buy-sell pair, set to 0 to beat fees faster
STOP_SPREAD = 0.001 # This is the delta for limits in stop-limit orders, this is relevant for sell prices
NEAR_PREDICTION_LEN = 30
PSM_EVAL_STEP_SIZE = 1.8 # This is the step size for PSM
ORDER_TIMEOUT = NEAR_PREDICTION_LEN * 60
CYCLE_TIME = 600
WHALE_MARGIN = 0.001

if not os.path.exists(SAVED_DATA_FILE_PATH):
    os.mkdir(SAVED_DATA_FILE_PATH)
else:
    override_saved_data = input('Override data in current saved data folder? (yes/no)' )
    if override_saved_data != 'yes':
        raise ValueError('Folder for saved plots already taken')

OPEN_ORDERS = None

class LiveRunSettings:

    settings = {'portfolio value offset':None, 'minnimum_spread':MIN_SPREAD, 'std':2, 'buy std':0.85, 'sell std':1}

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
                    setting_value = float(setting_str.group(0))
        if setting_name not in self.settings.keys():
            self.settings[setting_name] = setting_value

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
        for key in sorted(self.settings.keys()):
            write_str = write_str + key + ': ' + str(self.settings[key]) + '\n'
        with open(self.fname, 'w') as f:
            f.write(write_str)

class Bot:

    settings = LiveRunSettings(SETTINGS_FILE_PATH)
    spread = 1.01

    def __init__(self, api_key, secret_key, passphrase, syms=SYMBOLS, is_sandbox_api=False, base_currency='USD'):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        self.settings.update_settings()
        self.min_spread = MIN_SPREAD
        current_offset = self.settings.read_setting_from_file('portfolio value offset')
        self.portfolio = CombinedPortfolio(api_key, secret_key, passphrase, syms, is_sandbox=is_sandbox_api, offset_value=current_offset, base_currency=base_currency)
        self.symbols = syms
        self.current_price = {}
        self.spread_price_limits = {}

        for sym in syms:
            current_sell = self.settings.read_setting_from_file(sym + ' limit sell')
            current_buy = self.settings.read_setting_from_file(sym + ' limit buy')
            self.current_price[sym] = {'asks': None, 'bids': None}
            self.spread_price_limits[sym] = {'sell': current_sell, 'buy': current_buy}

    def update_current_prices(self):
        for side in ['asks', 'bids']:
            for sym in self.symbols:
                top_order = self.portfolio.wallets[sym].product.get_top_order(side)
                self.current_price[sym][side] = top_order

    def update_min_spread(self, mkr_fee, tkr_fee=0.0):
        global MIN_SPREAD
        global MAX_LIMIT_SPREAD
        MIN_SPREAD = 1 + 2 * mkr_fee + MIN_PROFIT
        MAX_LIMIT_SPREAD = 1 + 2 * tkr_fee + MIN_PROFIT + STOP_SPREAD + STOP_PROFIT
        self.settings.write_setting_to_file('minnimum_spread', MIN_SPREAD)
        self.min_spread = MIN_SPREAD

    def place_order(self, price, side, size, sym, post_only=True, time_out=False, stop_price=None):
        order_id = self.portfolio.wallets[sym].product.place_order(price, side, size, post_only=post_only, time_out=time_out, stop_price=stop_price)
        return order_id

    def get_full_portfolio_value(self):
        full_value = self.portfolio.get_full_portfolio_value()
        return full_value

    def update_spread_prices_limits(self, last_price, side, sym):
        self.spread_price_limits[sym][side] = last_price
        if side == 'buy':
            opposing_side = 'sell'
        else:
            opposing_side = 'buy'
        self.settings.write_setting_to_file(sym + ' limit ' + side, self.spread_price_limits[sym][side])
        self.settings.write_setting_to_file(sym + ' spread', self.spread)

    def cancel_orders_conditionally(self, side, sym, high_condition, low_condition, msg=None):
        # High condition is a function of the individual orders
        orders = self.portfolio.wallets[sym].product.get_open_orders()
        keys_to_delete = []

        for order in orders:
            if type(order) != dict:
                continue
            if order['side'] != side:
                continue
            if high_condition(order) > low_condition:
                keys_to_delete.append(order['id'])
        if (len(keys_to_delete) > 0) and (msg is not None):
            print(msg)
        for id in keys_to_delete:
            self.portfolio.remove_order(id)

    def cancel_out_of_bound_orders(self, side, price, sym, widen_spread=False):
        # If widen_spread flag is set, will cancel orders that are more conservative (lower sells/higher buys)
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1
        coeff = coeff * (-1)**(widen_spread)

        order_price = lambda x: coeff * float(x['price'])
        min_size = coeff * (price + coeff * self.portfolio.wallets[sym].product.usd_res)
        order_type = ' liberal '
        if widen_spread:
            order_type = ' conservative '

        self.cancel_orders_conditionally(side, sym, order_price, min_size, 'Cancelled' + order_type + sym + ' ' + side + ' orders due to out of bounds price\n')

    def cancel_timed_out_orders(self, side, timeout_time, sym):
        cancel_time = -time()
        order_time = lambda x: -convert_coinbase_timestr_to_timestamp(x['created_at']) - timeout_time

        self.cancel_orders_conditionally(side, sym, order_time, cancel_time)

    def cancel_single_order(self, id):
        try:
            private_pause()
            self.portfolio.auth.cancel_order(id)
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        except:
            print('Cannot cancel order, order not found')
    
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

class TrackerBot(Bot):

    def __init__(self, api_key, secret_key, passphrase, syms=SYMBOLS, is_sandbox_api=False, base_currency='USD', order_csv_path = './orders_tracking.csv'):
        super().__init__(api_key, secret_key, passphrase, syms, is_sandbox_api, base_currency)
        # Read dataframe from file if it exists and use that to initialize the product orders as well
        if order_csv_path is None:
            self.orders = pd.DataFrame(columns=['product_id', 'side', 'price', 'size', 'filled_size', 'corresponding_order', 'time', 'spread'])
            self.order_path = './orders_tracking.csv'
        else:
            self.order_path = order_csv_path
            self.orders = pd.read_csv(order_csv_path, index_col=0)
            ids = list(self.orders.index)
            for id in ids:
                private_pause()
                order_info = self.portfolio.auth.get_order(id)
                PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                if 'product_id' in order_info:
                    sym = order_info['product_id'].split('-')[0]
                    side = order_info['side']
                    product = self.portfolio.wallets[sym].product
                    product.orders[side][id] = order_info
                else:
                    self.orders.drop(id)

        # Create a dictionary for placeholder orders
        place_holder_orders = {}  # This variable stores orders yet to be placed {'BTC':{'side':'buy', 'price':9000, 'time':1582417135.072912}}
        for sym in syms:
            place_holder_orders[sym] = {'buy':None, 'sell':None}
        self.place_holder_orders = place_holder_orders

    # Methods to manage dataframe storing orders
    def add_order(self, id, sym, side, place_time, corresponding_buy_id, refresh=True, spread=np.nan):
        product = self.portfolio.wallets[sym].product

        if refresh:
            product.update_orders()
        order = None

        # If the id is in the stored orders grab the data
        if id in product.orders[side].keys():
            order = product.orders[side][id]
        # If the id is for a buy order check past data as well
        elif side == 'buy':
            try:
                private_pause()
                order = product.auth_client.get_order(id)
                PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                # Check if the buy order has already been taken care of
                if 'filled_size' not in order.keys():
                    print(sym + ' removed from tracking due to ' + order['message'] + '\n')
                    order = None
                else:
                    size = float(order['filled_size'])
                    if np.abs(float(corresponding_buy_id) - size) <= 2*self.portfolio.wallets[sym].product.crypto_res:
                        order = None
            except Exception as e:
                _ = print_err_msg('find new data', e, 1)
                order = None
        # Add the order info to the dataframe
        if (order is not None) and ('message' not in order.keys()):
            new_row_str_headers = ['product_id', 'side']
            new_row_float_headers = ['price', 'size', 'filled_size']
            new_row = []
            for header in new_row_str_headers:
                new_row.append(order[header])
            for header in new_row_float_headers:
                new_row.append(float(order[header]))
            new_row.append(corresponding_buy_id)
            new_row.append(place_time)
            new_row.append(spread)
            self.orders.loc[id] = new_row

    def cancel_single_order(self, id, remove_index=False):
        private_pause()
        response = self.portfolio.auth.cancel_order(id)
        PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        if type(response) == dict:
            response = response['message']
            print('Cannot cancel order due to ' + response)
            return False
        elif remove_index:
            self.orders = self.orders.drop(id)
            return True

    def update_id_in_order_df(self, id, sym, side, place_time, size, corresponding_buy_id=None, spread=np.nan):
        if side == 'sell':
            # For sell orders remove if the are no longer in the books
            stored_ids = self.orders.index
            if id in stored_ids:
                self.orders = self.orders.drop(id)
        else:
            # For buy orders remove if the are no longer in the books and the sell order for them has already gone through
            stored_ids = self.orders.index
            if (id in stored_ids):
                self.orders = self.orders.drop(id)
        self.add_order(id, sym, side, place_time, corresponding_buy_id, spread=spread)
        # Check if a sell order was cancelled, if so add its size to the buy irder corresponding order
        stored_ids = self.orders.index
        if (id not in stored_ids) and (corresponding_buy_id in stored_ids):
            if self.orders.loc[corresponding_buy_id]['corresponding_order']:
                self.orders.at[corresponding_buy_id, 'corresponding_order'] = float(self.orders.loc[corresponding_buy_id]['corresponding_order']) + size
            else:
                self.orders.at[corresponding_buy_id, 'corresponding_order'] = size

    def update_orders(self):
        updated_syms = []
        for id in self.orders.index:
            order = self.orders.loc[id]
            sym = order['product_id'].split('-')[0]
            side = order['side']
            corresponding_order_id = order['corresponding_order']
            place_time = order['time']
            size = order['size']
            spread = order['spread']
            self.update_id_in_order_df(id, sym, side, place_time, size, corresponding_buy_id=corresponding_order_id, spread=spread)

        self.orders.to_csv(self.order_path)

    def place_order(self, price, side, size, sym, post_only=True, time_out=False, stop_price=None):
        order_id = super().place_order(price, side, size, sym, post_only=post_only, time_out=time_out, stop_price=stop_price)
        self.add_order(order_id, sym, 'buy', time(), 0)

class SpreadBot(Bot):

    def determine_current_avg_price(self, sym):
        dt_str_format = '%Y-%m-%d %H:%M:%S %Z'
        t_len = 5*TRADE_LEN
        t_offset_str = offset_current_est_time(t_len, fmt=dt_str_format)
        cc = CryptoCompare(date_from=t_offset_str, exchange='Coinbase')
        data = cc.minute_price_historical(sym)[sym + '_close'].values
        sleep(0.3)
        coeff = np.polyfit(np.arange(0, len(data)), data, 1)
        price = np.polyval(coeff, len(data))

        return price

    def determine_price_based_on_std(self, sym, usd_available, side, std_mult=1.0, mu_mult=1.0):

        # initialize variables
        book_side, opposing_book_side, coeff = self.get_side_depedent_vars(side)

        wallet = self.portfolio.wallets[sym]
        mu, std, last_diff = wallet.product.get_mean_and_std_of_price_changes()
        if mu is None:
            return None, None, None, None, None
        mu *= mu_mult

        # Calculate the coefficient used to determine the which multiple of the std to use
        std_coeff = std_mult * self.settings.read_setting_from_file('std')

        # determine trade price
        std_offset = coeff * (std_coeff * std) + mu
        order_coeff = 1 + std_offset
        current_price = self.determine_current_avg_price(sym) #wallet.product.get_top_order(book_side)
        # if (side == 'buy') and ((last_diff-1) > np.abs(std_coeff * std + mu)):
        #     order_coeff = 1
        #     # current_price = wallet.product.get_top_order(opposing_book_side)
        # elif (coeff * last_diff < coeff * std_offset):
        #     order_coeff -= last_diff
        # else:
        #     order_coeff = 1
        #     current_price = wallet.product.get_top_order(opposing_book_side) + coeff * wallet.product.usd_res

        buy_price = order_coeff * current_price
        size = usd_available / buy_price

        return buy_price, wallet, size, std, mu

    def determine_price_based_on_fill_size(self, sym, usd_available, side, std_mult=1.0, mu_mult=1.0):
        # initialize variables
        book_side, opposing_book_side, coeff = self.get_side_depedent_vars(side)
        wallet = self.portfolio.wallets[sym]
        mu, std, existing_fill_size = wallet.product.get_mean_and_std_of_fill_sizes(side)
        mu *= mu_mult

        # Calculate the coefficient used to determine the which multiple of the std to use
        std_coeff = std_mult * self.settings.read_setting_from_file('std') # const factor of two due to low spreads

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

    def determine_trade_price(self, sym, usd_available, side, mean_multiplier=1):
        _, _, coeff = self.get_side_depedent_vars(side)
        price_p, wallet, size_p, std, mu = self.determine_price_based_on_std(sym, usd_available, side, mu_mult=mean_multiplier, std_mult=1.0)
        price_s, size_s = self.determine_price_based_on_fill_size(sym, usd_available, side, mu_mult=mean_multiplier, std_mult=1.0)
        return price_p, wallet, size_p, std, mu
        # if coeff * price_p > coeff * price_s: # Always use the more conservative price
        #     return price_p, wallet, size_p, std, mu
        # else:
        #     return price_s, wallet, size_s, std, mu

    def cancle_out_of_bound_buy_orders(self, top_syms):
        # Ensure to update portfolio value before running
        for sym in self.symbols:
            # Find price 2 standard deviations below the change (unlikely buy price to hit)
            wallet = self.portfolio.wallets[sym]
            _, _, last_diff = wallet.product.get_mean_and_std_of_price_changes()
            nominal_buy_price, _, _, std, mu = self.determine_trade_price(sym, 11, 'buy')
            std_coeff = self.settings.read_setting_from_file('std')
            high_scale_coeff = 1 + std_coeff * std_coeff
            low_scale_coeff = 1 - std_coeff * std_coeff

            # std_coeff = 1 - (3 * std_coeff * std) + mu * (mu < 0) # only factor in the mean when it gives a wider margin (make it harder to cancel an order due to short changes)
            if mu is None:
                self.cancel_out_of_bound_orders('buy', 0, sym)
            if (np.abs(last_diff) > (std + mu)) and (sym in top_syms):
                # Don't cancel orders due to one big jump, but do cancel orders that are not first choice
                continue

            # Cancel orders that are outside of a 2 std neighborhood from nominal
            self.cancel_out_of_bound_orders('buy', high_scale_coeff * nominal_buy_price, sym, widen_spread=True) # widen the range if the nominal price is 1 std below the current_price
            self.cancel_out_of_bound_orders('buy', low_scale_coeff * nominal_buy_price, sym,
                                            widen_spread=False)  # shorten the range if the nominal price is 1 std above the current price

    def cancel_old_orders(self):
        for sym in self.symbols:
            self.cancel_timed_out_orders('buy', TRADE_LEN * 60, sym)

    def sort_currencies(self, usd_available, print_sym):
        # setup
        ranking_dict = {}

        # create dictionary for symbols and relevant data
        for sym in self.symbols:
            if print_sym:
                print(sym)
            buy_price, wallet, size, std, mu = self.determine_trade_price(sym, usd_available, side='buy')
            if buy_price is None:
                continue
            sell_price, _, _, _, _ = self.determine_trade_price(sym, usd_available, side='sell', mean_multiplier=2) # Use mean_multiplier of 2 to account for movement after the buy
            current_price = wallet.product.get_top_order('bids')
            spread = calculate_spread(buy_price, sell_price)
            rank = (sell_price - current_price) / buy_price
            if (spread < MIN_SPREAD):
                rank = -1 # eliminate trades with low spreads
            ranking_dict[sym] = (spread, mu, buy_price, wallet, std, rank, size)

        # sort (by mean first then standard deviation)
        sorted_syms = sorted(ranking_dict.items(), key=itemgetter(1), reverse=True)

        return sorted_syms

    def rank_currencies(self, usd_available, print_sym=True, sym_ind=0):
        sorted_syms = self.sort_currencies(usd_available, print_sym)
        rank_arr = np.array([curr_sym[1][5] for curr_sym in sorted_syms])
        rank_std = np.std(rank_arr)
        min_rank = np.mean(rank_arr) - 2 * rank_std
        return_None = False
        if type(sym_ind) == int:
            top_sym_data = sorted_syms[sym_ind]
            top_sym = top_sym_data[0]
            mu = top_sym_data[1][1]
            buy_price = top_sym_data[1][2]
            wallet = top_sym_data[1][3]
            std = top_sym_data[1][4]
            spread = top_sym_data[1][0]
            size = top_sym_data[1][6]
            if spread < MIN_SPREAD:
                return_None = True
        else:
            top_sym = []
            spread = []
            buy_price = []
            wallet = []
            std = []
            mu = []
            size = []

            for ind in range(0, len(sorted_syms)):
                top_sym_data = sorted_syms[ind]
                if top_sym_data[1][0] < MIN_SPREAD:
                    continue
                if top_sym_data[1][5] < min_rank:
                    continue
                top_sym.append(top_sym_data[0])
                mu.append(top_sym_data[1][1])
                buy_price.append(top_sym_data[1][2])
                wallet.append(top_sym_data[1][3])
                std.append(top_sym_data[1][4])
                spread.append(top_sym_data[1][0])
                size.append(top_sym_data[1][6])

            if len(spread) == 0:
                return_None = True

        if not return_None:
            return buy_price, wallet, size, top_sym, std, mu, spread
        else:
            return None, None, None, None, None, None, None

    def place_order_for_nth_currency(self, buy_price, sell_price, wallet, size, std, mu, top_sym):
        spread = calculate_spread(buy_price, sell_price)
        # place order and record
        print('\n' + top_sym + ' Chosen as best buy')
        print('placing order\n' + 'price: ' + num2str(buy_price,
                                                      wallet.product.usd_decimal_num) + '\n' + 'size: ' + num2str(
            size, 3) + '\n' + 'std: ' + num2str(std, 6) + '\n' + 'mu: ' + num2str(mu, 8) + '\n' + 'projected spread: ' + num2str(spread, 6) + '\n')
        order_id = self.place_order(buy_price, 'buy', size, top_sym, post_only=False)
        if order_id is None:
            print('Buy Order rejected\n')
        else:
            print('Buy Order placed!\n')
            self.update_spread_prices_limits(buy_price, 'buy', top_sym)
            if spread < MIN_SPREAD:
                spread = MIN_SPREAD
                sell_price = spread * buy_price
            self.spread = spread
            self.update_spread_prices_limits(sell_price, 'sell', top_sym)

    def place_order_for_top_currencies(self):
        # Ensure to update portfolio value before running
        usd_hold = self.portfolio.get_usd_held()
        usd_available = self.portfolio.get_usd_available()
        desired_number_of_currencies = 5 # How many currecies (excluding USD) to hold
        sym_indices = list(range(0, desired_number_of_currencies)) # This chooses the indices to use for determining trades
        buy_prices, wallets, sizes, top_syms, stds, mus, spreads = self.rank_currencies(usd_available, print_sym=False, sym_ind=sym_indices)
        no_viable_trade = False

        # Cancel bad buy orders before continuing
        if top_syms is None:
            no_viable_trade = True
            top_syms = self.symbols  # If no viable trades are found then allow any symbol to remain
        if usd_hold > QUOTE_ORDER_MIN:
            self.cancel_old_orders()
            private_pause()
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)

        self.portfolio.update_value()
        # Check available cash after canceling the non_optimal buy orders and place the next order
        full_portfolio_value = self.get_full_portfolio_value()
        num_orders = len(top_syms)
        num_currencies_to_loop = np.min(np.array([len(top_syms) + 1, desired_number_of_currencies + 1]))
        fee_rate, tkr_fee = self.portfolio.get_fee_rate()
        self.update_min_spread(tkr_fee)
        for ind in range(1, num_currencies_to_loop):
            i = num_orders - ind
            usd_available = self.portfolio.get_usd_available()
            if (usd_available > QUOTE_ORDER_MIN):

                # Determine order size
                nominal_order_size = full_portfolio_value / desired_number_of_currencies
                if nominal_order_size < QUOTE_ORDER_MIN:
                    # never try to place an order smaller than the minimum
                    nominal_order_size = QUOTE_ORDER_MIN

                if nominal_order_size > (usd_available - QUOTE_ORDER_MIN):
                    # If placing the nominal order leaves an unusable amount of money then only use available
                    order_size = usd_available
                else:
                    order_size = nominal_order_size

                # Determine order properties
                if no_viable_trade:
                    pass #print('No viable trades found')
                else:
                    top_sym = top_syms[i]
                    buy_price, wallet, size, std, mu = self.determine_trade_price(top_sym, order_size, side='buy')
                    sell_price, _, _, _, _ = self.determine_trade_price(top_sym, order_size, side='sell')

                    # Always insure the order is small enough to go through in a reasonable amount of time
                    mean_size, _, _ = wallet.product.get_mean_and_std_of_fill_sizes('asks', weighted=False)
                    if order_size > (buy_price * mean_size):
                        order_size = buy_price * mean_size

                    if buy_price is None:
                        continue

                    existing_size = (wallet.value['SYM'])
                    orders = self.portfolio.wallets[top_sym].product.get_open_orders()
                    for order in orders:
                        if order['side'] == 'buy':
                            existing_size += float(order['size'])

                    amnt_held = existing_size * buy_price

                    # Scale size based on existing holdings
                    if amnt_held <= (order_size - QUOTE_ORDER_MIN):
                        size -= existing_size
                    # Don't place order if you already hold greater than or euqal to the nominal_order_size (within the QUOTE_ORDER_MIN)
                    else:
                        continue

                    # If the spread is too small don't place the order
                    spread = calculate_spread(buy_price, sell_price)
                    if spread < MIN_SPREAD:
                        print('Cannot place by order because projected sell price dropped\n')
                        continue

                    # Adjust the size to account for the fee rate
                    size = size / (1 + fee_rate)

                    # Place order
                    self.place_order_for_nth_currency(buy_price, sell_price, wallet, size, std, mu, top_sym)
                    self.portfolio.update_value()
            else:
                break

    def place_limit_sells(self):
        # Ensure to update portfolio value before running
        for sym in self.portfolio.wallets.keys():
            # Get relevant variables
            wallet = self.portfolio.wallets[sym]
            buy_price = self.spread_price_limits[sym]['buy']
            limit_price = self.spread_price_limits[sym]['sell']
            # cutoff_price, _ = self.determine_price_based_on_fill_size(sym, 11, 'sell', std_mult=3)
            alt_price, _ = self.determine_price_based_on_fill_size(sym, 11, 'sell', std_mult=3) # Check to see if the expected price has changed since the buy order was placed
            available = wallet.get_amnt_available('sell')

            if limit_price is None:
                continue

            # Ensure the portfolio does not fall below 7%
            spread = calculate_spread(buy_price, limit_price)
            current_price = wallet.product.get_top_order('bids')
            current_value = calculate_spread(buy_price, current_price)
            e_stop_trigger = 0.931
            e_stop_spread = 0.93

            if (current_value <= e_stop_trigger) and (limit_price > e_stop_spread * buy_price):
                limit_price = e_stop_spread * buy_price
                spread = calculate_spread(buy_price, limit_price)
                self.cancel_out_of_bound_orders('asks', limit_price, sym)
                print(sym + ' holdings have lost more than 7% of their purchase value, activating stop limit\n')
            elif spread < MIN_SPREAD:
                limit_price = MIN_SPREAD * buy_price
                spread = calculate_spread(buy_price, limit_price)
                print('Upping ' + sym + ' sell price to meet minnimum spread requriements\n')

            # Filter unnecessary currencies
            if (available < wallet.product.base_order_min):
                continue
            if (available * limit_price < QUOTE_ORDER_MIN):
                print(
                    'Cannot sell ' + sym + ' because available is less than minnimum Quote order. Manual sell required')
                continue

            print('Evaluating sell order for ' + sym)

            print('Placing ' + sym + ' spread sell order' + '\n' + 'price: ' + num2str(limit_price, wallet.product.usd_decimal_num) + '\n' + 'spread: ' + num2str(spread, 4))
            self.spread = spread
            self.update_spread_prices_limits(limit_price, 'sell', sym)
            order_id = self.place_order(limit_price, 'sell', available, sym, post_only=False)
            if order_id is None:
                print('Sell Order rejected\n')
            else:
                print('\nSell Order placed')

class PSMSpreadBot(SpreadBot):
    # class variables
    # propogator: the object that predicts future prices
    # predicted_prices: this contains the max, min, standard deviation of prices, and the mean offset for the true and predicted prices
    # del_len: the number of samples to average over for the price

    def __init__(self, api_key, secret_key, passphrase, syms=SYMBOLS, is_sandbox_api=False, base_currency='USD', slope_avg_len=5, train_len=480):
        super().__init__(api_key, secret_key, passphrase, syms, is_sandbox_api, base_currency)
        raw_data = {}
        self.fmt = '%Y-%m-%d %H:%M:%S %Z'
        self.train_len = train_len
        t_offset_str = offset_current_est_time(train_len, fmt=self.fmt)
        cc = CryptoCompare(date_from=t_offset_str, exchange='Coinbase')
        for sym in syms:
            data = cc.minute_price_historical(sym)[sym + '_close'].values
            raw_data[sym] = data

        raw_data_list = [raw_data[sym] for sym in self.symbols]
        self.raw_data = raw_data

        self.del_len = slope_avg_len
        self.propogator, coeff_list, shift_list = create_multifrequency_propogator_from_data(raw_data_list, self.symbols)
        self.predictions = {}
        self.reversed_predictions = {}
        self.coefficients = {}
        self.shifts = {}
        self.buy_cancel_times = {}

        for i in range(0, len(syms)):
            sym = syms[i]
            self.predictions[sym] = None # will use as max, min, true_std, mean_offset
            self.coefficients[sym] = coeff_list[i]
            self.shifts[sym] = shift_list[i]
            self.buy_cancel_times[sym] = TRADE_LEN

    def collect_next_data(self):
        # This method sets the propogator initial values to the most recent price
        raw_data = {}
        self.fmt = '%Y-%m-%d %H:%M:%S %Z'
        t_offset_str = offset_current_est_time(self.train_len, fmt=self.fmt)
        cc = CryptoCompare(date_from=t_offset_str, exchange='Coinbase')
        for sym in self.symbols:
            data = cc.minute_price_historical(sym)[sym + '_close'].values
            raw_data[sym] = data

        return raw_data

    def transform_single_sym(self, sym, data):
        transform_data = self.coefficients[sym] * data + self.shifts[sym]
        return transform_data

    def transform(self, raw_data):
        transform_data = {}
        syms = self.symbols
        for sym in syms:
            transform_data[sym] = self.transform_single_sym(sym, raw_data[sym])

        return transform_data

    def inverse_transform(self, raw_data):
        transform_data = {}
        syms = self.symbols
        for sym in syms:
            transform_data[sym] = (raw_data[sym] - self.shifts[sym]) / self.coefficients[sym]

        return transform_data

    def reset_propogator_start_point(self, raw_data):
        # This method sets the propogator initial values to the most recent price
        normalized_raw_data = self.inverse_transform(raw_data)
        normalized_raw_data_list = [normalized_raw_data[sym] for sym in self.symbols]
        # x0s = [x[-1] for x in normalized_raw_data_list]
        # y0s = [np.mean(np.diff(x[-self.del_len*(len(x) > self.del_len)::])) for x in normalized_raw_data_list]
        initial_polys = [np.polyfit(np.arange(0, len(x[-11::])), x[-11::], 1) for x in normalized_raw_data_list]
        x0s = [np.polyval(x, 11) for x in initial_polys]
        y0s = [x[0] for x in initial_polys]
        self.propogator.reset(x0s=x0s, y0s=y0s)

    def get_new_propogator(self, raw_data_list, verbose_on=False):
        # create the new propogator and set up variables
        self.propogator, coeff_list, shift_list = create_multifrequency_propogator_from_data(raw_data_list, self.symbols)
        self.predictions = {}
        self.coefficients = {}
        self.shifts = {}
        self.errors = {}
        syms = self.symbols

        # reset predictions and set new transformations
        for i in range(0, len(syms)):
            sym = syms[i]
            self.predictions[sym] = None  # will use as max, min, true_std, mean_offset
            self.coefficients[sym] = coeff_list[i]
            self.shifts[sym] = shift_list[i]

        # determine propogator error
        # Setup Initial Variables
        time_arr = np.arange(0, TRADE_LEN, PSM_EVAL_STEP_SIZE)
        step_size = PSM_EVAL_STEP_SIZE
        polynomial_order = 10

    def predict(self, verbose_on=False, get_new_propogator=True):
        # Setup Initial Variables
        time_arr = np.arange(0, TRADE_LEN)
        step_size = PSM_EVAL_STEP_SIZE
        polynomial_order = 5

        # Collect data and create propogator
        raw_data = self.collect_next_data()
        self.raw_data = raw_data
        raw_data_list = [raw_data[sym] for sym in self.symbols]
        self.get_new_propogator(raw_data_list, verbose_on=verbose_on)
        self.reset_propogator_start_point(raw_data)
        syms = self.symbols
        predictions = {}
        reversed_predictions = {}
        # transformed_raw_data = self.inverse_transform({syms[i]: raw_data_list[i] for i in range(0, len(raw_data_list))})

        # Predict
        for i in range(0, len(syms)):
            sym = syms[i]
            coeff = self.coefficients[sym]
            shift = self.shifts[sym]
            x, _ = self.propogator.evaluate_nth_polynomial(time_arr, step_size, polynomial_order, n=i + 1, verbose=verbose_on)
            self.errors[sym], x_reversed, real_data = self.propogator.err(step_size, polynomial_order, i + 1, coeff,
                                                         shift, verbose=verbose_on, data_len=TRADE_LEN)
            predictions[sym] = x - x[0] + real_data[-1]
            reversed_predictions[sym] = x_reversed - x[0] + real_data[-1]#x_reversed - np.mean(x_reversed) + np.mean(real_data)

        transformed_raw_data = self.inverse_transform(raw_data)
        self.predictions = self.transform(predictions)
        self.reversed_predictions = self.transform(reversed_predictions)

    def determine_price_based_on_std(self, sym, usd_available, side, std_mult=1.0, mu_mult=1.0, max_predict_len=TRADE_LEN):

        # initialize variables
        book_side, opposing_book_side, coeff = self.get_side_depedent_vars(side)
        wallet = self.portfolio.wallets[sym]
        predicted_evolution = self.predictions[sym][0:max_predict_len]
        mu, std, last_diff = wallet.product.get_psm_mean_and_std_of_price_changes(predicted_evolution, self.raw_data[sym])
        current_price = wallet.product.get_top_order(opposing_book_side)
        buy_std_coeff = std_mult * self.settings.read_setting_from_file('buy std')
        sell_std_coeff = std_mult * self.settings.read_setting_from_file('sell std')
        if mu is None:
            return None, None, None, None, None
        mu *= mu_mult

        # Calculate price using the predicted price
        sell_ind = np.argmax(predicted_evolution) # Only buy currencies if you predict that the mean will be soon
        fall_len=90
        data_fit = np.polyfit(np.arange(0, fall_len), self.raw_data[sym][-fall_len::], 1)
        mean_mvt = data_fit[0]
        residual = self.raw_data[sym][-fall_len::] - np.polyval(data_fit, np.arange(0, fall_len))
        std_err = np.std(residual)/(2*fall_len)
        if np.argmin(predicted_evolution) > sell_ind:
            return None, None, None, None, None
        if side == 'buy':
            if sell_ind > 0:
                mu *= TRADE_LEN * (np.argmin(predicted_evolution[0:sell_ind]) / len(predicted_evolution[0:sell_ind]))
                buy_price = buy_std_coeff * np.min(predicted_evolution[0:sell_ind] - current_price)+current_price
            else:
                buy_price = current_price
                mu = 0

        else:
            buy_price = sell_std_coeff * np.max(predicted_evolution-current_price)+current_price
            mu *= TRADE_LEN * (sell_ind / len(predicted_evolution))

        if coeff * buy_price < coeff * current_price:
            buy_price = current_price


        size = usd_available / buy_price

        return buy_price, wallet, size, std, mu

    def sort_currencies(self, usd_available, print_sym):
        # setup
        ranking_dict = {}

        # create dictionary for symbols and relevant data
        for sym in self.symbols:
            if print_sym:
                print(sym)
            buy_price, wallet, size, std, mu = self.determine_trade_price(sym, usd_available, side='buy')
            if buy_price is None:
                continue
            sell_price, _, _, _, _ = self.determine_trade_price(sym, usd_available, side='sell', mean_multiplier=2) # Use mean_multiplier of 2 to account for movement after the buy
            current_price = wallet.product.get_top_order('bids')
            spread = calculate_spread(buy_price, sell_price)
            rank = 1/self.errors[sym]#(sell_price - current_price) / buy_price
            ranking_dict[sym] = (spread, mu, buy_price, wallet, std, rank, size)

        # sort (by mean first then standard deviation)
        sorted_syms = sorted(ranking_dict.items(), key=itemgetter(1), reverse=True)

        return sorted_syms

    def cancel_old_orders(self):

        for sym in self.symbols:
            # Timed out orders
            self.cancel_timed_out_orders('buy', self.buy_cancel_times[sym] * 60, sym)

class PSMPredictBot(PSMSpreadBot):

    def __init__(self, api_key, secret_key, passphrase, syms=SYMBOLS, is_sandbox_api=False, base_currency='USD', order_csv_path = './orders_tracking.csv'):
        super().__init__(api_key, secret_key, passphrase, syms, is_sandbox_api, base_currency)
        # Read dataframe from file if it exists and use that to initialize the product orders as well
        if order_csv_path is None:
            self.orders = pd.DataFrame(columns=['product_id', 'side', 'price', 'size', 'filled_size', 'corresponding_order', 'time', 'spread', 'error'])
            self.order_path = './orders_tracking.csv'
        else:
            self.order_path = order_csv_path
            self.orders = pd.read_csv(order_csv_path, index_col=0)
            ids = list(self.orders.index)
            for id in ids:
                private_pause()
                order_info = self.portfolio.auth.get_order(id)
                PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                if 'product_id' in order_info:
                    sym = order_info['product_id'].split('-')[0]
                    side = order_info['side']
                    product = self.portfolio.wallets[sym].product
                    product.orders[side][id] = order_info
                else:
                    self.orders.drop(id)

        # Create a dictionary for placeholder orders
        place_holder_orders = {}  # This variable stores orders yet to be placed {'BTC':{'side':'buy', 'price':9000, 'time':1582417135.072912}}
        for sym in syms:
            place_holder_orders[sym] = {'buy':None, 'sell':None}
        self.place_holder_orders = place_holder_orders

    # Methods to manage dataframe storing orders
    def add_order(self, id, sym, side, place_time, corresponding_buy_id, refresh=True, spread=None, err=None):
        product = self.portfolio.wallets[sym].product

        if refresh:
            refresh_err_count = 0
            for i in range(0, 10):
                try:
                    product.update_orders()
                except Exception as e:
                    refresh_err_count = print_err_msg('Order refresh', e, refresh_err_count)
                    if i==9:
                        print(sym + ' order id ' + id + ' did not save')

        order = None

        # If the id is in the stored orders grab the data
        if id in product.orders[side].keys():
            order = product.orders[side][id]
        # If the id is for a buy order check past data as well
        elif side == 'buy':
            err_count = 0
            for i in range(0, 10):
                try:
                    private_pause()
                    order = product.auth_client.get_order(id)
                    PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                    # Check if the buy order has already been taken care of
                    if 'filled_size' not in order.keys():
                        print(sym + ' removed from tracking due to ' + order['message'] + '\n')
                        order = None
                    else:
                        size = float(order['filled_size'])
                        if np.abs(float(corresponding_buy_id) - size) <= 2 * self.portfolio.wallets[sym].product.crypto_res:
                            order = None
                    break
                except Exception as e:
                    err_count = print_err_msg('Add order', e, err_count)
                    if i==9:
                        print(sym + ' order id ' + id + ' did not save')
        # Add the order info to the dataframe
        if (order is not None) and ('message' not in order.keys()):
            new_row_str_headers = ['product_id', 'side']
            new_row_float_headers = ['price', 'size', 'filled_size']
            new_row = []
            for header in new_row_str_headers:
                new_row.append(order[header])
            for header in new_row_float_headers:
                new_row.append(float(order[header]))
            new_row.append(corresponding_buy_id)
            new_row.append(place_time)
            new_row.append(spread)
            new_row.append(err)
            self.orders.loc[id] = new_row

    def cancel_single_order(self, id, remove_index=False):
        private_pause()
        response = self.portfolio.auth.cancel_order(id)
        PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        if type(response) == dict:
            response = response['message']
            print('Cannot cancel order due to ' + response)
            return False
        elif remove_index:
            self.orders = self.orders.drop(id)
            return True

    def update_id_in_order_df(self, id, sym, side, place_time, size, corresponding_buy_id=None, spread=None, err=None):
        if side == 'sell':
            # For sell orders remove if the are no longer in the books
            stored_ids = self.orders.index
            if id in stored_ids:
                self.orders = self.orders.drop(id)
        else:
            # For buy orders remove if the are no longer in the books and the sell order for them has already gone through
            stored_ids = self.orders.index
            if (id in stored_ids):
                self.orders = self.orders.drop(id)

        self.add_order(id, sym, side, place_time, corresponding_buy_id, spread=spread, err=err)

        # Check if a sell order was cancelled, if so add its size to the buy irder corresponding order
        stored_ids = self.orders.index
        if (id not in stored_ids) and (corresponding_buy_id in stored_ids):
            if self.orders.loc[corresponding_buy_id]['corresponding_order']:
                self.orders.at[corresponding_buy_id, 'corresponding_order'] = float(self.orders.loc[corresponding_buy_id]['corresponding_order']) + size
            else:
                self.orders.at[corresponding_buy_id, 'corresponding_order'] = size

    def update_orders(self):
        updated_syms = []
        for id in self.orders.index:
            order = self.orders.loc[id]
            sym = order['product_id'].split('-')[0]
            side = order['side']
            corresponding_order_id = order['corresponding_order']
            place_time = order['time']
            size = order['size']
            spread = order['spread']
            err = order['error']
            self.update_id_in_order_df(id, sym, side, place_time, size, corresponding_buy_id=corresponding_order_id, spread=spread, err=err)

        self.orders.to_csv(self.order_path)

    def check_if_price_is_abve_std_coeff_from_mean(self, current_price, sym, std_coeff=0.75):
        # This method checks if the buy price is at least 1 standard deviation below the mean price
        raw_data = self.raw_data[sym]
        t = np.arange(0, len(raw_data))
        lin_fit = np.polyfit(t, raw_data, 1)
        mean_prices = np.polyval(lin_fit, t)
        mean_price = mean_prices[-1]
        residuals = raw_data - mean_prices
        expected_price_err = np.mean(residuals) + std_coeff * np.std(residuals)
        if current_price >= (mean_price - expected_price_err):
            # This means the proce is too high and the order should not be placed
            return True
        else:
            # This means the proce is in the right range
            return False

    def is_buy_order_fill_likely(self, current_price, sym, std_coeff=0.75):
        return self.check_if_price_is_abve_std_coeff_from_mean(current_price, sym, std_coeff=std_coeff)

    def is_sell_order_fill_unlikely(self, current_price, sym, std_coeff=0.75):
        return self.check_if_price_is_abve_std_coeff_from_mean(current_price, sym, std_coeff=-std_coeff)

    def cancel_old_orders(self):
        # First remove old placeholder orders
        for sym in self.symbols:
            for side in ('buy', 'sell'):
                placeholder_order = self.place_holder_orders[sym][side]
                if not placeholder_order is None:
                    placement_time = placeholder_order['time']
                    if (time() - placement_time) > TRADE_LEN * 60: self.place_holder_orders[sym][side]=None

        # Next remove old current orders
        for id in self.orders.index:
            # Only remove sell orders if they are no longer expected to be met
            order_time = float(self.orders.loc[id]['time'])
            if (self.orders.loc[id]['side'] == 'sell') and ((time() - order_time) > TRADE_LEN * 60):
                corresponding_id = self.orders.loc[id]['corresponding_order']
                sym = self.orders.loc[id]['product_id'][0:-4]
                new_sell_price, _, _, _, _ = self.determine_trade_price(sym, 1, side='sell')
                if new_sell_price is None:
                    continue

                # Is the new price far enough above the mean to be deemed unlikely to reach
                is_selling_unlikely = self.is_sell_order_fill_unlikely(new_sell_price, sym, std_coeff=1)

                # Is the new price far enough above the mean that even selling at a loss would be considered acceptable
                can_sell_at_a_loss = self.is_sell_order_fill_unlikely(new_sell_price, sym, std_coeff=1.5)

                # Will the buy price be within the standard variation assumed
                is_buying_at_profit_likely = self.is_buy_order_fill_likely(new_sell_price * (2 - MIN_SPREAD), sym)

                # Check if new price will still make a profit
                buy_price = self.orders.loc[corresponding_id]['price']
                current_spread = self.orders.loc[corresponding_id]['spread']
                new_spread = calculate_spread(buy_price, new_sell_price)

                can_lower_price_below_MIN_SPREAD = False#is_selling_unlikely and is_buying_at_profit_likely and (new_spread < current_spread)
                can_raise_price = (not is_selling_unlikely) and is_buying_at_profit_likely and (new_spread > current_spread)
                can_lower_price =  (new_spread > MIN_SPREAD) and (new_spread < current_spread)
                can_sell_at_a_loss = False#can_sell_at_a_loss and is_buying_at_profit_likely and (new_spread < current_spread)

                if  can_raise_price or can_lower_price or can_lower_price_below_MIN_SPREAD or can_sell_at_a_loss:
                    if can_lower_price:
                        print('Lowering sell price for ' + sym)
                    elif can_lower_price_below_MIN_SPREAD:
                        print('Lowering sell price for ' + sym + ' BELOW defined profit margins')
                    elif can_sell_at_a_loss:
                        print('Lowering sell price for ' + sym + ' to sell at a LOSS')
                    else:
                        print('Raising sell price for ' + sym)
                    self.cancel_single_order(id, remove_index=True)
                    self.orders.at[corresponding_id, 'spread'] = new_spread

            elif self.orders.loc[id]['filled_size']==self.orders.loc[id]['size']:
                continue
            elif (time() - order_time) > TRADE_LEN * 60:
                self.orders.at[id,'size'] = self.orders.loc[id]['filled_size']
                self.cancel_single_order(id)

    def place_order_for_nth_currency(self, buy_price, sell_price, wallet, size, std, mu, top_sym, err=None):
        # This method creates placeholder orders
        spread = calculate_spread(buy_price, sell_price)
        self.place_holder_orders[top_sym]['buy'] = {'price':buy_price, 'size':size, 'time':time(), 'spread':spread, 'error':err}
        print('\n' + top_sym + ' Chosen as best buy')
        print('watching\n' + 'price: ' + num2str(buy_price,
                                                      wallet.product.usd_decimal_num) + '\n' + 'size: ' + num2str(
            size, 3) + '\n' + 'std: ' + num2str(std, 6) + '\n' + 'mu: ' + num2str(mu, 8) + '\n' + 'projected spread: ' + num2str(spread, 6) + '\n')

    def buy_place_holders(self):
        com_wallet = self.portfolio.get_common_wallet()
        com_wallet.update_value()
        available = com_wallet.get_amnt_available('buy')
        mkr_fee, tkr_fee = self.portfolio.get_fee_rate()
        if available > QUOTE_ORDER_MIN:
            for sym in self.place_holder_orders.keys():
                order = self.place_holder_orders[sym]['buy']
                # Skip if there is no order
                if order is None: continue

                # Setup order variables
                nominal_price = order['price']
                nominal_size = order['size']
                wallet = self.portfolio.wallets[sym]
                current_price = wallet.product.get_top_order('asks')

                # Determine whether the price is low enough, if so set the limit and stop prices
                if current_price < nominal_price * (1 - STOP_SPREAD):
                    limit_price = current_price * (1 + STOP_SPREAD)
                    stop_price = current_price * (1 + STOP_SPREAD)
                else:
                    limit_price = nominal_price
                    stop_price = None

                # Check to see risk of not filling sell order
                product = self.portfolio.wallets[sym].product
                max_sell_price = product.detect_whale_price()
                # dont_buy = self.is_buy_order_fill_likely(nominal_price, sym)
                max_spread = calculate_spread(nominal_price, max_sell_price)

                if (max_spread < MIN_SPREAD): # or dont_buy:
                    continue
                elif (max_spread < order['spread']):
                    order['spread'] = max_spread

                # Scale the size based on the new price
                size = ( nominal_price * nominal_size ) / limit_price
                # Only buy the determined size if it's less than available, else buy all available
                if size * (limit_price*(1 + tkr_fee)) > available:
                    size = available / (limit_price*(1 + tkr_fee))

                #Place the order and print the status
                order_id = self.place_order(limit_price, 'buy', size, sym, post_only=False, stop_price=stop_price)
                if order_id is None:
                    print(sym + ' buy Order rejected\n')
                else:
                    print(sym + ' buy Order placed\n')

                    self.add_order(order_id, sym, 'buy', time(), 0, spread=order['spread'], err=self.errors[sym])

                    self.place_holder_orders[sym]['buy'] = None
                    com_wallet.update_value()
                    available = com_wallet.get_amnt_available('buy')

    def place_order_for_top_currencies(self):
        # Ensure to update portfolio value before running
        self.update_orders()
        usd_hold = self.portfolio.get_usd_held()
        usd_available = self.portfolio.get_usd_available()
        desired_number_of_currencies = 5 # How many currecies (excluding USD) to hold
        sym_indices = list(range(0, desired_number_of_currencies)) # This chooses the indices to use for determining trades
        buy_prices, wallets, sizes, top_syms, stds, mus, spreads = self.rank_currencies(usd_available, print_sym=False, sym_ind=sym_indices)
        no_viable_trade = False

        # Cancel bad buy orders before continuing
        self.cancel_old_orders()
        if top_syms is None:
            no_viable_trade = True
            top_syms = self.symbols  # If no viable trades are found then allow any symbol to remain
        if usd_hold > QUOTE_ORDER_MIN:
            private_pause()
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)

        # Check available cash after canceling the non_optimal buy orders and place the next order
        self.portfolio.update_value()
        full_portfolio_value = self.get_full_portfolio_value()
        num_orders = len(top_syms)
        num_currencies_to_loop = np.min(np.array([len(top_syms) + 1, desired_number_of_currencies + 1]))
        # Update fee rate
        mkr_fee, tkr_fee = self.portfolio.get_fee_rate()
        self.update_min_spread(mkr_fee, tkr_fee=tkr_fee)

        for ind in range(1, num_currencies_to_loop):
            i = num_orders - ind
            usd_available = self.portfolio.get_usd_available()
            top_sym = top_syms[i]
            current_placeholder = self.place_holder_orders[top_sym]['buy']
            if current_placeholder is not None:
                # Don't have more than one placeholder order out at a time
                continue
            if self.errors[top_sym] > 0.05:
                continue
            # Only continue if there are viable trades
            if (usd_available > QUOTE_ORDER_MIN) and (not no_viable_trade):

                # Determine order size
                nominal_order_size = full_portfolio_value / desired_number_of_currencies
                if nominal_order_size < QUOTE_ORDER_MIN:
                    # never try to place an order smaller than the minimum
                    nominal_order_size = QUOTE_ORDER_MIN

                if nominal_order_size > (usd_available - QUOTE_ORDER_MIN):
                    # If placing the nominal order leaves an unusable amount of money then only use available
                    order_size = usd_available
                else:
                    order_size = nominal_order_size

                # Determine order properties
                buy_price, wallet, size, std, mu = self.determine_trade_price(top_sym, order_size, side='buy')
                sell_price, _, _, _, _ = self.determine_trade_price(top_sym, order_size, side='sell')
                dont_buy = self.is_sell_order_fill_unlikely(sell_price, sym, std_coeff=1)
                if dont_buy:
                    continue

                # Always insure the order is small enough to go through in a reasonable amount of time
                mean_size, _, _ = wallet.product.get_mean_and_std_of_fill_sizes('asks', weighted=False)
                if order_size > (buy_price * mean_size):
                    order_size = buy_price * mean_size

                if buy_price is None:
                    continue

                existing_size = (wallet.value['SYM'])
                orders = self.portfolio.wallets[top_sym].product.get_open_orders()
                for order in orders:
                    if order['side'] == 'buy':
                        existing_size += float(order['size'])

                amnt_held = existing_size * buy_price

                # Scale size based on existing holdings
                if amnt_held <= (order_size - QUOTE_ORDER_MIN):
                    size -= existing_size
                # Don't place order if you already hold greater than or euqal to the nominal_order_size (within the QUOTE_ORDER_MIN)
                else:
                    continue

                # If the spread is too small don't place the order
                spread = calculate_spread(buy_price, sell_price)
                if spread < MIN_SPREAD:
                    print('Cannot place by order because projected sell price dropped\n')
                    continue

                # Adjust the size to account for the fee rate
                size = size / (1 + tkr_fee)

                # Place order
                self.place_order_for_nth_currency(buy_price, sell_price, wallet, size, std, mu, top_sym)
                self.portfolio.update_value()
            else:
                break
            self.buy_place_holders()

    def emergency_sell(self, id, current_price):
        order = self.orders.loc[id]
        buy_price = order['price']
        sym = order['product_id'].split('-')[0]
        # Check if the prices has fallen too far
        current_spread = calculate_spread(buy_price, current_price)
        if current_spread < 0.922:
            # Find if any live sells that exist for this order and cancel them
            for sell_id in self.orders.index:
                if self.orders.loc[sell_id]['corresponding_order'] == id:
                    did_cancel = self.cancel_single_order(sell_id, remove_index=True)
            print('Placed emergency stop loss for ' + sym + ' due to losses beyond 5%')
            self.orders.at[id, 'corresponding_order'] = self.orders.loc[id]['filled_size']
            _ = self.place_order(buy_price*0.92, 'sell', order['size'], sym, post_only=False, stop_price=buy_price*0.921)
            # Don't add the order to the tracker. It can only be cancelled by a human

    def place_sell_order(self, sym, limit_price, available, wallet, spread, buy_id, order_type='limit', stop=None):
        order_id = self.place_order(limit_price, 'sell', available, sym, post_only=False, stop_price=stop)
        print('Placing ' + sym + ' ' + order_type + ' sell order' + '\n' + 'price: ' + num2str(limit_price,
                                                                                               wallet.product.usd_decimal_num) + '\n' + 'spread: ' + num2str(
            spread, 4))
        if order_id is None:
            print('Sell Order rejected\n')
        else:
            print('\nSell Order placed\n')
            # Do not refresh the orders just in case the sell order was filled immediately
            self.add_order(order_id, sym, 'sell', time(), buy_id, refresh=False, err=self.orders.loc[buy_id]['error'])


        return order_id

    def place_limit_sells(self):
        # Ensure to update portfolio value before running

        order_index = list(self.orders.index)
        for id in order_index:
            if id not in self.orders.index:
                continue
            order = self.orders.loc[id]
            sym = order['product_id'].split('-')[0]
            wallet = self.portfolio.wallets[sym]
            filled = order['filled_size']
            size = order['size']
            buy_price = order['price']
            if (order['side']=='sell'):
                continue
            # If an order is placed without a spread ignore it
            if np.isnan(order['spread']):
                continue

            current_price = wallet.product.get_top_order('bids')
            current_spread = calculate_spread(buy_price, current_price)
            if (current_spread < 0.995) and (filled==0):
                # If the current price is below the buy price and the order has not been filled then it should be a stop
                # limit order, If it has fallen half a percent from the expected price cancel it
                self.cancel_single_order(id)
            # # Check if the currency has fallen a signifigant amount
            # self.emergency_sell(id, current_price)
            order = self.orders.loc[id] # Reiniate the order in case an emergency sell occured
            corresponding_size = float(order['corresponding_order'])
            if corresponding_size:
                filled -= corresponding_size # Account for already completed sells
            # Skip orders that are not relevant
            if (filled < wallet.product.base_order_min):
                continue
            # Determine whether or not all of the filled size is accounted for, if so continue
            # Setup variables
            already_handled_size = 0
            existing_ids = []
            existing_prices = []

            # Loop through current orders to find the ones that were to handle this buy
            for order_id in self.orders.index:
                order_dat = self.orders.loc[order_id]
                if order_dat['corresponding_order']==id:
                    already_handled_size+=order_dat['size']
                    existing_ids.append(order_id)
                    existing_prices.append(order_dat['price'])
            if len(existing_ids) > 0:
                if len(existing_ids) > 1: # If there is more than one sell order cancel them and consolidate
                    for order_id in existing_ids: self.cancel_single_order(order_id, remove_index=True)
                elif order['spread'] <= MAX_LIMIT_SPREAD: # Don't cancel fixed limit orders
                    continue
                elif (np.abs(already_handled_size-filled) >= 2*wallet.product.crypto_res) or (np.abs(existing_prices[0]-current_price) >= 0.002*current_price):
                    # If the price has moved out of bounds or the existing orders do not account for the entire value
                    for order_id in existing_ids: self.cancel_single_order(order_id, remove_index=True)
                else: # If an order already exists and meets the criteria then continue
                    continue
            # If the expected spread is small enough just place a regular limit order
            elif order['spread'] <= MAX_LIMIT_SPREAD:
                wallet.update_value()
                available = wallet.get_amnt_available('sell')
                if available > filled:  # Only sell for this order
                    available = filled
                limit_price = order['spread'] * buy_price
                # if the amount available is too small to sell
                if ((available < wallet.product.base_order_min) or (available * limit_price < QUOTE_ORDER_MIN)):
                    # print message and remove order if there is not enough remaining in the buy order to sell
                    if ((available+size-filled) < wallet.product.base_order_min):
                        print('Cannot sell ' + sym + ' because available is less than minnimum allowable order size. Manual sell required. Removing order ' + id + ' from tracking')
                        self.orders = self.orders.drop(id)
                    continue
                self.place_sell_order(sym, limit_price, available, wallet, order['spread'], id, order_type='limit')
                continue


            # Determine the sell price
            if current_spread < order['spread']:
                continue
            else:
                limit_price = current_price * (1 - 1.5*STOP_SPREAD)
                stop_price = current_price * (1 - STOP_SPREAD)

            wallet.update_value()
            available = wallet.get_amnt_available('sell')
            if available > filled: # Only sell for this order
                available = filled

            spread = calculate_spread(buy_price, limit_price)
            # Filter unnecessary currencies
            if (available < wallet.product.base_order_min) or (available * limit_price < QUOTE_ORDER_MIN):
                # print message and remove order if there is not enough remaining in the buy order to sell
                if ((available + size - filled) < wallet.product.base_order_min):
                    print(
                        'Cannot sell ' + sym + ' because available is less than minnimum allowable order size. Manual sell required. Removing order ' + id + ' from tracking')
                    self.orders = self.orders.drop(id)
                continue

            self.spread = spread
            self.update_spread_prices_limits(limit_price, 'sell', sym)
            self.place_sell_order(sym, limit_price, available, wallet, spread, id, order_type='stop limit', stop=stop_price)

class PredictBot(PSMPredictBot):
    def __init__(self, api_key, secret_key, passphrase, syms=SYMBOLS, is_sandbox_api=False, base_currency='USD', order_csv_path = './orders_tracking.csv'):
        super().__init__(api_key, secret_key, passphrase, syms, is_sandbox_api, base_currency, order_csv_path)

        # Setup attributes to handle data from the websocket
        products = (sym+'-'+base_currency for sym in syms)
        self.order_errors = {}
        self.queque_dict = {sym:Queue() for sym in syms}
        self.socket = Websocket(self.queque_dict, products=products)

    def listen_for_price(self):
        self.websocket_thread = Thread(target=self.socket.start)
        self.websocket_thread.start()

    def stop_listening(self):
        self.socket.close()
        self.websocket_thread.join()

    def is_possible_bubble(self, sym):
        fit_coeff = {
            ticker: np.polyfit(np.arange(0, len(self.raw_data[ticker])), self.raw_data[ticker] / np.mean(self.raw_data[ticker]), 1) for
            ticker in self.raw_data.keys()
        }
        current_mvt = fit_coeff[sym][0]
        mean_mvt = np.array([fit_coeff[ticker][0] for ticker in fit_coeff.keys()])
        lower_cutoff = -4.167e-05
        upper_cutoff = np.mean(mean_mvt) + 2 * np.std(mean_mvt)
        if (current_mvt < lower_cutoff) or (current_mvt > upper_cutoff):
            return True
        return False

    def cancel_old_orders(self):
        # First remove old placeholder orders
        for sym in self.symbols:
            for side in ('buy', 'sell'):
                placeholder_order = self.place_holder_orders[sym][side]
                if not placeholder_order is None:
                    placement_time = placeholder_order['time']
                    if (time() - placement_time) > ORDER_TIMEOUT: self.place_holder_orders[sym][side] = None

        # Next remove old current orders
        for id in self.orders.index:
            # Only remove sell orders if they are no longer expected to be met
            order_time = float(self.orders.loc[id]['time'])
            if (self.orders.loc[id]['side'] == 'sell') and ((time() - order_time) > ORDER_TIMEOUT):
                corresponding_id = self.orders.loc[id]['corresponding_order']
                sym = self.orders.loc[id]['product_id'][0:-4]
                old_err = self.orders.loc[id]['error']
                new_err = self.errors[sym]
                new_sell_price, _, _, _, _ = self.determine_trade_price(sym, 1, side='sell', max_predict_len=NEAR_PREDICTION_LEN)
                product = self.portfolio.wallets[sym].product

                if (new_sell_price is None) or ((old_err < new_err) and (not (time() - order_time) > 6*ORDER_TIMEOUT)):
                    continue

                # Check if new price will still make a profit
                buy_price = self.orders.loc[corresponding_id]['price']
                whale_price = product.detect_whale_price()
                new_spread = calculate_spread(buy_price, new_sell_price)
                current_spread = self.orders.loc[corresponding_id]['spread']
                max_spread = calculate_spread(buy_price, whale_price) - WHALE_MARGIN

                # Don't move the sell price above a whale order
                if (new_spread > current_spread) and (new_spread >= max_spread):
                    continue
                if new_spread < MIN_SPREAD:
                    new_spread = MIN_SPREAD
                # Don't cancel the old order if the new order is identical
                if np.isclose(new_spread, current_spread, atol=1e-5):
                    continue

                self.cancel_single_order(id, remove_index=True)
                self.orders.at[corresponding_id, 'spread'] = new_spread

            elif self.orders.loc[id]['filled_size'] == self.orders.loc[id]['size']:
                continue
            elif (time() - order_time) > ORDER_TIMEOUT:
                self.orders.at[id, 'size'] = self.orders.loc[id]['filled_size']
                self.cancel_single_order(id)

    def buy_single_place_holder(self, com_wallet, tkr_fee, available_queue : Queue, sym, time_out):
        try:
            # This method turns place holders into live orders once the price is near the predicted
            order = self.place_holder_orders[sym]['buy']
            t0 = time()
            # Skip if there is no order
            if order is None:
                return

            # Setup order variables
            price_bounce_tracker = 0
            nominal_price = order['price']
            nominal_size = order['size']
            wallet = self.portfolio.wallets[sym]

            # Periodically check to see if the price is near the purchase price
            q = self.queque_dict[sym]
            order_data = safe_get(q, time_out, sym + ' initial get timeout')
            if order_data is None:
                return
            price = float(order_data['price'])
            print('Watching ' + sym + ' for buy trigger')
            sleep(0.01)
            while (price > nominal_price) and ((time() - t0) < time_out):
                order_data = safe_get(q, time_out, sym + ' trigger get timeout')
                if order_data is None:
                    return
                price = float(order_data['price'])
                # print('trigger ' + sym + ' price: ' + num2str(price))
            if (time() - t0) >= time_out:
                print(sym + ' watch timeout')
                sleep(0.01)
                return

            print('Watching ' + sym + ' for bounce')
            sleep(0.01)
            # Periodically check to see if the price has risen
            while (price_bounce_tracker < 2) and ((time() - t0) < time_out):
                last_price = price
                order_data = safe_get(q, time_out, sym + ' bounce get timeout')
                if order_data is None:
                    return
                price = float(order_data['price'])
                # print('bounce ' + sym + ' price: ' + num2str(price))

                # Check to see if the price has been rising for multiple trades
                if (order_data['side'] == 'buy') and (price >= last_price):
                    price_bounce_tracker += 1
                else:
                    price_bounce_tracker -= 1

            # If the conditions were not met, don't place a buy order
            if (time() - t0) >= time_out:
                print(sym + ' watch timeout')
                sleep(0.01)
                return

            current_price = float(order_data["best_bid"])
            # If the price is lower than nominal place at that price
            if current_price < nominal_price:
                limit_price = current_price
            else:
                limit_price = nominal_price
            stop_price = None

            # Check to see risk of not filling sell order
            product = self.portfolio.wallets[sym].product
            max_sell_price = product.detect_whale_price()
            # dont_buy = self.is_buy_order_fill_likely(nominal_price, sym)
            max_spread = calculate_spread(limit_price, max_sell_price) - WHALE_MARGIN

            if (max_spread < MIN_SPREAD):  # or dont_buy:
                print('Cannot buy ' + sym + ' due to whale')
                sleep(0.01)
                return
            elif (max_spread < order['spread']):
                order['spread'] = max_spread

            # Scale the size based on the new price
            size = (nominal_price * nominal_size) / limit_price
            # Only buy the determined size if it's less than available, else buy all available
            available = safe_get(available_queue, time_out, sym + ' available get timeout') # get the available amount of crypto from the queue, this also pauses any other thread trying to place an order (and therefore change the amount available
            if available is None:
                return
            if size * (limit_price * (1 + tkr_fee)) > available:
                size = available / (limit_price * (1 + tkr_fee))

            # Place the order and print the status
            print('Buying ' + sym + '\n' + 'price: ' + num2str(limit_price, wallet.product.usd_decimal_num) + '\n' + 'size: ' + num2str(
                size, 3) + '\n' + 'projected spread: ' + num2str(order['spread'], 6) + '\n')
            sleep(0.01)
            order_id = self.place_order(limit_price, 'buy', size, sym, post_only=False, stop_price=stop_price)
            if order_id is None:
                print(sym + ' buy Order rejected\n')
                sleep(0.01)
            else:
                print(sym + ' buy Order placed\n')
                sleep(0.01)

                self.add_order(order_id, sym, 'buy', time(), 0, spread=order['spread'], err=order['error'])

                self.place_holder_orders[sym]['buy'] = None
                com_wallet.update_value()
                available = com_wallet.get_amnt_available('buy')
                available_queue.put(available)
                print('available queue filled by ' + sym)
            return
        except Exception as e:
            print_err_msg(' place buy orders for ' + sym, e, 0)

    def continuous_sell(self, time_out):
        t0 = time()
        err_counter = 0
        while (time() - t0) < time_out:
            try:
                self.place_limit_sells()
                self.update_orders()
                err_counter = 0
            except Exception as e:
                err_counter = print_err_msg(' place sell orders', e, err_counter)

    def buy_place_holders(self, time_out=CYCLE_TIME):
        com_wallet = self.portfolio.get_common_wallet()
        com_wallet.update_value()
        available = com_wallet.get_amnt_available('buy')
        available_queue = Queue()
        available_queue.put(available)
        mkr_fee, tkr_fee = self.portfolio.get_fee_rate()

        place_holder_ids = []
        for sym in self.place_holder_orders.keys():
            if not self.place_holder_orders[sym]['buy'] is None:
                place_holder_ids.append(self.portfolio.wallets[sym].product.product_id)
        sell_thread = Thread(target=self.continuous_sell, args=(time_out,))
        sell_thread.start()
        if (available > QUOTE_ORDER_MIN) and (len(place_holder_ids) > 0):
            self.socket = Websocket(self.queque_dict, products=place_holder_ids)
            self.listen_for_price()
            sym_threads = []
            for sym in self.place_holder_orders.keys():
                current_sym_thread = Thread(target=self.buy_single_place_holder, args=(com_wallet, tkr_fee, available_queue, sym, time_out))
                current_sym_thread.start()
                sym_threads.append(current_sym_thread)

            for current_thread in sym_threads:
                current_thread.join()
            self.stop_listening()
        sell_thread.join()

    def determine_trade_price(self, sym, usd_available, side, mean_multiplier=1, max_predict_len=TRADE_LEN):
        _, _, coeff = self.get_side_depedent_vars(side)
        price_p, wallet, size_p, std, mu = self.determine_price_based_on_std(sym, usd_available, side, mu_mult=mean_multiplier, std_mult=1.0, max_predict_len=TRADE_LEN)
        return price_p, wallet, size_p, std, mu

    def place_order_for_top_currencies(self):
        # Ensure to update portfolio value before running
        self.update_orders()
        usd_hold = self.portfolio.get_usd_held()
        usd_available = self.portfolio.get_usd_available()
        desired_number_of_currencies = 5  # How many currecies (excluding USD) to hold
        sym_indices = list(
            range(0, desired_number_of_currencies))  # This chooses the indices to use for determining trades
        buy_prices, wallets, sizes, top_syms, stds, mus, spreads = self.rank_currencies(usd_available, print_sym=False,
                                                                                        sym_ind=sym_indices)
        no_viable_trade = False

        # Cancel bad buy orders before continuing
        self.cancel_old_orders()
        if top_syms is None:
            no_viable_trade = True
            top_syms = self.symbols  # If no viable trades are found then allow any symbol to remain
        if usd_hold > QUOTE_ORDER_MIN:
            private_pause()
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)

        # Check available cash after canceling the non_optimal buy orders and place the next order
        self.portfolio.update_value()
        full_portfolio_value = self.get_full_portfolio_value()
        num_orders = len(top_syms)
        num_currencies_to_loop = np.min(np.array([len(top_syms) + 1, desired_number_of_currencies + 1]))
        # Update fee rate
        mkr_fee, tkr_fee = self.portfolio.get_fee_rate()
        self.update_min_spread(mkr_fee, tkr_fee=tkr_fee)

        for ind in range(1, num_currencies_to_loop):
            i = num_orders - ind
            usd_available = self.portfolio.get_usd_available()
            top_sym = top_syms[i]
            current_placeholder = self.place_holder_orders[top_sym]['buy']
            # if self.errors[top_sym] > 0.05:
            #     continue
            # Only continue if there are viable trades
            if (usd_available > QUOTE_ORDER_MIN) and (not no_viable_trade):

                # Determine order size
                nominal_order_size = full_portfolio_value / desired_number_of_currencies
                if nominal_order_size < QUOTE_ORDER_MIN:
                    # never try to place an order smaller than the minimum
                    nominal_order_size = QUOTE_ORDER_MIN

                if nominal_order_size > (usd_available - QUOTE_ORDER_MIN):
                    # If placing the nominal order leaves an unusable amount of money then only use available
                    order_size = usd_available
                else:
                    order_size = nominal_order_size

                # Determine order properties
                buy_price, wallet, size, std, mu = self.determine_trade_price(top_sym, order_size, side='buy')
                sell_price, _, _, _, _ = self.determine_trade_price(top_sym, order_size, side='sell')
                dont_buy = self.is_possible_bubble(top_sym) #self.is_buy_order_fill_likely(buy_price, top_sym)
                if dont_buy:
                    continue
                elif (current_placeholder is not None) and (current_placeholder['price'] < buy_price):
                    # Don't have more than one placeholder order out at a time
                    continue

                # Always insure the order is small enough to go through in a reasonable amount of time
                mean_size, _, _ = wallet.product.get_mean_and_std_of_fill_sizes('asks', weighted=False)
                if order_size > (buy_price * mean_size):
                    order_size = buy_price * mean_size

                if buy_price is None:
                    continue

                existing_size = (wallet.value['SYM'])
                orders = self.portfolio.wallets[top_sym].product.get_open_orders()
                for order in orders:
                    if order['side'] == 'buy':
                        existing_size += float(order['size'])

                amnt_held = existing_size * buy_price

                # Scale size based on existing holdings
                if amnt_held <= (order_size - QUOTE_ORDER_MIN):
                    size -= existing_size
                # Don't place order if you already hold greater than or euqal to the nominal_order_size (within the QUOTE_ORDER_MIN)
                else:
                    continue

                # If the spread is too small don't place the order
                spread = calculate_spread(buy_price, sell_price)
                if spread < MIN_SPREAD:
                    print('Cannot place by order because projected sell price dropped\n')
                    continue

                # Adjust the size to account for the fee rate
                size = size / (1 + tkr_fee)

                # Place order
                self.place_order_for_nth_currency(buy_price, sell_price, wallet, size, std, mu, top_sym, err=self.errors[top_sym])
                self.portfolio.update_value()
            else:
                break

    def run(self, drop_box_key):
        global OPEN_ORDERS

        self.portfolio.update_value()
        print('Initializing portfolio tracking')
        portfolio_tracker = PortfolioTracker(self.portfolio, dbx_key=drop_box_key)
        portfolio_value = portfolio_tracker.initial_value
        print('SpreadBot starting value ' + num2str(portfolio_value, 2))

        last_check = 0
        last_update = 0
        last_plot = 0
        last_portfolio_refresh = datetime.now().timestamp()
        plot_period = 60
        check_period = 60
        propogator_update_period = CYCLE_TIME
        portfolio_refresh_period = 24 * 3600
        err_counter = 0

        while (MIN_PORTFOLIO_VALUE < portfolio_value) and (err_counter < 10):
            current_time = datetime.now().timestamp()
            #loop
            if (current_time > (last_check + check_period)):
                try:
                    private_pause()
                    PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                    # Update propogator frequencies
                    if (current_time > (last_update + propogator_update_period)):
                        self.predict()
                        last_update = datetime.now().timestamp()
                    # Trade
                    self.portfolio.update_value()
                    last_check = datetime.now().timestamp()
                    self.place_order_for_top_currencies()
                    self.buy_place_holders()
                    self.place_limit_sells()

                    # Update Settings
                    self.settings.update_settings()
                    new_offset = self.settings.settings['portfolio value offset']
                    self.portfolio.update_offset_value(new_offset)

                    current_time = datetime.now().timestamp()
                    if (current_time > (last_plot + plot_period)):
                        portfolio_value = portfolio_tracker.plot_returns()
                        last_plot = datetime.now().timestamp()

                    if (current_time > (portfolio_refresh_period + last_portfolio_refresh)):
                        portfolio_tracker.move_data_to_drop_box()
                        print('Portfolio Data moved to DropBox\n')
                        portfolio_tracker.reset()
                        last_portfolio_refresh = current_time

                    err_counter = 0

                except Exception as e:
                    if type(e) == JSONDecodeError:
                        print_err_msg(' get data', e, 0)
                        print('\n JSON error. taking a time out')
                        sleep(5 * 60)

                    else:
                        err_counter = print_err_msg(' complete main loop', e, err_counter)

        print('Loop END')

class PortfolioTracker:

    def __init__(self, portfolio, dbx_key=None):
        self.portfolio = portfolio
        percentage_data = {'Market': 100, 'Algorithm': 100}
        current_datetime = current_est_time()
        self.returns = pd.DataFrame(data=percentage_data, index=[current_datetime])
        self.initial_price = portfolio.wallets['BTC'].product.get_top_order('bids')
        self.initial_value = portfolio.get_full_portfolio_value()
        self.prediction_ticker = 'BTC'
        absolute_data = {'Portfolio Value:':self.initial_value}
        self.portfolio_value = pd.DataFrame(data=absolute_data, index=[current_datetime])
        self.dbx_key = dbx_key
        self.data_path = SAVED_DATA_FILE_PATH

    def reset(self):
        percentage_data = {'Market': 100, 'Algorithm': 100}
        current_datetime = current_est_time()
        self.returns = pd.DataFrame(data=percentage_data, index=[current_datetime])
        self.initial_price = self.portfolio.wallets['BTC'].product.get_top_order('bids')
        self.initial_value = self.portfolio.get_full_portfolio_value()
        self.prediction_ticker = 'BTC'
        absolute_data = {'Portfolio Value:': self.initial_value}
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

    def move_data_to_drop_box(self):
        global SAVED_DATA_FILE_PATH

        # Define the dropbox path
        dbx_path = r'/Roger Hobbies/crypto/Portfolio Returns' + SAVED_DATA_FILE_PATH[1::] # the 1:: is to remove the period at the beginning
        returns_csv_path = r'/returns.csv'
        return_png_path = r'/returns.png'
        value_csv_path = r'/value.csv'
        value_png_path = r'/value.png'
        files_to_move = (returns_csv_path, return_png_path, value_csv_path, value_png_path)

        # Move the current folders contents to dropbox
        for fpath in files_to_move:
            print(dbx_path + fpath)
            save_file_to_dropbox(SAVED_DATA_FILE_PATH + fpath, dbx_path + fpath, self.dbx_key)

        # Delete the current folder and create a new one
        files_in_directory = os.listdir(SAVED_DATA_FILE_PATH)
        for file in files_in_directory:
            os.remove(SAVED_DATA_FILE_PATH + r'/' + file)
        os.rmdir(SAVED_DATA_FILE_PATH)
        SAVED_DATA_FILE_PATH = portfolio_file_path_generator()

        if not os.path.exists(SAVED_DATA_FILE_PATH):
            os.mkdir(SAVED_DATA_FILE_PATH)

def run_bot(bot_type='psm'):
    global OPEN_ORDERS
    # -- Secret/changing variable declerations
    if len(sys.argv) > 2:
        # Definition from a shell file
        api_input = sys.argv[1]
        secret_input = sys.argv[2]
        passphrase_input = sys.argv[3]
        drop_box_key = sys.argv[4]
    else:
        api_input = input('What is the Coinbase api key? ')
        secret_input = input('What is the Coinbase secret key? ')
        passphrase_input = input('What is the Coinbase passphrase? ')
        drop_box_key = input('What is the DropBox Key? ')

    # Setup initial variables
    print('Initializing bot')
    if bot_type == 'psm':
        bot = PSMPredictBot(api_input, secret_input, passphrase_input)
    else:
        bot = SpreadBot(api_input, secret_input, passphrase_input)
    bot.portfolio.update_value()
    print('Initializing portfolio tracking')
    portfolio_tracker = PortfolioTracker(bot.portfolio, dbx_key=drop_box_key)
    portfolio_value = portfolio_tracker.initial_value
    print('SpreadBot starting value ' + num2str(portfolio_value, 2))

    sleep(1)
    last_check = 0
    last_predict = 0
    last_update = 0
    last_plot = 0
    last_portfolio_refresh = datetime.now().timestamp()
    plot_period = 60
    check_period = 60
    predict_period = 15*60
    propogator_update_period = TRADE_LEN*60
    portfolio_refresh_period = 24*3600
    err_counter = 0

    while (MIN_PORTFOLIO_VALUE < portfolio_value) and (err_counter < 10):
        current_time = datetime.now().timestamp()
        if (current_time > (last_check + check_period)):
            try:
                private_pause()
                PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
                # Update propogator frequencies
                if (current_time > (last_update + propogator_update_period)) and (bot_type == 'psm'):
                    bot.predict()
                    last_predict = datetime.now().timestamp()
                    last_update = datetime.now().timestamp()
                # Predict using psm
                elif (current_time > (last_predict + predict_period)) and (bot_type == 'psm'):
                    bot.predict(get_new_propogator=False)
                    last_predict = datetime.now().timestamp()
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

                if (current_time > (portfolio_refresh_period + last_portfolio_refresh)):
                    portfolio_tracker.move_data_to_drop_box()
                    print('Portfolio Data moved to DropBox\n')
                    portfolio_tracker.reset()
                    last_portfolio_refresh = current_time


                err_counter = 0

            except Exception as e:
                if type(e) == JSONDecodeError:
                    print_err_msg(' get data', e, 0)
                    print('\n JSON error. taking a time out')
                    sleep(5*60)

                else:
                    err_counter = print_err_msg(' find new data', e, err_counter)

    print('Loop END')

def run_predict_bot():
    global OPEN_ORDERS
    # -- Secret/changing variable declerations
    if len(sys.argv) > 2:
        # Definition from a shell file
        api_input = sys.argv[1]
        secret_input = sys.argv[2]
        passphrase_input = sys.argv[3]
        drop_box_key = sys.argv[4]
    else:
        api_input = input('What is the Coinbase api key? ')
        secret_input = input('What is the Coinbase secret key? ')
        passphrase_input = input('What is the Coinbase passphrase? ')
        drop_box_key = input('What is the DropBox Key? ')

    # Setup initial variables
    print('Initializing bot')
    bot = PredictBot(api_input, secret_input, passphrase_input)
    bot.run(drop_box_key)


if __name__ == "__main__":
    run_type = 'run_predict_bot'
    if run_type == 'run_predict_bot':
        run_predict_bot()
    if run_type == 'run':
        run_bot()
    elif run_type == 'other':
        api_input = input('What is the api key? ')
        secret_input = input('What is the secret key? ')
        passphrase_input = input('What is the passphrase? ')
        psmbot = PSMPredictBot(api_input, secret_input, passphrase_input)
        bot = SpreadBot(api_input, secret_input, passphrase_input)
        psmbot.predict(verbose_on=True)

        for sym in psmbot.symbols:
            plt.figure()
            prediction = psmbot.predictions[sym]
            reversed_prediction = psmbot.reversed_predictions[sym]
            raw_data = psmbot.raw_data[sym]
            err = psmbot.errors[sym]
            plt.plot(np.arange(0, len(raw_data)), raw_data)
            plt.plot(np.arange(len(raw_data), len(prediction) + len(raw_data)), prediction)
            plt.plot(np.arange(len(raw_data)-len(reversed_prediction), len(raw_data)), reversed_prediction)
            plt.title(sym)
            plt.xlabel('Time (min)')
            plt.ylabel('Price ($)')
            price_psm, _, _, psm_std, psm_mu = psmbot.determine_trade_price(sym, 10, 'buy')
            price, _, _, std, mu = bot.determine_trade_price(sym, 10, 'buy')
            print(sym)
            if price_psm is None:
                price_psm = 0
                psm_mu = 0
                psm_std = 0
            print('PSM price: ' + num2str(price_psm, 4) + ' , Naive price: '  + num2str(price, 4))
            print('PSM mu: ' + num2str(100*psm_mu, 4) + '% , Naive mu: ' + num2str(100*mu, 4) + '%')
            print('PSM std: ' + num2str(100*psm_std, 4) + '% , Naive std: ' + num2str(100*std, 4) + '%')
            print('Calculated Mean Error (percentage of real): ' + num2str(TRADE_LEN*100*err))
            print('--------\n')

        plt.show()

    elif run_type == 'new':
        new_sym = 'MKR'
        api_input = input('What is the api key? ')
        secret_input = input('What is the secret key? ')
        passphrase_input = input('What is the passphrase? ')
        new_prod = Wallet(api_input, secret_input, passphrase_input, sym=new_sym)
        new_prod.offset_value = 0
        t0 = None
        buy_price = None
        while True:
            # Get the price data
            new_prod.product.get_current_book()
            price_arr = np.array([float(order[0]) for order in new_prod.product.order_book['asks']])
            if len(price_arr) < 1:
                continue
            elif t0 is None:
                t0 = time()
                print(new_sym + ' Order Book Open')
            t = time() - t0
            # if the order book is open place orders at the top of the book
            price = np.min(price_arr[0] - new_prod.product.usd_res)
            new_prod.product.auth_client.cancel_all(new_sym + '-USD')
            avail_to_sell = 100#new_prod.get_amnt_available('buy')
            size = avail_to_sell / (price * (1.006))
            if avail_to_sell > 10:
                new_prod.product.place_order(price, 'buy', size, post_only=False)
                buy_price = price
                print('Placing ' + new_sym + ' buy order')
                break
            elif t > 60:
                break

        while True:
            avail_to_sell = new_prod.get_amnt_available('sell') / 1.006
            if avail_to_sell > new_prod.product.base_order_min:
                new_prod.product.place_order(buy_price * 2, 'sell', avail_to_sell, post_only=False)
                break