import matplotlib
matplotlib.use('Agg')
from queue import Queue
import cbpro
import numpy as np
import warnings
from itertools import islice
from itertools import compress
from time import sleep
from time import time
from CryptoBot.CryptoBot_Shared_Functions import num2str, str_list_to_timestamp, private_pause, public_pause, print_err_msg
from typing import Dict
from CryptoBot.constants import EXCHANGE_CONSTANTS, QUOTE_ORDER_MIN, PUBLIC_SLEEP, PRIVATE_SLEEP, TRADE_LEN, PRIVATE_SLEEP_QUEUE, PUBLIC_SLEEP_QUEUE, OPEN_ORDERS



# These classes collect raw data

class Websocket(cbpro.WebsocketClient):
    def __init__(self, queque_dict: Dict[str, Queue], products=None, message_type="subscribe", mongo_collection=None,
                 should_print=True, auth=False, api_key="", api_secret="", api_passphrase="", channels=('ticker',)):
        super().__init__(products=products, message_type=message_type, mongo_collection=mongo_collection,
                 should_print=should_print, auth=auth, api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase, channels=channels)
        self.queques = queque_dict # This dictionary has a queque for each product id

    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.message_count = 0
        print("Websocket Initiated")
    def on_message(self, msg): # TODO implement error handling
        if 'product_id' in msg.keys():
            self.queques[msg['product_id'][0:-4]].put_nowait(msg)
    def on_close(self):
        print("Websocket Closed")

# These classes interface with the exchange
class Product:

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False, auth_client=None, pub_client=None, base_ticker='USD'):

        self.product_id = prediction_ticker.upper() + '-' + base_ticker
        if not (base_ticker == 'USD'):
            prediction_ticker = self.product_id
        self.orders = {'buy': {}, 'sell': {}}
        self.usd_decimal_num = EXCHANGE_CONSTANTS[prediction_ticker]['resolution']
        self.usd_res = 10**(-self.usd_decimal_num)
        self.quote_order_min = QUOTE_ORDER_MIN
        self.base_order_min = EXCHANGE_CONSTANTS[prediction_ticker]['base order min']
        self.base_decimal_num = EXCHANGE_CONSTANTS[prediction_ticker]['base resolution']
        self.crypto_res = 10**(-self.base_decimal_num)
        self.filt_fills = None
        self.all_fills = None
        self.order_book = None
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
        public_pause()
        order_book = self.pub_client.get_product_order_book(self.product_id, level=2)
        PUBLIC_SLEEP_QUEUE.put(time() + PUBLIC_SLEEP)
        ts = time()
        if not ('bids' in order_book.keys()):
            print('Get order book error, the returned dict is: ' + str(order_book))
        else:
            self.order_book = order_book
            order_book['time'] = ts

    def detect_whale_price(self):
        self.get_current_book()
        ask_size_arr = np.array([])
        bid_size_arr = np.array([float(order[1]) for order in self.order_book['bids']])
        price_arr = np.array([])
        for order in self.order_book['asks']:
            price = float(order[0])
            size = float(order[1])
            if price*size < 15:
                continue
            ask_size_arr = np.append(ask_size_arr, size)
            price_arr = np.append(ask_size_arr, price)

        max_size = np.mean(ask_size_arr) + 2.5 * np.std(ask_size_arr)

        bid_sum = np.sum(bid_size_arr)
        ask_sum = np.sum(ask_size_arr)
        if bid_sum > 1.5*ask_sum:
            max_size = np.mean(ask_size_arr) + 3.5 * np.std(ask_size_arr)
        elif bid_sum > ask_sum:
            max_size = np.mean(ask_size_arr) + 3 * np.std(ask_size_arr)

        for i in range(0, len(ask_size_arr)):
            size = ask_size_arr[i]
            if i > 0:
                price = price_arr[i-1]
            else:
                price = price_arr[0]
            if size > max_size:
                return price
        return price_arr[-1]


    def get_top_order(self, side, refresh=True):
        if refresh:
            self.get_current_book()
        if not side in ['asks', 'bids']:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = float(self.order_book[side][0][0])
        return top_order

    def get_recent_fills(self, fill_number=1000, return_time=False):
        if (time() - self.fill_time) > 7:
            # If fills have not been scraped recently (last 3s) then scrape
            recent_fills = None
            for i in range(0, 10):
                public_pause()
                recent_fills = list(islice(self.pub_client.get_product_trades(product_id=self.product_id), fill_number))
                PUBLIC_SLEEP_QUEUE.put(time() + PUBLIC_SLEEP)
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

        if filtered_recent_fills is None:
            return None, None

        if return_time:
            return filtered_recent_fills, filt_fill_ts_ls_r
        else:
            return filtered_recent_fills, recent_fills

    def get_recent_fill_prices(self, fill_num=1000, return_t=False):
        fills, _ = self.get_recent_fills(fill_number=fill_num, return_time=return_t)
        fill_prices = np.array([float(fill['price']) for fill in fills])

        return fill_prices

    def get_num_price_momentum_switches_per_time(self, t_interval_in_seconds=TRADE_LEN * 60):
        fills, fill_ts_ls = self.get_recent_fills(return_time=True)
        if fills is None:
            return None

        num_trades_per_t = []
        num_trades = 0
        if len(fill_ts_ls) > 1: # greater than 1 because a single trade is not enough to make an average
            ts0 = fill_ts_ls[0]
        else:
            return None
        t = 0

        for ts in fill_ts_ls:
            t = ts - ts0
            if t >= t_interval_in_seconds:
                num_trades_per_t.append(num_trades)
                ts0 = ts
                num_trades = 0
            else:
                num_trades += 1

        if (t > (0.1 * t_interval_in_seconds)) and (num_trades > 0):
            num_trades_per_t.append((t_interval_in_seconds / t) * num_trades)

        avg_num_trades = np.mean(np.array(num_trades_per_t))
        if (avg_num_trades is None) or np.isnan(avg_num_trades):
            avg_num_trades = (t_interval_in_seconds / t) * num_trades

        return avg_num_trades

    def adjust_fill_data(self, mu, std, fill_diff_ratio):
        # This adjusts the mean and std of the price changes based on a number of trades to represent a particular
        # amount of time
        avg_num_trades = self.get_num_price_momentum_switches_per_time()
        if (avg_num_trades is None) or (np.isinf(avg_num_trades)):
            return None, None, None
        weighted_mu = avg_num_trades * mu
        weighted_std = np.sqrt(avg_num_trades) * std
        if avg_num_trades < 1:
            last_fill = fill_diff_ratio[-1]
        else:
            last_fill = np.sum(fill_diff_ratio[-int(avg_num_trades)::])

        return weighted_mu, weighted_std, last_fill

    def normalize_price_changes(self, fill_arr):
        # This method returns the normalized price changes
        fill_diff = np.diff(fill_arr)
        fill_diff_mask = np.abs(fill_diff) > self.usd_res  # ignore small bounces between the minimum resolution
        fill_diff_ratio = np.append(0, fill_diff) / fill_arr
        fill_diff_ratio = fill_diff_ratio[1::][fill_diff_mask]

        return fill_diff_ratio

    def get_mean_and_std_of_price_changes(self):
        fills, _ = self.get_recent_fills()
        if fills is None:
            return None, None, None
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

    def get_psm_mean_and_std_of_price_changes(self, predicted_fills, fill_arr):
        # fills, _ = self.get_recent_fills()
        # if fills is None:
        #     return None, None, None
        # # Get the standard deviation based on past price movements
        # fill_arr = np.array([float(fill['price']) for fill in fills])
        fill_diff_ratio = self.normalize_price_changes(fill_arr)
        std = np.std(fill_diff_ratio)

        # Get the mean based on the predicted price movement
        t = np.linspace(0, TRADE_LEN, len(predicted_fills))
        coeff = np.polyfit(t, predicted_fills, 1)
        weighted_mu = coeff[0] / np.mean(fill_arr) # This does not need to be wheighted because it is already based off time
        weighted_std = np.std(predicted_fills - np.polyval(coeff, np.arange(0, len(predicted_fills)))) / np.mean(fill_arr)

        # Adjust mu and std to account for number of trades over time
        _, _, last_fill = self.adjust_fill_data(weighted_mu, std, fill_diff_ratio)

        return weighted_mu, weighted_std, last_fill

    def get_mean_and_std_of_fill_sizes(self, side, weighted=True):
        _, fills = self.get_recent_fills()
        if fills is None:
            return None, None, None
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
        if weighted:
            weighted_mu, weighted_std, last_fill = self.adjust_fill_data(mu, std, size_arr)
        else:
            _, _, last_fill = self.adjust_fill_data(mu, std, size_arr)
            weighted_mu = mu
            weighted_std = std


        return weighted_mu, weighted_std, last_fill

    def place_order(self, price, side, size, coeff=1, post_only=True, time_out=False, stop_price=None):

        if not side in ['buy', 'sell']:
            raise ValueError(side + ' is not a valid orderbook side')
        if price * size < self.quote_order_min:
            print(num2str(price * size, self.usd_decimal_num) + ' is smaller than the minimum quote size')
            return None
        if size < self.base_order_min:
            print(num2str(size, 6) + ' is smaller than the minimum base size')
            return None

        new_order_id = None
        # Some orders are not placing due to order size, so the extra subtraction below is to ensure they are small enough
        price_str = num2str(price, self.usd_decimal_num)
        size_str = num2str(coeff * size, self.base_decimal_num)
        if stop_price:
            stop_str = num2str(stop_price, self.usd_decimal_num)

        if side == 'buy':
            stop_type = 'entry'
        else:
            stop_type = 'loss'

        private_pause()
        if time_out and stop_price:
            # TODO fix stop order
            order_info = self.auth_client.place_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only, time_in_force='GTT', cancel_after='hour', stop=stop_type, order_type='limit', stop_price=stop_str)
        elif stop_price:
            order_info = self.auth_client.place_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only, stop=stop_type, order_type='limit', stop_price=stop_str)
        elif time_out:
            order_info = self.auth_client.place_limit_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only, time_in_force='GTT', cancel_after='hour')
        else:
            order_info = self.auth_client.place_limit_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only)
        PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)

        if type(order_info) == dict:
            if "price" in order_info.keys():
                new_order_id = order_info["id"]
        if new_order_id is None:
            print(order_info)
        else:
            self.orders[side][new_order_id] = order_info

        return new_order_id

    def get_open_orders(self):
        if OPEN_ORDERS is None:
            private_pause()
            orders = list(self.auth_client.get_orders(self.product_id))
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        else:
            orders = []
            for order in OPEN_ORDERS:
                if self.product_id == order['product_id']:orders.append(order)
        return orders

    def update_orders(self):
        # Update still open orders
        open_orders = self.get_open_orders()
        open_ids = []
        for order in open_orders:
            if type(order) != dict:
                continue
            id = order['id']
            open_ids.append(id)
            # Check to ensure the order was placed by this bot
            if (not id in self.orders['buy'].keys()) or (not id in self.orders['sell'].keys()):
                side = order['side']
                self.orders[side][id] = order

        # Remove completed orders
        for side in ('buy', 'sell'):
            id_list = list(self.orders[side].keys())
            for id in id_list:
                if id not in open_ids: del self.orders[side][id]

class Wallet:
    offset_value = None # Offset value is subtracted from the amount in usd to ensure only the desired amount of money is traded
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, api_key, secret_key, passphrase, sym='ETH', base='USD', is_sandbox_api=False, auth_client=None, pub_client=None):
        self.product = Product(api_key, secret_key, passphrase, prediction_ticker=sym, base_ticker=base, is_sandbox_api=is_sandbox_api, auth_client=auth_client, pub_client=pub_client)
        self.ticker = sym
        self.value = {'USD': 15, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
        self.base = base
        self.last_buy_price = None
        self.last_sell_price = None

    def get_wallet_values(self, currency, data):
        # Data should come from self.auth_client.get_accounts()
        ind = []
        for acc in data:
            if type(acc) is str:
                print('WARNING: string found in account data: ' + acc)
                continue
            ind.append(acc["currency"] == currency)
        wallet = data[ind.index(True)]
        balance = wallet["balance"]
        hold_balance = wallet["hold"]

        return balance, hold_balance

    def update_value(self, data=None):
        if data is None:
            data = self.product.auth_client.get_accounts()
        sleep(PRIVATE_SLEEP)
        usd_balance, usd_hold_balance = self.get_wallet_values(self.base, data)
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

class CombinedPortfolio:

    def __init__(self, api_key, secret_key, passphrase, sym_list, base_currency='USD', offset_value=70, is_sandbox=False):
        self.wallets = {}

        if is_sandbox:
            api_base = 'https://api-public.sandbox.pro.coinbase.com'
            private_pause()
            auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=api_base)
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
            pub_client = cbpro.PublicClient(api_url=api_base)
        else:
            private_pause()
            auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase)
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
            pub_client = cbpro.PublicClient()

        self.auth = auth_client

        for product_id in sym_list:
            # The base can be put in the symbols in the format Quote-Base for use with multiple currencies
            if (type(base_currency) is list) or (type(base_currency) is tuple):
                product_dat = product_id.split('-')
                symbol = product_dat[0]
                base = product_dat[1]
                if base not in base_currency:
                    continue
            else:
                symbol = product_id
                base = base_currency
            self.wallets[product_id] = Wallet(api_key, secret_key, passphrase, base=base, sym=symbol, auth_client=auth_client, pub_client=pub_client)
            self.wallets[product_id].offset_value = offset_value

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
            if ('-' in sym) and ('USD' not in sym):
                continue
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
        private_pause()
        self.auth.cancel_order(id)
        PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)

    def update_value(self):
        wallet = self.get_common_wallet()
        # Get the data for all wallets to reduce api calls
        wallets_data = wallet.product.auth_client.get_accounts()

        for sym in self.symbols:
            self.wallets[sym].update_value(data=wallets_data)

    def get_fee_rate(self):
        private_pause()
        fee_rates = self.auth._send_message('get', '/fees')
        PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        maker_rate = float(fee_rates['maker_fee_rate'])
        taker_rate = float(fee_rates['taker_fee_rate'])
        return maker_rate, taker_rate

    def get_all_open_orders(self):
        orders = None
        if OPEN_ORDERS is None:
            private_pause()
            try:
                orders = list(self.auth.get_orders())
            except Exception as e:
                print_err_msg('Failed to get open orders', e, 0)
            PRIVATE_SLEEP_QUEUE.put(time() + PRIVATE_SLEEP)
        else:
            orders = OPEN_ORDERS
        return orders