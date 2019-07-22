from CryptoBot.CryptoBot_Shared_Functions import num2str
import cbpro
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import islice
from time import sleep
from time import time
from CryptoBot.CryptoBot_Shared_Functions import num2str
import re

SETTINGS_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/spread_bot_settings.txt'


class Product:
    orders = {'buy': {}, 'sell': {}}
    order_book = None

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False, auth_client=None, pub_client=None):

        self.product_id = prediction_ticker.upper() + '-USD'

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
            return None
        else:
            self.order_book = order_book
            order_book['time'] = ts

    def get_top_order(self, side):
        _ = self.get_current_book()
        if not side in ['asks', 'bids']:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = float(self.order_book[side][0][0])
        return top_order

    def get_recent_fills(self, fill_number):
        recent_fills = list(islice(self.pub_client.get_product_trades(product_id=self.product_id), fill_number))
        sleep(0.5)
        return recent_fills

    def get_mean_and_std(self):
        fills = self.get_recent_fills(300)
        fill_arr = np.array([float(fill['price']) for fill in fills])
        fill_diff = np.diff(fill_arr)
        fill_diff_ratio = np.append(0, fill_diff) / fill_arr
        std = np.std(fill_diff_ratio)
        mu = np.mean(fill_diff_ratio)
        return mu, std

    def place_order(self, price, side, size, coeff=1, post_only=True):

        if not side in ['buy', 'sell']:
            raise ValueError(side + ' is not a valid orderbook side')

        new_order_id = None
        price_str = num2str(price, 2)
        size_str = num2str(coeff * size, 4)

        order_info = self.auth_client.place_limit_order(product_id=self.product_id, side=side, price=price_str, size=size_str, post_only=post_only)
        sleep(0.5)

        if type(order_info) == dict:
            if "price" in order_info.keys():
                new_order_id = order_info["id"]

        return new_order_id

class Wallet:
    value = {'USD': 15, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
    offset_value = None # Offset value is subtracted from the amount in usd to ensure only the desired amount of money is traded
    last_buy_price = None
    last_sell_price = None
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, api_key, secret_key, passphrase, sym='ETH', is_sandbox_api=False, auth_client=None, pub_client=None):
        self.exchange = Product(api_key, secret_key, passphrase, prediction_ticker=sym, is_sandbox_api=is_sandbox_api, auth_client=auth_client, pub_client=pub_client)
        self.ticker = sym

    def get_wallet_values(self, currency, data):
        # Data should come from self.auth_client.get_accounts()
        ind = [acc["currency"] == currency for acc in data]
        wallet = data[ind.index(True)]
        balance = wallet["balance"]
        hold_balance = wallet["hold"]

        return balance, hold_balance

    def update_value(self):
        data = self.exchange.auth_client.get_accounts()
        sleep(0.5)
        usd_balance, usd_hold_balance = self.get_wallet_values('USD', data)
        sym_balance, sym_hold_balance = self.get_wallet_values(self.ticker, data)
        usd_float_balance = float(usd_balance)
        if not self.offset_value:
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

    settings = {'portfolio value offset':None, 'limit buy':None, 'limit sell':None, 'spread':1.004, 'std':2}

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
                    setting_value = float(setting_str[0])

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
    def __init__(self, api_key, secret_key, passphrase, sym_list, is_sandbox=False):
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

        for symbol in sym_list:
            self.wallets[symbol] = Wallet(api_key, secret_key, passphrase, sym=symbol, auth_client=auth_client, pub_client=pub_client)
            self.auth = auth_client

    def get_full_portfolio_value(self):

        full_value = 0

        for sym in self.wallets.keys():
            wallet = self.wallets[sym]
            current_price = {'asks':None, 'bids':None}
            _ = wallet.exchange.get_current_book()

            for side in current_price.keys():
                top_order = wallet.exchange.get_top_order(side)
                current_price[side] = top_order

            wallet.update_value()
            price = np.mean([current_price['asks'], current_price['bids']])

            sym = wallet.value['SYM']
            full_value += sym*price

        usd = wallet.value['USD']
        full_value += usd

        return full_value

    def remove_order(self, id):
        self.auth.cancel_order(id)
        sleep(0.5)

class Bot:

    settings = LiveRunSettings(SETTINGS_FILE_PATH)
    spread = 1.004

    def __init__(self, api_key, secret_key, passphrase, syms=('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'EOS', 'XLM', 'ETC', 'LINK', 'REP', 'ZRX'), is_sandbox_api=False):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        self.portfolio = CombinedPortfolio(api_key, secret_key, passphrase, syms, is_sandbox=is_sandbox_api)
        self.symbols = syms
        self.current_price = {}
        self.spread_price_limits = {}
        for sym in syms:
            self.current_price[sym] = {'asks': None, 'bids': None}
            self.spread_price_limits[sym] = {'sell': 0, 'buy': 1000000}


    def update_current_price(self):
        for side in ['asks', 'bids']:
            for sym in self.symbols:
                top_order = self.portfolio.wallets[sym].exchange.get_top_order(side)
                self.current_price[sym][side] = top_order

    def place_order(self, price, side, size, sym, post_only=True):
        order_id = self.portfolio.wallets[sym].exchange.place_order(price, side, size, post_only=post_only)
        return order_id

    def get_full_portfolio_value(self):

        full_value = self.portfolio.get_full_portfolio_value()

        return full_value

    def cancel_out_of_bound_orders(self, side, price, sym):
        orders = list(self.portfolio.wallets[sym].exchange.auth_client.get_orders(self.portfolio.wallets[sym].exchange.product_id))
        sleep(0.5)
        keys_to_delete = []
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1

        for order in orders:
            if order['side'] != side:
                continue

            if (coeff * float(order['price']) < coeff * price):
                keys_to_delete.append(order['id'])

        for id in keys_to_delete:
            self.portfolio.remove_order(id)


def run_bot(symbols=('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'EOS', 'XLM', 'ETC', 'LINK', 'REP', 'ZRX')):
    api_input = input('What is the api key? ')
    secret_input = input('What is the secret key? ')
    passphrase_input = input('What is the passphrase? ')

    bot = Bot(api_input, secret_input, passphrase_input, syms=symbols)
    largest_var = -1

    for sym in symbols:
        mu, std = bot.portfolio.wallets[sym].exchange.get_mean_and_std()



if __name__ == "__main__":
    pass