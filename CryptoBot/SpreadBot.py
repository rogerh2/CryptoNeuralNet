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
import re

SETTINGS_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/spread_bot_settings.txt'
SAVED_DATA_FILE_PATH = r'/Users/rjh2nd/Dropbox (Personal)/crypto/Live Run Data/CryptoFillsBotReturns/Test' + str(current_est_time().date()).replace('-', '')

if not os.path.exists(SAVED_DATA_FILE_PATH):
    os.mkdir(SAVED_DATA_FILE_PATH)
else:
    override_saved_data = input('Override data in current saved data folder? (yes/no)' )
    if override_saved_data != 'yes':
        raise ValueError('Folder for saved plots already taken')


EXCHANGE_CONSTANTS = {'BTC':{'resolution':2, 'base order min':0.001},
                      'ETH':{'resolution':2, 'base order min':0.01},
                      'XRP':{'resolution':4, 'base order min':1},
                      'LTC':{'resolution':2, 'base order min':0.1},
                      'BCH':{'resolution':2, 'base order min':0.01},
                      'EOS':{'resolution':3, 'base order min':0.1},
                      'XLM':{'resolution':6, 'base order min':1},
                      'ETC':{'resolution':3, 'base order min':0.1},
                      'LINK':{'resolution':5, 'base order min':1},
                      'REP':{'resolution':2, 'base order min':0.1},
                      'ZRX':{'resolution':6, 'base order min':1}
                      }

QUOTE_ORDER_MIN = 10

class Product:
    orders = {'buy': {}, 'sell': {}}
    order_book = None

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False, auth_client=None, pub_client=None):

        self.product_id = prediction_ticker.upper() + '-USD'
        self.usd_decimal_num = EXCHANGE_CONSTANTS[prediction_ticker]['resolution']
        self.usd_res = 10**(-self.usd_decimal_num)
        self.quote_order_min = 10
        self.base_order_min = EXCHANGE_CONSTANTS[prediction_ticker]['base order min']

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

    def get_recent_fills(self, fill_number=1000):
        recent_fills = None
        for i in range(0, 10):
            recent_fills = list(islice(self.pub_client.get_product_trades(product_id=self.product_id), fill_number))
            sleep(0.5)
            if 'message' in recent_fills:
                sleep(1)
            else:
                break

        fill_mask = [recent_fills[i]['side']!=recent_fills[i-1]['side'] for i in range(0, len(recent_fills))]
        recent_fills = list(compress(recent_fills, fill_mask))

        return recent_fills

    def get_mean_and_std(self):
        fills = self.get_recent_fills()
        fill_arr = np.array([float(fill['price']) for fill in fills])
        fill_diff = np.diff(fill_arr)
        fill_diff_mask = np.abs(fill_diff) > self.usd_res #ignore small bounces between the minimum resolution
        fill_diff_ratio = np.append(0, fill_diff) / fill_arr
        fill_diff_ratio = fill_diff_ratio[1::][fill_diff_mask]
        std = np.std(fill_diff_ratio)
        mu = np.mean(fill_diff_ratio)

        # TODO get last sell and use for better price prediction
        last_fill = fills[-1]
        second_to_last_fill = fills[-2]

        return mu, std

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
        size_str = num2str(coeff * size, 4)

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

    def remove_order(self, id):
        self.auth.cancel_order(id)
        sleep(0.5)

    def update_value(self):
        wallet = self.get_common_wallet()
        # Get the data for all wallets to reduce api calls
        wallets_data = wallet.product.auth_client.get_accounts()

        for sym in self.symbols:
            self.wallets[sym].update_value(data=wallets_data) # TODO investigate why two wallet objects will affect each other


class Bot:

    settings = LiveRunSettings(SETTINGS_FILE_PATH)
    spread = 1.01

    def __init__(self, api_key, secret_key, passphrase, syms=('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'EOS', 'XLM', 'ETC', 'LINK', 'REP', 'ZRX'), is_sandbox_api=False):
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

    def cancel_out_of_bound_orders(self, side, price, sym):
        orders = list(self.portfolio.wallets[sym].product.auth_client.get_orders(self.portfolio.wallets[sym].product.product_id))
        sleep(0.5)
        keys_to_delete = []
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1

        for order in orders:
            if order['side'] != side:
                continue

            if coeff * float(order['price']) < coeff * (price + coeff * self.portfolio.wallets[sym].product.usd_res):
                keys_to_delete.append(order['id'])

        for id in keys_to_delete:
            self.portfolio.remove_order(id)

    def rank_currencies(self):
        # setup
        ranking_dict = {}

        # create dictionary for symbols and relevant data
        for sym in self.symbols:
            print('Evaluating ' + sym)
            mu, std = self.portfolio.wallets[sym].product.get_mean_and_std()
            ranking_dict[sym] = (mu, std)

        # sort (by mean first then standard deviation)
        sorted_syms = sorted(ranking_dict.items(), key=itemgetter(1), reverse=True)

        return sorted_syms

    def update_spread_prices_limits(self, last_price, side, sym):
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1
        self.spread_price_limits[sym][side] = last_price
        self.settings.write_setting_to_file('limit ' + side, self.spread_price_limits[sym][side])

    def place_order_for_top_currencies(self, order_ind=0):
        # determine whether enough crypto is available to order
        self.portfolio.update_value()
        usd_available = self.portfolio.get_usd_available()
        if usd_available > QUOTE_ORDER_MIN:
            # determine trade symbol
            sorted_syms = self.rank_currencies()
            top_sym = sorted_syms[order_ind][0]
            std = sorted_syms[order_ind][1][1]
            mu = sorted_syms[order_ind][1][0]
            std_coeff = self.settings.read_setting_from_file('std')
            print(top_sym + ' chosen as best trade with a std of ' + num2str(std, 4) + ' and a mean of ' + num2str(mu, 4) + '\n')

            # determine trade price
            order_coeff = 1 - (std_coeff * std) + mu # TODO make better determination for price (e.g. use an aggregated diff for the std)
            wallet = self.portfolio.wallets[top_sym]
            current_price = wallet.product.get_top_order('bids')
            buy_price = order_coeff * current_price
            size = usd_available / buy_price

            # place order and record
            print('placing order\n' + 'price: ' + num2str(buy_price, wallet.product.usd_decimal_num) + '\n' + 'size: ' + num2str(size, 3) + '\n')
            order_id = self.place_order(buy_price, 'buy', size, top_sym)
            if order_id is None:
                print('Order rejected\n')
            else:
                print('Order placed!\n')
                self.update_spread_prices_limits(buy_price, 'buy', top_sym)
                spread = 1 + (2 * std) + mu #self.settings.read_setting_from_file('spread')
                if spread < 1.004:
                    spread = 1.004
                self.settings.write_setting_to_file('spread', spread)
                self.update_spread_prices_limits(spread * buy_price, 'sell', top_sym)

    def place_limit_sells(self):
        self.portfolio.update_value()

        for sym in self.portfolio.wallets.keys():
            # Get relevant variables
            wallet = self.portfolio.wallets[sym]
            limit_price = self.spread_price_limits[sym]['sell']
            available = wallet.get_amnt_available('sell')

            # Filter unnecessary currencies
            # TODO fix so that sells go through, currently sets price to None and never resets if a buy is pending
            if available < wallet.product.base_order_min:
                continue
            print('Evaluating ' + sym + ' for spread sell')

            order_id = self.place_order(limit_price, 'sell', available, sym)
            if order_id is None:
                print('Order rejected\n')
            else:
                print('Order placed!\n')

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
    api_input = input('What is the api key? ')
    secret_input = input('What is the secret key? ')
    passphrase_input = input('What is the passphrase? ')

    # Setup initial variables
    bot = Bot(api_input, secret_input, passphrase_input)
    bot.portfolio.update_value()
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
                err_counter = 0
                last_check = datetime.now().timestamp()
                bot.place_order_for_top_currencies(0)
                bot.place_limit_sells()

                # Update Settings
                bot.settings.update_settings()
                new_offset = bot.settings.settings['portfolio value offset']
                bot.portfolio.update_offset_value(new_offset)

                if (current_time > (last_plot + plot_period)):
                    portfolio_value = portfolio_tracker.plot_returns()
                    last_plot = datetime.now().timestamp()

            except Exception as e:
                    err_counter = print_err_msg('find new data', e, err_counter)
                    continue


    print('Loop END')



if __name__ == "__main__":
    run_bot()