import matplotlib
matplotlib.use('Agg')
import sys

from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoBot.CryptoBot_Shared_Functions import rescale_to_fit
import CryptoBot.CryptoForecast as cf

# use the below for AWS
sys.path.append("home/ubuntu/CryptoNeuralNet")
# from CryptoForecast import CryptoPriceModel
# from CryptoBot_Shared_Functions import num2str
# from CryptoBot_Shared_Functions import create_number_from_bools

import cbpro
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from datetime import datetime
from time import sleep
from time import time
import pytz
import os
import re
import traceback

SETTINGS_FILE_PATH = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/fill_bot_settings'


class Strategy:

    def prediction_stat(self, predictions, is_plus=True):
        if len(predictions) > 1:
            if is_plus:
                prediction = np.mean(predictions)
            else:
                prediction = np.mean(predictions)
            order_std = np.std(predictions)
        elif len(predictions) == 1:
            prediction = predictions[0]
            order_std = 100
        else:
            prediction = 0
            order_std = 100

        return prediction, order_std

    def condition_prediction(self, side, predictions, prices):

        if side == 'buy':
            coeff = -1
        else:
            coeff = 1


        price_mask = np.abs( prices - prices[-1] ) < np.std(prices) * np.ones(prices.shape)

        if len(price_mask) == len(predictions):
            norm_predictions = coeff * predictions[price_mask]
        else:
            norm_predictions = coeff * predictions[price_mask[1::]]
        plus_predictions = norm_predictions[norm_predictions > 0.01]
        minus_predictions = norm_predictions[norm_predictions < -0.01]

        plus_prediction, plus_std = self.prediction_stat(plus_predictions)
        minus_prediction, minus_std = self.prediction_stat(minus_predictions, is_plus=False)

        return plus_prediction, minus_prediction, coeff, plus_std, minus_std

    def determine_move(self, predictions, order_book, portfolio, bids, asks):

        is_holding_usd = portfolio.value['USD'] > 10

        if is_holding_usd:
            side = 'buy'
            prices = bids
            opposing_prices = asks
        else:
            side = 'sell'
            prices = asks
            opposing_prices = bids

        current_price = prices[-1]
        current_opposing_price = opposing_prices[-1]
        prediction, opposing_prediction, coeff, plus_std, minus_std = self.condition_prediction(side, predictions, prices)
        plus_del = prediction # Opposing prediction is a negative quantity (the predictions are deltas)
        minus_del = opposing_prediction
        plus_price = current_price + coeff * ( prediction)
        minus_price = current_opposing_price - coeff * 0.01

        if (plus_std < minus_std):
            decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        # elif (plus_std > minus_std):
        #     decision = {'side': side, 'size coeff': 1, 'price': minus_price, 'is maker': False}
        else:
            decision = None

        # decision = {'side': side, 'size coeff': 1, 'price': plus_price, 'is maker': True}
        # decision = {'side': 'bids', 'size coeff': 1, 'price': 128}

        return decision, plus_std

class Exchange:
    orders = {'buy': {}, 'sell': {}}

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False):

        self.prediction_ticker = prediction_ticker.upper()
        self.product_id = prediction_ticker.upper() + '-USD'

        if is_sandbox_api:
            self.api_base = 'https://api-public.sandbox.pro.coinbase.com'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)
        else:
            self.api_base = 'https://api.pro.coinbase.com'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)

    def get_current_book(self):
        order_book = self.auth_client.get_product_order_book(self.product_id, level=2)
        sleep(0.5)
        ts = str(time())
        if not ('bids' in order_book.keys()):
            print('Get order book error, the returned dict is: ' + str(order_book))
            return None
        else:
            bids = order_book['bids']
            asks = order_book['asks']
            num_order_book_entries = 20  # How far up the order book to scrape
            num_cols = 3 * 2 * num_order_book_entries

            bid_row = []
            ask_row = []

            for i in range(0, num_order_book_entries):
                bid_row = bid_row + bids[i]
                ask_row = ask_row + asks[i]

            new_row = [ts] + bid_row + ask_row
            new_row_of_floats = [[float(x) for x in new_row]]
            header_names = ['ts'] + [str(x) for x in range(0, num_cols)]

            self.order_book = pd.DataFrame(data=new_row_of_floats, columns=header_names)
            return self.order_book

    def get_top_order(self, side):
        _ = self.get_current_book()
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
        else:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = self.order_book[col].values[0]
        return top_order

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

        self.orders[side][new_order_id] = order_info

        return new_order_id

    def remove_order(self, id):
        self.auth_client.cancel_order(id)
        self.orders.pop(id)
        sleep(0.5)

class Portfolio:
    value = {'USD': 15, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
    offset_value = None # Offset value is subtracted from the amount in usd to ensure only the desired amount of money is traded
    last_buy_price = None
    last_sell_price = None
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False):
        self.exchange = Exchange(api_key, secret_key, passphrase, prediction_ticker=prediction_ticker, is_sandbox_api=is_sandbox_api)
        self.ticker = prediction_ticker

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
            raise ValueError('side value set to' + side + ', side must be either "sell" or "buy"')
        available = float(self.value[sym]) - float(self.value[sym + ' Hold'])
        return available

class LiveRunSettings:

    settings = {'total value':None, 'limit buy':None, 'limit sell':None}

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

    def write_setting_to_file(self, setting_name, setting_val):
        self.settings[setting_name] = setting_val
        write_str = 'total value: ' + str(self.settings['total value']) + '\nlimit buy: ' + str(self.settings['limit buy']) + '\nlimit sell: ' + str(self.settings['limit sell'])
        with open(self.fname, 'w') as f:
            f.write(write_str)


    def update_settings(self):
        for setting_name in self.settings.keys():
            setting_value = self.read_setting_from_file(setting_name)
            self.settings[setting_name] = setting_value

class LiveBaseBot:
    current_price = {'asks': None, 'bids': None}
    spread_price_limits = {'sell': 0, 'buy': 1000000}
    settings = LiveRunSettings(SETTINGS_FILE_PATH)
    order_stds = {}
    fills = None
    order_books = None

    def __init__(self, model_path, strategy, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        self.strategy = strategy
        self.model = cf.CryptoFillsModel('ETH', model_path=model_path, suppress_output=True)
        self.model.create_formatted_cbpro_data()
        self.portfolio = Portfolio(api_key, secret_key, passphrase, prediction_ticker=prediction_ticker, is_sandbox_api=is_sandbox_api)
        self.ticker = prediction_ticker

    def get_order_book(self):
        order_book = self.portfolio.exchange.get_current_book()

        if self.order_books is None:
            self.order_books = order_book
        else:
            if len(self.order_books.index) >= 30:
                self.order_books = self.order_books.drop([0])
            self.order_books = self.order_books.append(order_book, ignore_index=True)

        return order_book

    def update_current_price(self):
        for side in ['asks', 'bids']:
            top_order = self.portfolio.exchange.get_top_order(side)
            self.current_price[side] = top_order

    def place_order(self, price, side, size, allow_taker=True):
        order_id = self.portfolio.exchange.place_order(price, side, size, post_only=allow_taker)
        return order_id

    def fit_to_data(self, predicted, true_price):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                coeff = np.polyfit(true_price, predicted, 1)
                fit_data = coeff[0] * predicted + coeff[1]
            except np.RankWarning:
                fit_data = predicted

        return fit_data

    def predict(self):
        order_book = self.get_order_book()
        temp_input_arr = self.order_books.drop(['ts'], axis=1).values
        arr = temp_input_arr.reshape(temp_input_arr.shape[0], temp_input_arr.shape[1], 1)
        full_prediction = self.model.model.predict(arr)
        bids = self.order_books['0'].values
        asks = self.order_books['60'].values
        int_len = 10
        if len(bids) > int_len:
            scaled_prediction = rescale_to_fit(full_prediction, bids)
            prediction = scaled_prediction[::, 0]
        else:
            prediction = np.zeros(full_prediction.shape)

        if len(prediction) > 10:
            prediction_del = np.diff(prediction)
        else:
            prediction_del = np.zeros(prediction.shape)

        return prediction_del, order_book, bids, asks

    def get_full_portfolio_value(self):
        self.update_current_price()
        self.portfolio.update_value()
        price = np.mean([self.current_price['asks'], self.current_price['bids']])
        usd = self.portfolio.value['USD']
        sym = self.portfolio.value['SYM']
        full_value = usd + sym*price

        return full_value

    def cancel_out_of_bound_orders(self, side, price, order_std):
        # TODO take into account order reason (e.g. placing order at current price to hit future prediction vs placing order at projected future price)
        orders = self.portfolio.exchange.orders[side]
        keys_to_delete = []
        if side == 'buy':
            coeff = -1
        else:
            coeff = 1

        for id in orders.keys():
            # if coeff * orders[id]['price'] > coeff * price:
            if (self.order_stds[id] > order_std) and (coeff * float(orders[id]["price"]) > coeff * price):
                keys_to_delete.append(id)

        for id in keys_to_delete:
            self.portfolio.exchange.remove_order(id)

    def update_last_price(self, last_price, spread, alt_price=0):
        if last_price is None:
            spread = alt_price
        else:
            spread = spread * last_price

        return spread

    def update_spread_prices_limits(self, last_price, side, spread=1.004):
        if side == 'sell':
            self.spread_price_limits['buy'] = self.update_last_price(last_price, 1 / spread, alt_price=1000000)
            self.settings.write_setting_to_file('limit buy', self.spread_price_limits['buy'])
        elif side == 'buy':
            self.spread_price_limits['sell'] = self.update_last_price(last_price, spread)
            self.settings.write_setting_to_file('limit sell', self.spread_price_limits['sell'])

    def trade_action(self):
        prediction, order_book, bids, asks = self.predict()
        decision, order_std = self.strategy.determine_move(prediction, order_book, self.portfolio, bids, asks) # returns None for hold

        if (decision is not None):
            side = decision['side']
            price = decision['price']
            self.cancel_out_of_bound_orders(side, price, order_std)
            available = self.portfolio.get_amnt_available(side)
            if side == 'buy':
                size = available * decision['size coeff'] / decision['price']
                if price > self.spread_price_limits['buy']:
                    return None
            else:
                size = available * decision['size coeff']
                if price < self.spread_price_limits['sell']:
                    return None

            print('Evaluating ' + side + ' of ' + num2str(size, 3) + ' ' + self.ticker + ' at $' + num2str(price, 2) + ' based on std of ' + num2str(order_std, 4))
            is_maker = decision['is maker']

            order_id = self.place_order(price, side, size, allow_taker=is_maker)
            self.order_stds[order_id] = order_std

            # -- this filters out prices for orders that were not placed --
            if order_id is None:
                price = None
            else:
                self.update_spread_prices_limits(price, side)
        else:
            price = None
            side = None
            size = None

        return price, side, size

def print_err_msg(section_text, e, err_counter):
    sleep(5*60) #Most errors are connection related, so a short time out is warrented
    err_counter += 1
    print('failed to' + section_text + ' due to error: ' + str(e))
    print('number of consecutive errors: ' + str(err_counter))
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(traceback.format_exc())

    return err_counter

def run_bot():
    # -- Secret/changing variable declerations
    if len(sys.argv) > 2:
        # Definition from a shell file
        model_path = sys.argv[1]
        api_input = sys.argv[2]
        secret_input = sys.argv[3]
        passphrase_input = sys.argv[4]
        sandbox_bool = bool(int(sys.argv[5]))

    else:
        # Manual definition
        model_path = input('What is the model path? ')
        api_input = input('What is the api key? ')
        secret_input = input('What is the secret key? ')
        passphrase_input = input('What is the passphrase? ')
        sandbox_bool = bool(int(input('Is this for a sandbox? ')))

    print(model_path)
    print(api_input)
    print(secret_input)
    print(passphrase_input)
    print(sandbox_bool)

    settings = LiveRunSettings('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/fill_bot_settings')

    strategy = Strategy()
    bot = LiveBaseBot(model_path, strategy, api_input, secret_input, passphrase_input, is_sandbox_api=sandbox_bool)
    portfolio_value = bot.get_full_portfolio_value()
    starting_price = bot.current_price['bids']

    # This method keeps the bot running continuously
    current_time = datetime.now().timestamp()

    # This message is shown at the beginning
    print('Begin trading at ' + datetime.strftime(datetime.now(), '%m-%d-%Y %H:%M')
          + ' with current price of $' + str(
        starting_price) + ' per ' + bot.ticker + 'and a portfolio worth $' + num2str(portfolio_value, 2))
    sleep(1)
    last_check = 0
    check_period = 1
    err_counter = 0

    while 11 < portfolio_value:
        current_time = datetime.now().timestamp()
        if (current_time > (last_check + check_period)):
            try:
                price, side, size = bot.trade_action()
                if price:
                    print('Placed ' + side + ' order for ' + num2str(size, 3) + ' ' + bot.ticker + ' at $' + num2str(price,2))
                err_counter = 0
                last_check = datetime.now().timestamp()
            except Exception as e:
                    err_counter = print_err_msg('find new data', e, err_counter)
                    continue
            finally:
                portfolio_value = bot.get_full_portfolio_value()
                bot.settings.update_settings()
                bot.spread_price_limits['buy'] = bot.settings.settings['limit buy']
                bot.spread_price_limits['sell'] = bot.settings.settings['limit sell']

    print('Loop END')


if __name__ == '__main__':
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/ETH/ETHmodel_3layers_30fill_leakyreluact_adamopt_mean_absolute_percentage_errorloss_60neurons_14epochs1553129847.019871.h5'
    run_bot()