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
import traceback

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

        if side == 'bids':
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
            side = 'bids'
            prices = bids
            opposing_prices = asks
        else:
            side = 'asks'
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
        else:
            bids = [float(x) for x in order_book['bids']]
            asks = [float(x) for x in order_book['asks']]
            num_order_book_entries = 20  # How far up the order book to scrape
            num_cols = 3 * 2 * num_order_book_entries

            bid_row = []
            ask_row = []

            for i in range(0, num_order_book_entries):
                bid_row = bid_row + bids[i]
                ask_row = ask_row + asks[i]

            new_row = [[ts] + bid_row + ask_row]
            header_names = ['ts'] + [str(x) for x in range(0, num_cols)]

            self.order_book = pd.DataFrame(data=new_row, columns=header_names)

    def get_top_order(self, side):
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
        else:
            raise ValueError('Side must be either "sell" or "buy"')
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
            if "id" in order_info.keys():
                new_order_id = order_info["id"]

        self.orders[side][new_order_id] = order_info

        return new_order_id

    def remove_order(self, id):
        self.auth_client.cancel_all(product_id=id)
        sleep(0.5)

class Portfolio:
    value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
    last_buy_price = None
    last_sell_price = None
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, api_key, secret_key, passphrase, prediction_ticker='ETH', is_sandbox_api=False):
        self.exchange = Exchange(api_key, secret_key, passphrase, prediction_ticker=prediction_ticker, is_sandbox_api=is_sandbox_api)
        self.ticker = prediction_ticker
        value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}

    def get_wallet_values(self, currency, data):
        # Data should come from self.auth_client.get_accounts()
        ind = [acc["currency"] == currency for acc in data]
        usd_wallet = data[ind.index(True)]
        balance = usd_wallet["balance"]
        hold_balance = usd_wallet["hold"]

        return balance, hold_balance

    def update_value(self):
        data = self.exchange.auth_client.get_accounts()
        sleep(0.5)
        usd_balance, usd_hold_balance = self.get_wallet_values('USD', data)
        sym_balance, sym_hold_balance = self.get_wallet_values(self.ticker, data)
        self.value['USD'] = usd_balance
        self.value['USD Hold'] = usd_hold_balance
        self.value['SYM'] = sym_balance
        self.value['SYM Hold'] = sym_hold_balance

    def get_amnt_available(self, side):
        if side == 'sell':
            sym = 'SYM'
        elif side == 'buy':
            sym = 'USD'
        else:
            raise ValueError('side must be either "sell" or "buy"')
        available = self.value[sym] - self.value[sym + ' Hold']
        return available

class LiveBaseBot:
    current_price = {'sell': None, 'buy': None}
    spread_price_limits = {'sell': None, 'buy': None}
    order_stds = {}
    fills = None
    prior_price = None
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
        for side in ['sell', 'bids']:
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
        self.model.data_obj.historical_order_books = self.order_books
        full_prediction = self.model.model_actions('forecast')
        bids = self.order_books['0'].values
        asks = self.order_books['60'].values
        int_len = 10
        if len(bids) > int_len:
            scaled_prediction = rescale_to_fit(full_prediction, bids)
            prediction = scaled_prediction
        else:
            prediction = np.array([0])

        if len(prediction) > 1:
            prediction_del = np.diff(prediction)
        else:
            prediction_del = np.zeros(prediction.shape)

        return prediction_del, order_book, bids, asks

    def get_full_portfolio_value(self):
        self.update_current_price()
        self.portfolio.update_value()
        price = np.mean([self.current_price['sell'], self.current_price['buy']])
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
            if self.order_stds[id] > order_std:
                keys_to_delete.append(id)

        for id in keys_to_delete:
            self.portfolio.exchange.remove_order(id)

    def update_last_price(self, last_price, spread, alt_price=0):
        if last_price is None:
            spread = alt_price
        else:
            spread = spread * last_price

        return spread

    def update_spread_prices_limits(self, spread=1.004):
        last_sell_price = self.portfolio.last_sell_price
        last_buy_price = self.portfolio.last_buy_price

        self.spread_price_limits['buy'] = self.update_last_price(last_sell_price, 1 / spread, alt_price=1000000)
        self.spread_price_limits['sell'] = self.update_last_price(last_buy_price, spread, alt_price=0)

    def trade_action(self):
        prediction, order_book, bids, asks = self.predict()
        decision, order_std = self.strategy.determine_move(prediction, order_book, self.portfolio, bids, asks) # returns None for hold
        self.update_spread_prices_limits()

        if (decision is not None):
            side = decision['side']
            price = decision['price']
            available = self.portfolio.get_amnt_available(side)
            if side == 'buy':
                size = available * decision['size coeff'] / decision['price']
                if price > self.spread_price_limits['buy']:
                    return self.prior_price
            else:
                size = available * decision['size coeff']
                if price < self.spread_price_limits['sell']:
                    return self.prior_price
            is_maker = decision['is maker']

            self.cancel_out_of_bound_orders(side, price, order_std)
            order_id = self.place_order(price, side, size, allow_taker=is_maker)
            self.order_stds[order_id] = order_std

            # -- this filters out prices for orders that were not placed --
            if order_id is None:
                price = self.prior_price
            self.prior_price = price
        else:
            price = self.prior_price

        return price


def print_err_msg(self, section_text, e, err_counter):
    sleep(5*60) #Most errors are connection related, so a short time out is warrented
    err_counter += 1
    print('failed to' + section_text + ' due to error: ' + str(e))
    print('number of consecutive errors: ' + str(err_counter))
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #print(exc_type, fname, exc_tb.tb_lineno)
    print(traceback.format_exc())

    return err_counter

def run_bot():
    if len(sys.argv) > 2:
        minute_path = sys.argv[1]
        api_input = sys.argv[2]
        secret_input = sys.argv[3]
        passphrase_input = sys.argv[4]
        sandbox_bool = bool(int(sys.argv[5]))

    else:
        minute_path = input('What is the model path? ')
        api_input = input('What is the api key? ')
        secret_input = input('What is the secret key? ')
        passphrase_input = input('What is the passphrase? ')
        sandbox_bool = bool(int(input('Is this for a sandbox? ')))

    print(minute_path)
    print(api_input)
    print(secret_input)
    print(passphrase_input)
    print(sandbox_bool)

    strategy = Strategy()
    bot = LiveBaseBot('saved_model.h5', strategy, api_input, secret_input, passphrase_input, is_sandbox_api=sandbox_bool)
    portfolio_value = bot.get_full_portfolio_value()
    starting_price = bot.current_price['buy']

    # This method keeps the bot running continuously
    current_time = datetime.now().timestamp()

    # This message is shown at the beginning
    print('Begin trading at ' + datetime.strftime(datetime.now(), '%m-%d-%Y %H:%M')
          + ' with current price of $' + str(
        starting_price) + ' per ' + bot.ticker + 'and a portfolio worth $' + num2str(portfolio_value, 2))
    sleep(1)
    err_counter = 0
    check_period = 1
    last_plot = 0
    last_buy_msg = ''
    last_sell_msg = ''
    fmt = '%Y-%m-%d %H:%M:'
    portfolio_returns = 0
    market_returns = 0

    # TODO make the below into a loop to trade with new software

    # while 11 < portfolio_value:
    #     if (current_time > (last_check + check_period)) & (current_time < (last_training_time + 2 * 3600)):
    #
    #         # Scrape price from cryptocompae
    #         try:
    #             order_dict = bot.auth_client.get_product_order_book(bot.product_id, level=2)
    #             sleep(0.4)
    #             accnt_data = bot.auth_client.get_accounts()
    #             sleep(0.4)
    #             bot.scrape_granular_price()
    #             last_check = current_time
    #             if (current_time > (last_scrape + 65)):
    #                 price, portfolio_value = bot.get_portfolio_value(order_dict, accnt_data)
    #                 bot.spread_bot_predict()
    #                 last_scrape = current_time
    #                 # self.order_status = 'active' #This forces the order to be reset as a stop order after 1 minute passes
    #                 portfolio_returns, market_returns = bot.update_returns_data(price, portfolio_value)
    #                 bot.timer['buy'] += 1
    #                 bot.timer['sell'] += 1
    #
    #             err_counter = 0
    #
    #
    #         except Exception as e:
    #             err_counter = bot.print_err_msg('find new data', e, err_counter)
    #             continue
    #
    #         # Plot returns
    #         try:
    #             if (current_time > (last_plot + 5 * 60)):
    #                 bot.plot_returns(portfolio_returns, portfolio_value, market_returns, price)
    #                 last_plot = current_time
    #                 err_counter = 0
    #         except Exception as e:
    #             err_counter = bot.print_err_msg('plot', e, err_counter)
    #             continue
    #
    #         # Make trades
    #         try:
    #             err, fit_coeff, fit_offset, const_diff, fuzziness = bot.find_fit_info()
    #             buy_msg = bot.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'buy', order_dict,
    #                                               accnt_data)
    #             sell_msg = bot.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'sell',
    #                                                order_dict, accnt_data)
    #             current_datetime = current_est_time()
    #             prez_fmt = '%Y-%m-%d %H:%M:%S'
    #             sell_msg = sell_msg.title()
    #             buy_msg = buy_msg.title()
    #
    #             if (buy_msg != last_buy_msg) and (buy_msg != ''):
    #                 print('\nCurrent time is ' + current_datetime.strftime(prez_fmt) + ' EST')
    #                 print('Buy message: ' + buy_msg)
    #                 last_buy_msg = buy_msg
    #
    #             if (sell_msg != last_sell_msg) and (sell_msg != ''):
    #                 print('\nCurrent time is ' + current_datetime.strftime(prez_fmt) + ' EST')
    #                 print('Sell message: ' + sell_msg)
    #                 last_sell_msg = sell_msg
    #
    #             err_counter = 0
    #
    #         except Exception as e:
    #             err_counter = bot.print_err_msg('trade', e, err_counter)
    #             continue
    #
    #
    #     # Update model training
    #     elif current_time > (last_training_time + 2 * 3600):
    #         try:
    #             last_scrape = 0
    #             last_training_time = current_time
    #             bot.price_model.model_actions('train', train_saved_model=True, save_model=False)
    #             bot.price_model.model.save(bot.save_str)
    #
    #             # Reinitialize CoinPriceModel
    #             bot.reinitialize_model()
    #             err_counter = 0
    #
    #         except Exception as e:
    #             err_counter = bot.print_err_msg('trade', e, err_counter)
    #             continue
    #
    #     current_time = datetime.now().timestamp()
    #     if err_counter > 12:
    #         print('Process aborted due to too many exceptions')
    #         break
    #
    # print(
    #     'Algorithm failed either due to underperformance or due to too many exceptions. Now converting all crypto to USD')
    # bot.auth_client.cancel_all(bot.product_id)
    # accnt_data = bot.auth_client.get_accounts()
    # sleep(0.4)
    # usd_available, crypto_available = bot.get_available_wallet_contents(accnt_data)
    # bot.auth_client.place_market_order(bot.product_id, side='sell', size=num2str(crypto_available, 8))