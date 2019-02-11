import numpy as np
import matplotlib.pyplot as plt
# import pickle
# import keras
# from datetime import datetime
# from datetime import timedelta
# from time import sleep
import pandas as pd
import CryptoBot.CryptoForecast as cf
from CryptoBot.CryptoStrategies import Strategy

class BackTestExchange:
    orders = {'bids': {}, 'asks': {}}
    time = 0

    def __init__(self, order_book_path):
        if order_book_path is not None:
            self.order_books = pd.read_csv(order_book_path)
        # self.model = keras.models.load_model(model_path)

    def get_top_order(self, side):
        ind = self.time
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
        else:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = self.order_books.iloc[[ind]][col].values[0]
        return top_order

    def get_current_book(self):
        ind = self.time
        current_order_book_row = self.order_books.iloc[[ind]]
        return current_order_book_row

    def place_order(self, price, side, size):

        if side == 'asks':
            coeff = 1
            opposing_side = 'bids'
        elif side == 'bids':
            coeff = -1
            opposing_side = 'asks'
        else:
            raise ValueError(side + ' is not a valid orderbook side')
        top_opposing_order = self.get_top_order(opposing_side)

        if coeff*price >= coeff*top_opposing_order:
            # This ensures the order is only placed onto the appropiate order book side
            this_side_orders = self.orders[side]
            if len(this_side_orders) > 0:
                existing_ids = np.array(list(this_side_orders.keys()))
                new_order_id = np.max(existing_ids) + 1
            else:
                new_order_id = 0
            this_side_orders[new_order_id] = {'size': size, 'price': price, 'filled': False}

    def update_fill_status(self):

        sides = ['bids', 'asks']

        for side_ind in [0, 1]:
            # This loop cycles through the bids and asks books and fills any orders that are on the wrong book
            side = sides[side_ind]
            opposing_side = sides[not side_ind]
            coeff = (-1)**side_ind
            top_opposing_order = self.get_top_order(opposing_side)

            for order in self.orders[side].values():
                if (not order['filled']) and (coeff*order['price'] >= coeff*top_opposing_order):
                    order['filled'] = True

    def remove_order(self, side, order_id):
        self.orders[side].pop(order_id)

class BackTestPortfolio:
    value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, order_book_path=None):
        self.exchange = BackTestExchange(order_book_path)

    def update_value(self, fee=0):
        self.value['USD Hold'] = 0
        self.value['SYM Hold'] = 0
        for side in ['asks', 'bids']:
            ids_to_remove = []
            orders = self.exchange.orders[side]
            for order_id in orders.keys():
                order = orders[order_id]
                if side == 'bids':
                    from_val = order['size'] * order['price']
                    to_val = order['size']
                    to_sym = 'SYM'
                    from_sym = 'USD'
                else:
                    to_val = order['size'] * order['price']
                    from_val = order['size']
                    to_sym = 'USD'
                    from_sym = 'SYM'


                if order['filled']:
                    self.value[from_sym] -= from_val
                    ids_to_remove.append(order_id)
                    self.value[to_sym] += to_val * (1 - fee)
                else:
                    from_sym += ' Hold'
                    self.value[from_sym] += from_val

            for old_id in ids_to_remove:
                self.exchange.remove_order(side, old_id)

    def get_amnt_available(self, side):
        if side == 'asks':
            sym = 'SYM'
        elif side == 'bids':
            sym = 'USD'
        else:
            raise ValueError('side must be either "asks" or "bids"')
        available = self.value[sym] - self.value[sym + ' Hold']
        return available

class BackTestBot:
    current_price = {'asks': None, 'bids': None}
    fills = None

    def __init__(self, model_path, strategy):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        self.strategy = strategy
        self.model = cf.CryptoFillsModel('ETH', model_path=model_path)
        self.model.create_formatted_cbpro_data()
        self.portfolio = BackTestPortfolio()

    def load_model_data(self, historical_order_books_path, historical_fills_path, train_test_split):
        # Load all data
        historical_order_books = pd.read_csv(historical_order_books_path)
        historical_fills = pd.read_csv(historical_fills_path)

        # Filter data so that only test data remains
        training_length = (int(len(historical_order_books) * (1 - train_test_split)))
        order_books = historical_order_books[training_length::]
        fills_ts_values = self.model.data_obj.str_list_to_timestamp(historical_fills.time.values)
        historical_fills_mask =  fills_ts_values > order_books.ts.values[0]
        self.fills = historical_fills[historical_fills_mask].values[0, ::]

        order_books = order_books.reset_index(drop=True)

        # Add filtered data to objects
        self.portfolio.exchange.order_books = order_books
        # self.model.data_obj.historical_order_books = order_books
        # self.model.data_obj.historical_fills = fills

    def get_order_book(self):
        order_book = self.portfolio.exchange.get_current_book()

        return order_book

    def update_current_price(self):
        for side in ['asks', 'bids']:
            top_order = self.portfolio.exchange.get_top_order('asks')
            self.current_price[side] = top_order

    def place_order(self, price, side, size):
        self.portfolio.exchange.place_order(price, side, size)

    def predict(self):
        self.model.data_obj.historical_order_books = self.get_order_book()
        prediction = self.model.model_actions('forecast')

        return prediction

    def get_full_portfolio_value(self):
        self.update_current_price()
        price = np.mean([self.current_price['asks'], self.current_price['bids']])
        usd = self.portfolio.value['USD']
        sym = self.portfolio.value['SYM']
        full_value = usd + sym*price

        return full_value

    def trade_action(self):
        order_book = self.get_order_book()
        prediction = self.predict()
        decision = self.strategy.determine_move(prediction, order_book) # returns None for hold
        side = decision['side']
        available = self.portfolio.get_amnt_available(side)
        size = available*decision['size coeff']
        if decision is not None:
            self.place_order(decision['price'], side, size)


def run_backtest(model_path, strategy, historical_order_books_path, historical_fills_path, train_test_split=0.33):

    bot = BackTestBot(model_path, strategy)
    bot.load_model_data(historical_order_books_path, historical_fills_path, train_test_split)
    times = np.array(bot.portfolio.exchange.order_books.index)
    portfolio_history = np.array([])  # This will track the bot progress

    for time in times:
        bot.portfolio.exchange.time = time
        bot.trade_action()
        val = bot.get_full_portfolio_value()
        portfolio_history = np.append(portfolio_history, val)

    return portfolio_history


if __name__ == "__main__":
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/BTC/BTCmodel_1layers_30fill_leakyreluact_adamopt_mean_absolute_percentage_errorloss_70neurons_4epochs1546857397.855116.h5'
    strategy = Strategy()
    historical_order_books_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/BTC_historical_order_books_20entries.csv'
    historical_fills_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/BTC_fills_20entries.csv'

    algorithm_returns = run_backtest(model_path, strategy, historical_order_books_path, historical_fills_path)

    plt.plot(algorithm_returns)