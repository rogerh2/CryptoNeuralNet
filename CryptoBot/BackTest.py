import numpy as np
# import matplotlib.pyplot as plt
# import pickle
import keras
# from datetime import datetime
# from datetime import timedelta
# from time import sleep
import pandas as pd
import CryptoBot.CryptoForecast as cf

class BackTestExchange:
    orders = {'bids': {}, 'asks': {}}

    def __init__(self, order_book_path):
        self.order_books = pd.read_csv(order_book_path)
        # self.model = keras.models.load_model(model_path)

    def get_top_order(self, ind, side):
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
        else:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = self.order_books.iloc[[ind]][col].values[0]
        return top_order

    def get_current_book(self, ind):
        current_order_book_row = self.order_books.iloc[[ind]]
        current_order_book_row = current_order_book_row.drop(['ts'], axis=1)
        return current_order_book_row

    def place_order(self, price, side, size, ind):

        if side == 'asks':
            coeff = 1
            opposing_side = 'bids'
        elif side == 'bids':
            coeff = -1
            opposing_side = 'asks'
        else:
            raise ValueError(side + ' is not a valid orderbook side')
        top_opposing_order = self.get_top_order(ind, opposing_side)

        if coeff*price >= coeff*top_opposing_order:
            # This ensures the order is only placed onto the appropiate order book side
            this_side_orders = self.orders[side]
            if len(this_side_orders) > 0:
                existing_ids = np.array(list(this_side_orders.keys()))
                new_order_id = np.max(existing_ids) + 1
            else:
                new_order_id = 0
            this_side_orders[new_order_id] = {'size': size, 'price': price, 'filled': False}

    def update_fill_status(self, ind):
        sides = ['bids', 'asks']

        for side_ind in [0, 1]:
            # This loop cycles through the bids and asks books and fills any orders that are on the wrong book
            side = sides[side_ind]
            opposing_side = sides[not side_ind]
            coeff = (-1)**side_ind
            top_opposing_order = self.get_top_order(ind, opposing_side)

            for order in self.orders[side].values():
                if (not order['filled']) and (coeff*order['price'] >= coeff*top_opposing_order):
                    order['filled'] = True

    def remove_order(self, side, order_id):
        self.orders[side].pop(order_id)


class BackTestPortfolio:
    value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}

    def __init__(self, order_book_path):
        self.exchange = BackTestExchange(order_book_path)

    def update_value(self, fee=0):
        self.value['USD Hold'] = 0
        self.value['SYM Hold'] = 0
        for side in ['asks', 'bids']:
            ids_to_remove = []
            orders = self.exchange.orders[side]
            for id in orders.keys():
                order = orders[id]
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
                    ids_to_remove.append(id)
                    self.value[to_sym] += to_val * (1 - fee)
                else:
                    from_sym += ' Hold'
                    self.value[from_sym] += from_val

            for old_id in ids_to_remove:
                self.exchange.remove_order(side, old_id)

class BackTestBot:
    current_price = {'asks':None, 'bids':None}

    def __init__(self, model_path, strategy, historical_order_books_path, historical_fills_path):
        # strategy is a function that tells to bot to either buy or sell or hold, and at what price
        self.strategy = strategy
        self.model = cf.CryptoFillsModel('ETH', model_path=model_path)
        self.model.create_formatted_cbpro_data(historical_order_books_path, historical_fills_path)
        self.portfolio = BackTestPortfolio(historical_order_books_path)

    def get_order_book(self, ind):
        order_book_df = self.portfolio.exchange.get_current_book(ind)
        order_book = np.array([])
        for i in range(0, 120):
            order_book = np.append(order_book, order_book_df[str(int(i))].values[0])
        self.order_book = order_book

    def update_current_price(self, ind):
        for side in ['asks', 'bids']:
            top_order = self.portfolio.exchange.get_top_order(ind, 'asks')
            self.current_price[side] = top_order