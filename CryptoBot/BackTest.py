import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras
from datetime import datetime
from datetime import timedelta
from time import sleep
import pandas as pd


class BackTestExchange():
    orders = {'bids':{}, 'asks':{}}

    def __init__(self, order_book_path):
        self.order_books = pd.read_csv(order_book_path)
        #self.model = keras.models.load_model(model_path)

    def get_top_order(self, ind, side):
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
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

        if coeff*price >= coeff*top_opposing_order: # This ensures the order is only placed onto the appropiate order book side
            this_side_orders = self.orders[side]
            if len(this_side_orders) > 0:
                existing_ids = np.array(list(this_side_orders.keys()))
                new_order_id = np.max(existing_ids) + 1
            else:
                new_order_id = 0
            this_side_orders[new_order_id] = {'size':size, 'price':price, 'filled':False}

    def update_fill_status(self, ind):
        sides = ['bids', 'asks']

        for side_ind in [0, 1]:
            # This loop cycles through the bids and asks books and fills any orders that are on the wrong book
            side = sides[side_ind]
            opposing_side = sides[not side_ind]
            coeff = (-1)**(side_ind)
            top_opposing_order = self.get_top_order(ind, opposing_side)

            for order in self.orders[side].values():
                if (not order['filled']) and (coeff*order['price'] >= coeff*top_opposing_order):
                    order['filled'] = True

    def remove_order(self, side, order_id):
        self.orders[side].pop(order_id)


class BackTestPortfolio():
    value = {'USD':100, 'BTC':0}

    def __init__(self, order_book_path):
        self.exchange = BackTestExchange(order_book_path)

    def place_bid(self):
        temp = True