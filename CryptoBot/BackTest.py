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

    def get_top_ask(self, ind):
        top_ask = self.order_books['60'].values[ind]
        return top_ask

    def get_top_bid(self, ind):
        top_ask = self.order_books['0'].values[ind]
        return top_ask

    def get_current_book(self, ind):
        current_order_book_row = self.order_books[ind]
        current_order_book_row = current_order_book_row.drop(['ts'], axis=1)
        return current_order_book_row

    def place_order(self, price, side, size, ind):
        if side == 'ask':
            top_order = self.get_top_ask(ind)
            coeff = 1
        elif side == 'bid':
            top_order = self.get_top_bid(ind)
            coeff = -1
        else:
            raise ValueError(side + ' is not a valid orderbook side')

        if coeff*price >= top_order: # This ensures the order is only placed onto the appropiate order book side
            this_side_orders = self.orders[side]
            existing_ids = np.array(list(this_side_orders.keys()))
            new_order_id = np.max(existing_ids) + 1
            this_side_orders[new_order_id] = {'size':size, 'price':price, 'settled':False}

    def remove_order(self, order_id):
        self.orders.pop(order_id)


class BackTestPortfolio():
    value=100

    def __init__(self, order_book_path):
        self.exchange = BackTestExchange(order_book_path)

