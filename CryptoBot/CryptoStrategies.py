import numpy as np

class Strategy:

    def determine_move(self, prediction, order_book):

        if prediction < 0:
            decision = {'side': 'asks', 'size coeff': 1, 'price': (order_book['60'].values[0]+0.01)}
        elif prediction > 0:
            decision = {'side': 'bids', 'size coeff': 1, 'price': (order_book['0'].values[0]-0.01)}
        else:
            decision = None

        return decision