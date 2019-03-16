import numpy as np

class Strategy:

    def determine_move(self, prediction, order_book, portfolio):

        current_price = order_book['0'].values[0]
        decision = {'side': 'bids', 'size coeff': 1, 'price': 115.68}

        # if prediction < 0:
        #     current_price = order_book['60'].values[0]
        #     is_holding = portfolio.value['SYM'] * current_price > 10
        #     if is_holding:
        #         decision = {'side': 'bids', 'size coeff': 1, 'price': current_price + prediction}
        #     else:
        #         decision = {'side': 'asks', 'size coeff': 1, 'price': current_price}
        # elif prediction > 0:
        #     current_price = order_book['0'].values[0]
        #     is_holding = portfolio.value['USD'] > 10
        #     if is_holding:
        #         decision = {'side': 'asks', 'size coeff': 1, 'price': current_price + prediction}
        #     else:
        #         decision = {'side': 'bids', 'size coeff': 1, 'price': current_price}
        # else:
        #     decision = None

        return decision