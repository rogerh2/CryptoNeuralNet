import numpy as np

class Strategy:

    def determine_move(self, prediction, order_book):
        test_num = np.random.random()
        if test_num <= 0.33:
            decision = {'side': 'asks', 'size coeff': 1, 'price': order_book['120']}
        elif test_num <= 0.67:
            decision = {'side': 'asks', 'size coeff': 1, 'price': order_book['0']}
        else:
            decision = None

        return decision