import unittest
import numpy as np
from CryptoBot.BackTest import BackTestPortfolio


class BackTestPortfolioTestCase(unittest.TestCase):

    def test_does_update_usd(self):
        self.test_obj = BackTestPortfolio(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/order_book_for_back_test_unit_tests.csv')
        self.test_obj.value = {'USD': 100, 'SYM': 1, 'USD Hold': 0, 'SYM Hold': 0}
        sizes = np.array([0.1, 0.15, 0.2, 0.1])
        prices = np.array([100, 110, 107, 106])
        self.test_obj.exchange.orders = {'bids':
                                             {0: {'size': sizes[0], 'price': prices[0], 'filled': True},
                                              1: {'size': sizes[1], 'price': prices[1], 'filled': False}},
                                         'asks':
                                             {0: {'size': sizes[2], 'price': prices[2], 'filled': False},
                                              1: {'size': sizes[3], 'price': prices[3], 'filled': False}}
                                         }
        self.test_obj.update_value()
        usd = self.test_obj.value['USD']

        self.assertEqual(usd, 100 - sizes[0]*prices[0], 'Incorrect Value')
        self.assertEqual(len(self.test_obj.exchange.orders['bids']), 1, 'Did not Remove Order')

    def test_does_update_sym(self):
        self.test_obj = BackTestPortfolio(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/order_book_for_back_test_unit_tests.csv')
        self.test_obj.value = {'USD': 100, 'SYM': 1, 'USD Hold': 0, 'SYM Hold': 0}
        sizes = np.array([0.1, 0.15, 0.2, 0.1])
        prices = np.array([100, 110, 107, 106])
        self.test_obj.exchange.orders = {'bids':
                                             {0: {'size': sizes[0], 'price': prices[0], 'filled': False},
                                              1: {'size': sizes[1], 'price': prices[1], 'filled': False}},
                                         'asks':
                                             {0: {'size': sizes[2], 'price': prices[2], 'filled': False},
                                              1: {'size': sizes[3], 'price': prices[3], 'filled': True}}
                                         }
        self.test_obj.update_value()
        sym = self.test_obj.value['SYM']

        self.assertEqual(sym, 1 - sizes[3])
        self.assertEqual(len(self.test_obj.exchange.orders['asks']), 1, 'Did not Remove Order')

    def test_does_update_usd_hold(self):
        self.test_obj = BackTestPortfolio(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/order_book_for_back_test_unit_tests.csv')
        self.test_obj.value = {'USD': 100, 'SYM': 1, 'USD Hold': 0, 'SYM Hold': 0}
        sizes = np.array([0.1, 0.15, 0.2, 0.1])
        prices = np.array([100, 110, 107, 106])
        self.test_obj.exchange.orders = {'bids':
                                             {0: {'size': sizes[0], 'price': prices[0], 'filled': True},
                                              1: {'size': sizes[1], 'price': prices[1], 'filled': False}},
                                         'asks':
                                             {0: {'size': sizes[2], 'price': prices[2], 'filled': False},
                                              1: {'size': sizes[3], 'price': prices[3], 'filled': True}}
                                         }
        self.test_obj.update_value()
        usd_hold = self.test_obj.value['USD Hold']

        self.assertEqual(usd_hold, sizes[1] * prices[1])

    def test_does_update_sym_hold(self):
        self.test_obj = BackTestPortfolio(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/order_book_for_back_test_unit_tests.csv')
        self.test_obj.value = {'USD': 100, 'SYM': 1, 'USD Hold': 0, 'SYM Hold': 0}
        sizes = np.array([0.1, 0.15, 0.2, 0.1])
        prices = np.array([100, 110, 107, 106])
        self.test_obj.exchange.orders = {'bids':
                                             {0: {'size': sizes[0], 'price': prices[0], 'filled': False},
                                              1: {'size': sizes[1], 'price': prices[1], 'filled': False}},
                                         'asks':
                                             {0: {'size': sizes[2], 'price': prices[2], 'filled': False},
                                              1: {'size': sizes[3], 'price': prices[3], 'filled': True}}
                                         }
        self.test_obj.update_value()
        sym_hold = self.test_obj.value['SYM Hold']

        self.assertEqual(sym_hold, sizes[2])


if __name__ == '__main__':
    unittest.main()
