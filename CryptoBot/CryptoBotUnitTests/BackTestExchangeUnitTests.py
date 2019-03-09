import unittest
from CryptoBot.BackTest import BackTestExchange

class BackTestExchangeTestCase(unittest.TestCase):
    test_obj = BackTestExchange(
        '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/order_book_for_back_test_unit_tests.csv')

    def test_can_return_top_order(self):
        self.test_obj.time = 0
        top_bid = self.test_obj.get_top_order('bids')
        top_ask = self.test_obj.get_top_order('asks')

        self.assertEqual(top_bid, 100, 'Incorrect bid')
        self.assertEqual(top_ask, 101, 'Incorrect ask')

    def test_returns_current_order_book(self):
        self.test_obj.time = 3
        test_book = self.test_obj.get_current_book()
        self.assertEqual(test_book['0'].values[0], 102)
        self.assertEqual(test_book['60'].values[0], 105)

    def test_places_bid_with_correct_price_and_size_and_removes_order(self):
        price = 100
        size = 1
        side = 'bids'
        self.test_obj.time = 3
        self.test_obj.place_order(price, 'bids', size)

        self.assertEqual(price, self.test_obj.orders[side][0]['price'], 'Incorrect price')
        self.assertEqual(size, self.test_obj.orders[side][0]['size'], 'Incorrect size')

        self.test_obj.remove_order(side, 0)
        self.assertEqual(len(self.test_obj.orders[side]), 0, 'Did not remove order')

    def test_places_ask_with_correct_price_and_size_and_removes_order(self):
        price = 105.1
        size = 1
        side = 'asks'
        self.test_obj.time = 3
        self.test_obj.place_order(price, side, size)

        self.assertEqual(price, self.test_obj.orders[side][0]['price'], 'Incorrect price')
        self.assertEqual(size, self.test_obj.orders[side][0]['size'], 'Incorrect size')

        self.test_obj.remove_order(side, 0)
        self.assertEqual(len(self.test_obj.orders[side]), 0, 'Did not remove order')

    def test_will_not_place_bid_on_ask_book(self):
        price = 110
        size = 1
        side = 'bids'
        self.test_obj.time = 3
        self.test_obj.place_order(price, 'bids', size)

        self.assertEqual(len(self.test_obj.orders[side]), 0, 'Placed bid on ask book')

    def test_will_not_place_ask_on_ask_book(self):
        price = 90
        size = 1
        side = 'asks'
        self.test_obj.time = 3
        self.test_obj.place_order(price, 'asks', size)

        self.assertEqual(len(self.test_obj.orders[side]), 0, 'Placed ask on bid book')

    def test_can_update_fill_status_for_bids(self):
        price = 102.5
        size = 1
        side = 'bids'
        self.test_obj.time = 1
        self.test_obj.place_order(price, side, size)
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Bid auto filled')
        self.test_obj.time = 2
        self.test_obj.update_fill_status()
        self.assertTrue(self.test_obj.orders[side][0]['filled'], 'Bid did not fill')

    def test_can_update_fill_status_for_asks(self):
        price = 103
        size = 1
        side = 'asks'
        self.test_obj.time = 3
        self.test_obj.place_order(price, side, size)
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Ask auto filled')
        self.test_obj.time = 4
        self.test_obj.update_fill_status()
        self.assertTrue(self.test_obj.orders[side][0]['filled'], 'Ask did not fill')

    def test_does_not_incorrectly_update_fill_status_for_asks(self):
        price = 120
        size = 1
        side = 'asks'
        self.test_obj.time = 3
        self.test_obj.place_order(price, side, size)
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Ask auto filled')
        self.test_obj.time = 4
        self.test_obj.update_fill_status()
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Ask auto filled on update')

    def test_does_not_incorrectly_update_fill_status_for_bids(self):
        price = 90
        size = 1
        side = 'bids'
        self.test_obj.time = 2
        self.test_obj.place_order(price, side, size)
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Bid auto filled')
        self.test_obj.time = 3
        self.test_obj.update_fill_status()
        self.assertFalse(self.test_obj.orders[side][0]['filled'], 'Bid auto filled on update')

    def tearDown(self):
        if len(self.test_obj.orders['asks']) > 0:
            self.test_obj.remove_order('asks', 0)

        if len(self.test_obj.orders['bids']) > 0:
            self.test_obj.remove_order('bids', 0)


if __name__ == '__main__':
    unittest.main()