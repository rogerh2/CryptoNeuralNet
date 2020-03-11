import unittest
import pandas as pd
import numpy as np
from CryptoBot.SpreadBot import PSMPredictBot
from cbpro import AuthenticatedClient
from time import time

API_KEY = input('Enter the API Key:')
SECRET_KEY = input('Enter the Secret Key:')
PASSPHRASE = input('Enter the Passphrase:')
CSV_PATH = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/test_orders.csv'

class PSMPredictBotOrderStorage(unittest.TestCase):
    ids_to_cancel = []
    api_key=API_KEY
    secret_key=SECRET_KEY
    passphrase=PASSPHRASE

    def test_bot_creates_csv_of_orders_and_loads_at_startup(self):
        # Initialize starting variables
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH, is_sandbox_api=True)
        buy_price = 1000
        sell_price = 10**6
        sym = 'BTC'

        # Place orders and save
        # Buy order with corresponding sell
        id_buy = bot.portfolio.wallets[sym].product.place_order(buy_price, 'buy', 11/buy_price)
        self.ids_to_cancel.append(id_buy)
        bot.add_order(id_buy, sym, 'buy', time(), 0)
        # Sell order to keep
        id_sell = bot.portfolio.wallets[sym].product.place_order(sell_price, 'sell', 1100 / sell_price)
        self.ids_to_cancel.append(id_sell)
        # Cancelled sell order
        sell_id_drop = bot.portfolio.wallets[sym].product.place_order(sell_price, 'sell', 1100 / sell_price)
        self.ids_to_cancel.append(sell_id_drop)
        bot.add_order(id_sell, sym, 'sell', time(), id_buy)
        # Cancelled buy order
        buy_id_drop = bot.portfolio.wallets[sym].product.place_order(buy_price, 'buy', 11 / buy_price)
        self.ids_to_cancel.append(buy_id_drop)
        bot.add_order(buy_id_drop, sym, 'buy', time(), 11 / buy_price)
        bot.portfolio.auth.cancel_order(buy_id_drop)
        # Buy order without corresponding sell
        buy_stop_id = bot.portfolio.wallets[sym].product.place_order(buy_price, 'buy', 11 / buy_price, stop_price=buy_price*0.998, post_only=False)
        self.ids_to_cancel.append(buy_stop_id)
        bot.add_order(buy_stop_id , sym, 'buy', time(), 0)
        bot.update_orders()

        # Check if orders saved
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertIn(id_buy, csv_df.index, 'Buy order did not save')
        self.assertIn(id_sell, csv_df.index, 'Sell order did not save')
        self.assertIn(id_buy, csv_df.loc[id_sell]['corresponding_order'], 'Corresponding order did not save')
        self.assertIn(buy_stop_id, csv_df.index, 'Unfilled buy order did not save')
        self.assertNotIn(buy_id_drop, csv_df.index, 'Bot kept old by order')

        bot.portfolio.wallets[sym].product.auth_client.cancel_order(sell_id_drop)

        # Create a new instance of the bot
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True)
        bot.update_orders()
        for id in bot.orders.index:
            self.assertIn(id, csv_df.index, 'New instance did not load old orders')

        product = bot.portfolio.wallets[sym].product
        self.assertIn(id_buy, product.orders['buy'].keys(), 'Buy order did not load to new instance product')
        product = bot.portfolio.wallets[sym].product
        self.assertIn(id_sell, product.orders['sell'].keys(), 'Sell order did not load to new instance product')
        self.assertNotIn(sell_id_drop, product.orders['sell'].keys(), 'Product loaded old order')
        self.assertNotIn(sell_id_drop, bot.orders.index, 'Bot loaded old order')

    def test_bot_adds_and_saves_orders_that_fill_immediately(self):
        sym = 'BTC'
        size = 1
        buy_price = 20000
        sell_price = 1000
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH, is_sandbox_api=True)
        buy_order_id = bot.place_order(buy_price, 'buy', size, sym, post_only=False)
        bot.add_order(buy_order_id, sym, 'buy', time(), 0)
        bot.update_orders()
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertIn(buy_order_id, csv_df.index, 'Buy order did not save')

        sell_order_id = bot.place_order(sell_price, 'sell', size, sym, post_only=False)
        bot.add_order(sell_order_id,sym, 'sell', time(), buy_order_id, refresh=False)
        self.assertIn(sell_order_id, bot.orders.index, 'Sell did not save')
        bot.update_orders()
        self.assertNotIn(sell_order_id, bot.orders.index, 'Sell order was not removed')
        bot.update_orders()

        # Check if orders saved
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertNotIn(buy_order_id, csv_df.index, 'Buy order was not removed')


    def test_does_turn_placeholders_into_buy_orders(self):
        sym = 'BTC'
        size = 1
        buy_price = 200000
        spread = 2
        t = time()
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True, syms=('BTC', 'ATOM', 'OXT', 'LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP'))
        wallet = bot.portfolio.wallets[sym]
        bot.place_order_for_nth_currency(buy_price, buy_price*spread, wallet, size, 1, 1, sym)
        bot.buy_place_holders()
        bot.update_orders()

        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertTrue(len(csv_df.index) == 1, 'Buy order did not save')
        np.testing.assert_almost_equal(csv_df.iloc[0]['size']*csv_df.iloc[0]['price'], size*buy_price, 2*bot.portfolio.wallets[sym].product.crypto_res, 'buy order placed with either incorrect size or price')
        self.assertEqual(csv_df.iloc[0]['spread'], spread, 'buy order placed with incorrect spread')
        self.assertIsNotNone(csv_df.iloc[0]['time'], 'buy order placed without time')

    def test_does_up_sell_price_when_out_of_bounds(self):
        # TODO finish this test
        # Setup initial variables
        sym = 'BTC'
        size = 1
        buy_price = 20000 # This price ensures the buy order goes through
        recorded_buy_price = 500 # This is the price that will be used for the test
        sell_price = 1000
        sell_stop_price = 1100
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True)

        # Place buy order
        buy_order_id = bot.place_order(buy_price, 'buy', size, sym, post_only=False)
        bot.add_order(buy_order_id, sym, 'buy', time(), 0, spread=2)
        bot.update_orders()
        bot.orders.at[buy_order_id, 'price'] = recorded_buy_price

        # Place initial sell order
        sell_order_id = bot.place_sell_order(sym, sell_price, 100, bot.portfolio.wallets[sym], 100, buy_order_id, order_type='stop limit', stop=sell_stop_price)
        order = bot.portfolio.auth.get_order(sell_order_id)
        self.assertEqual(order['stop'], 'loss', 'Bot did not place stop loss order')
        bot.update_orders()
        bot.orders.at[buy_order_id, 'price'] = recorded_buy_price

        # Update the orders
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertIn(sell_order_id, csv_df.index, 'Bot stop loss order did not save')

        # Test to see if the bot ups the price accordingly
        bot.place_limit_sells()
        bot.update_orders()
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertIn(buy_order_id, csv_df.index, 'Bot stop loss order did not save')
        sell_order_price = csv_df.iloc[1]['price']
        self.assertGreater(sell_order_price, sell_price)
        self.assertNotIn(sell_order_id, csv_df.index)



    def test_does_emergency_sell(self):
        sym = 'BTC'
        size = 1
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True)
        buy_price = 1.01 * bot.portfolio.wallets[sym].product.get_top_order('bids')
        current_price = 0.8 * buy_price
        # Place buy order
        buy_order_id = bot.place_order(buy_price, 'buy', size, sym, post_only=False)
        bot.add_order(buy_order_id, sym, 'buy', time(), 0)
        bot.update_orders()

        # Add a real corresponding sell
        sell_order_id = bot.place_order(100000, 'sell', size, sym, post_only=False, stop_price=99999)
        bot.add_order(sell_order_id, sym, 'sell', time(), buy_order_id, refresh=False)
        bot.emergency_sell(buy_order_id, current_price)

        self.assertNotIn(sell_order_id, bot.orders.index, 'sell order was not cancelled')
        bot.update_orders()
        self.assertNotIn(buy_order_id, bot.orders.index, 'buy order was not cancelled')
        bot.update_orders()
        # self.assertTrue(len(list(bot.portfolio.auth.get_orders())) == 1)

    def test_does_place_non_stop_limit_orders(self):
        sym = 'BTC'
        size = 1
        desired_spread = 1.0085
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True)
        buy_price = 1.01*bot.portfolio.wallets[sym].product.get_top_order('bids')

        # Place buy order
        buy_order_id = bot.place_order(buy_price, 'buy', size, sym, post_only=False)
        bot.add_order(buy_order_id, sym, 'buy', time(), 0, spread=desired_spread)
        bot.update_orders()
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        test_spread = csv_df.loc[buy_order_id]['spread']
        self.assertEqual(test_spread, desired_spread, 'Spread size did not save properly')

        # Place sell orders
        bot.place_limit_sells()
        bot.update_orders()
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        sell_id = csv_df.index[1]
        np.testing.assert_almost_equal(buy_price*desired_spread, csv_df.loc[sell_id]['price'], 2*bot.portfolio.wallets[sym].product.crypto_res)


    def tearDown(self):
        # Remove any orders placed on the book
        api_base = 'https://api-public.sandbox.pro.coinbase.com'
        auth_client = AuthenticatedClient(self.api_key, self.secret_key, self.passphrase, api_url=api_base)
        auth_client.cancel_all('BTC-USD')
        df = pd.DataFrame(columns=['product_id', 'side', 'price', 'size', 'filled_size', 'corresponding_order', 'time', 'spread'])
        df.to_csv(CSV_PATH)


if __name__ == '__main__':
    unittest.main()
