import unittest
import pandas as pd
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
        id_buy = bot.portfolio.wallets[sym].product.place_order(buy_price, 'buy', 11/buy_price)
        self.ids_to_cancel.append(id_buy)
        bot.add_order(id_buy, sym, 'buy', time(), None)
        id_sell = bot.portfolio.wallets[sym].product.place_order(sell_price, 'sell', 1100 / sell_price)
        self.ids_to_cancel.append(id_sell)
        id_drop = bot.portfolio.wallets[sym].product.place_order(sell_price, 'sell', 1100 / sell_price)
        self.ids_to_cancel.append(id_drop)
        bot.add_order(id_sell, sym, 'sell', time(), id_buy)
        bot.update_orders()

        # Check if orders saved
        csv_df = pd.read_csv(CSV_PATH, index_col=0)
        self.assertIn(id_buy, csv_df.index, 'Buy order did not save')
        self.assertIn(id_sell, csv_df.index, 'Sell order did not save')
        self.assertIn(id_buy, csv_df.loc[id_sell]['corresponding_order'], 'Corresponding order did not save')

        bot.portfolio.wallets[sym].product.auth_client.cancel_order(id_drop)

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
        self.assertNotIn(id_drop, product.orders['sell'].keys(), 'Product loaded old order')
        self.assertNotIn(id_drop, bot.orders.index, 'Bot loaded old order')

    def test_does_turn_placeholders_into_buy_orders(self):
        bot = PSMPredictBot(self.api_key, self.secret_key, self.passphrase, order_csv_path=CSV_PATH,
                            is_sandbox_api=True)
        buy_price = 1000
        sell_price = 10 ** 6
        sym = 'BTC'

    def tearDown(self):
        # Remove any orders placed on the book
        api_base = 'https://api-public.sandbox.pro.coinbase.com'
        auth_client = AuthenticatedClient(self.api_key, self.secret_key, self.passphrase, api_url=api_base)
        orders = auth_client.get_orders()
        all_ids = [x['id'] for x in orders]
        for id in self.ids_to_cancel:
            if id in all_ids:
                auth_client.cancel_order(id)
        df = pd.DataFrame(columns=['product_id', 'side', 'price', 'size', 'filled_size', 'corresponding_order', 'time'])
        df.to_csv(CSV_PATH)


if __name__ == '__main__':
    unittest.main()
