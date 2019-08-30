from CryptoBot import SpreadBot as sb
from CryptoBot.CryptoBot_Shared_Functions import num2str
from time import time

DEFAULT_SYM_PAIRS= (
    'EOS-USD',
    'REP-USD',
    'LTC-USD',
    'XTZ-BTC',
    'EOS-BTC',
    'BTC-USD',
    'LTC-BTC',
    'BCH-BTC',
    'REP-BTC',
    'XLM-BTC',
    'LINK-ETH',
    'ZRX-BTC',
    'LINK-USD',
    'XRP-USD',
    'ETH-BTC',
    'XTZ-USD',
    'BCH-USD',
    'XLM-USD',
    'XRP-BTC',
    'ETC-USD',
    'ETH-USD',
    'ALGO-USD',
    'ETC-BTC',
    'ZRX-USD',
    'BAT-USDC',
    'BAT-ETH',
    'ZEC-BTC',
    'ZEC-USDC',
    'BTC-USD',
    'ETH-BTC',
    'ETH-USDC',
    'BTC-USDC',
)

#     (
# 'BAT-USDC',
# 'BAT-ETH',
# 'ZEC-BTC',
# 'ZEC-USDC',
# 'BTC-USD',
# 'ETH-BTC',
# 'ETH-USDC',
# 'ETH-USD'
# )

class TriangularArbitridgeBot(sb.Bot):

    def __init__(self, api_key, secret_key, passphrase, syms=DEFAULT_SYM_PAIRS):
        super(TriangularArbitridgeBot, self).__init__(api_key, secret_key, passphrase, syms=syms, is_sandbox_api=False, base_currency=('ETH', 'USD', 'USDC', 'BTC'))

    def find_arbitradge_spreads( self, base_ticker, quote_ticker, stable_ticker='USD' ):
        quote_usd = self.portfolio.wallets[quote_ticker + '-' + stable_ticker].product
        base_usd = self.portfolio.wallets[base_ticker + '-' + stable_ticker].product
        base_quote = self.portfolio.wallets[base_ticker + '-' + quote_ticker].product

        p1 = quote_usd.get_top_order('asks')
        p2 = base_quote.get_top_order('asks')
        p3 = base_usd.get_top_order('bids')

        p1r = float(base_usd.order_book['asks'][0][0])
        p2r = float(base_quote.order_book['bids'][0][0])
        p3r = float(quote_usd.order_book['bids'][0][0])
        amt = 100
        forward_value = (0.9975**3) * p3 * amt / ( p1 * p2 ) - 100
        reverse_value = (0.9975**3) * p3r * p2r * amt / p1r - 100

        # p1 = prod_u.get_top_order('bids')
        # p2 = prod_u.get_top_order('asks')
        # p2 = prod_e.get_top_order('asks')
        # p3 = prod_u.get_top_order('asks')
        # prod_ue = Product(api, secret, passphrase)
        # p3 = prod_ue.get_top_order('asks')
        # amt = 12
        # amt * p1 * 0.997 / p2
        # amt * 0.997 / p1
        # p2 * amt * (0.997 ** 3) / p1
        # p3 * p2 * amt * (0.997 ** 3) / p1
        # p3 * p2 * amt / p1

        return forward_value, reverse_value

    #TODO make script to find optimal chain
if __name__ == "__main__":
    api_input = input('What is the api key? ')
    secret_input = input('What is the secret key? ')
    passphrase_input = input('What is the passphrase? ')
    bot = TriangularArbitridgeBot(api_input, secret_input, passphrase_input)
    print('starting')
    t0 = time()
    t = 0
    i = 0
    print(str(int(time())))
    sym_pairs = DEFAULT_SYM_PAIRS
    while t < 24 * 3600:

        t = time() - t0

        for pair in sym_pairs:
            if 'USD' in pair:
                continue
            base = pair.split('-')[0]
            quote = pair.split('-')[1]
            if (base + '-USD') in bot.portfolio.wallets.keys():
                stable = 'USD'
            elif (base + '-USDC') in bot.portfolio.wallets.keys():
                stable = 'USDC'
            else:
                continue
            f, r = bot.find_arbitradge_spreads(base, quote, stable_ticker=stable)
            if (f > -0.5) or (r > -0.5):
                print(pair)
                print(str(int(time())))
                print('Forwad value: ' + num2str(f, 3) + '%')
                print('Reverse value: ' + num2str(r, 3) + '%\n')
                sym_pairs = (pair,)
            else:
                sym_pairs = DEFAULT_SYM_PAIRS