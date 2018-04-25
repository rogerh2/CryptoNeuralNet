import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

class cryptocompare:

    symbol = 'LTC'
    comparison_symbols = ['USD']
    exchange = ''

    def __init__(self, symbol=None, comparison_symbols=None, exchange=None, date_from=None):

        if symbol:
            self.symbol = symbol

        if comparison_symbols:
            self.comparison_symbols = comparison_symbols

        if exchange:
            self.exchange = exchange

        fmt = '%Y-%m-%d %H:%M:%S %Z'
        self.date_from = datetime.strptime(date_from, fmt)

    def datedelta(self, units):
        d1_ts = time.mktime(self.date_from.timetuple())
        d2_ts = time.mktime(datetime.now().timetuple())

        if units == "days":
            del_date = int((d2_ts-d1_ts)/86400)
        elif units == "hours":
            del_date = int((d2_ts - d1_ts) / 3600)
        elif units == "minutes":
            del_date = int((d2_ts - d1_ts) / 60)

        return del_date

    def price(self):
        symbol = self.symbol
        comparison_symbols = self.comparison_symbols
        exchange = self.exchange

        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}' \
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()
        return data

    def daily_price_historical(self, all_data=True, limit=1, aggregate=1):
        symbol = self.symbol
        comparison_symbol = self.comparison_symbols
        exchange = self.exchange
        if self.date_from:
            all_data=False
            limit = self.datedelta("days")

        url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        if all_data:
            url += '&allData=true'
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.fromtimestamp(d) for d in df.time]
        return df

    def hourly_price_historical(self, aggregate=1):
        symbol = self.symbol
        comparison_symbol = self.comparison_symbols
        exchange = self.exchange
        limit = self.datedelta("hours")

        url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df

    def minute_price_historical(self, aggregate = 1):
        symbol = self.symbol
        comparison_symbol = self.comparison_symbols
        exchange = self.exchange
        limit = self.datedelta("hours")

        url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df

    def coin_list(self):
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def coin_snapshot_full_by_id(self, symbol_id_dict={}):
        symbol = self.symbol
        
        if not symbol_id_dict:
            symbol_id_dict = {
                'BTC': 1182,
                'ETH': 7605,
                'LTC': 3808
            }
        symbol_id = symbol_id_dict[symbol.upper()]
        url = 'https://www.cryptocompare.com/api/data/coinsnapshotfullbyid/?id={}' \
            .format(symbol_id)
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def live_social_status(self, symbol_id_dict={}):
        symbol = self.symbol

        if not symbol_id_dict:
            symbol_id_dict = {
                'BTC': 1182,
                'ETH': 7605,
                'LTC': 3808
            }
        symbol_id = symbol_id_dict[symbol.upper()]
        url = 'https://www.cryptocompare.com/api/data/socialstats/?id={}' \
            .format(symbol_id)
        page = requests.get(url)
        data = page.json()['Data']
        return data