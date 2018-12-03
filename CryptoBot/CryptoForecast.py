import requests
from requests.auth import AuthBase
import re
from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
import pytz
import pickle
import base64
import hashlib
import hmac
import cbpro
import sys
from tzlocal import get_localzone
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import smtplib
from CryptoBot.CryptoBot_Shared_Functions import convert_time_to_uct
from CryptoBot.CryptoBot_Shared_Functions import get_current_tz


class DataScraper:

    comparison_symbols = ['USD']
    exchange = 'Coinbase'
    aggregate = 1

    def __init__(self, date_from, date_to=None, exchange=None, current_tz=None):

        if exchange:
            self.exchange = exchange

        if current_tz:
            self.tz = current_tz
        else:
            self.tz = get_localzone()

        fmt = '%Y-%m-%d %H:%M:%S'
        self.date_from = convert_time_to_uct(datetime.strptime(date_from, fmt))

        if date_to:
            self.date_to = convert_time_to_uct(datetime.strptime(date_to, fmt))
        else:
            self.date_to = None

    def datedelta(self, units):
        d1_ts = time.mktime(self.date_from.timetuple())
        if self.date_to:
            d2_ts = time.mktime(self.date_to.timetuple())
        else:
            d2_ts = time.mktime(datetime.now().timetuple())

        if units == "days":
            del_date = int((d2_ts-d1_ts)/86400)
        elif units == "hours":
            del_date = int((d2_ts - d1_ts) / 3600)
        elif units == "minutes":
            del_date = int((d2_ts - d1_ts) / 60)

        return del_date

    def price(self, symbol):
        comparison_symbols = self.comparison_symbols
        exchange = self.exchange

        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}' \
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        try:
            page = requests.get(url)
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
        data = page.json()
        return data

    def create_data_frame(self, url, symbol, return_time_stamp=False):
        try:
            page = requests.get(url)
            data = page.json()['Data']
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
            data = page.json()['Data']
        symbol = symbol.upper()
        df = pd.DataFrame(data)
        df = df.add_prefix(symbol + '_')
        df.insert(loc=0, column='date', value=[datetime.fromtimestamp(d) for d in df[symbol + '_time']])

        if return_time_stamp:
            time_stamps = df[symbol + '_time'].values
            time_stamp = time_stamps[0]
            df = df.drop(columns=[symbol + '_time'])
            return df, time_stamp

        df = df.drop(columns=[symbol + '_time']) #Drop this because the unix timestamp column is no longer needed
        return df

    def daily_price_historical(self, symbol, all_data=True, limit=1):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        if self.date_from:
            all_data=False
            limit = self.datedelta("days")

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate, int(self.date_to.timestamp()))
        else:
            url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        if all_data:
            url += '&allData=true'

        df = self.create_data_frame(url, symbol)
        return df

    def hourly_price_historical(self, symbol):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("hours")

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate, int(self.date_to.timestamp()))
        else:
            url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

        df = self.create_data_frame(url, symbol)
        return df

    def minute_price_historical(self, symbol):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("minutes")
        first_lim = limit
        minute_lim = 2000

        if limit > minute_lim:
            first_lim = minute_lim

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate, self.date_to.timestamp())
        else:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate)
        temp_url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

        loop_len = int(np.ceil(limit / minute_lim))
        if limit > minute_lim: # This if statement is to allow the gathering of historical minute data beyond 2000 points (the limit)
            df, time_stamp = self.create_data_frame(url, symbol, return_time_stamp=True)
            for num in range(1, loop_len):
                toTs = time_stamp - 60 # have to subtract a value of 60 had to be added to avoid repeated indices
                url_new = temp_url + '&toTs={}'.format(toTs)
                if num == (loop_len - 1):
                    url_new = 'https://min-api.CryptoCompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                        .format(symbol.upper(), comparison_symbol.upper(), limit - num * minute_lim, self.aggregate, toTs)
                df_to_append, time_stamp = self.create_data_frame(url_new, symbol, return_time_stamp=True)
                df = df_to_append.append(df, ignore_index=True) # The earliest data go on top
            return df

        df = self.create_data_frame(url, symbol)
        return df

    def get_historical_price(self, symbol, unit='min'):

        if unit == 'min':
            data = self.minute_price_historical(symbol)
        elif unit == 'hr':
            data = self.hourly_price_historical(symbol)
        elif unit == 'day':
            data = self.daily_price_historical(symbol)
        else:
            raise ValueError('unit must be in [''min'', ''hr'', ''day'']')

        return data

    def coin_list(self):
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def coin_snapshot_full_by_id(self, symbol, symbol_id_dict={}):#TODO fix the id argument mutability

        if not symbol_id_dict:
            symbol_id_dict = {
                'BTC': 1182,
                'ETH': 7605,
                'LTC': 3808
            }
        symbol_id = symbol_id_dict[symbol.upper()]
        url = 'https://www.cryptocompare.com/api/data/coinsnapshotfullbyid/?id={}' \
            .format(symbol_id)
        try:
            page = requests.get(url)
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
        data = page.json()['Data']
        return data

    def live_social_status(self, symbol, symbol_id_dict=None):

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

    def news(self, symbols, categories='Regulation,Altcoin,Blockchain,Mining,Trading,Market', date_before_ts=None):
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        if len(symbols) > 1:
            symbols_str = ','.join(symbols)
        else:
            symbols_str = symbols[0]

        if date_before_ts is None:
            date_before_ts = convert_time_to_uct(datetime.now()).timestamp()

        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={},{}&sortOrder=latest&lTs={}" \
        .format(symbols_str.upper(), categories.title(), int(date_before_ts))
        try:
            page = requests.get(url)
            data = page.json()['Data']
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
            data = page.json()['Data']
        return data

    def iteratively_scrape_news(self, symbols, categories='Regulation,Altcoin,Blockchain,Mining,Trading,Market'):
        to_ts = self.date_to.timestamp()
        from_ts = self.date_from.timestamp()
        data = self.news(symbols, categories, to_ts)
        earliest_ts = data[-1]['published_on']
        # iter_count ensures there are no infinite loops ( assumes that less than 1 article is published per minute )
        iter_count = 0
        news_len_cutoff = 1
        fmt = '%Y-%m-%d %H:%M'

        # --Continues crapes articles if requested time goes beyond one return--
        while (earliest_ts > from_ts) or (iter_count > (to_ts-from_ts)/60):
            data_new = self.news(symbols, categories, earliest_ts)
            data.extend(data_new)
            earliest_ts = data[-1]['published_on']
            iter_count += 1
            print('Scraped back to ' + datetime.fromtimestamp(earliest_ts).strftime(fmt) + ' ' + get_current_tz())

        data.reverse()
        # --Ensures that only articles between date_from and date_to are kept--
        for news_article in data:
            current_ts = news_article['published_on']
            if current_ts > from_ts:
                break
            news_len_cutoff += 1

        data = data[news_len_cutoff::]
        return data

class FormattedData:

    raw_data = None

    def __init__(self, date_from, date_to, ticker, sym_list=None, time_units='min', news_hourly_offset=5):
        if sym_list is None:
            sym_list = ['BTC', 'LTC']

        sym_list.insert(0, ticker)

        self.sym_list = sym_list
        self.ticker = ticker
        self.date_from = date_from
        self.date_to = date_to
        self.time_units = time_units

    def scrape_data(self, date_to=None):
        # If a value for date_to is entered then the object will attempt to create new data to add to an existing dataset
        if date_to:
            date_from = self.date_to
            self.date_to = date_to
        else:
            date_from = self.date_from
            date_to = self.date_to

        scraper = DataScraper(date_from=date_from, date_to=date_to)

        for sym in self.sym_list:

             current_scrape = scraper.get_historical_price(symbol=sym, unit=self.time_units)
             if self.raw_data is None:
                 self.raw_data = current_scrape

             else:
                 current_scrape = current_scrape.drop(['date'], axis=1)
                 self.raw_data = pd.concat([self.raw_data, current_scrape], axis=1, join_axes=[self.raw_data.index])

