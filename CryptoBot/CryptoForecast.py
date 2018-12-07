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
from CryptoBot.CryptoBot_Shared_Functions import progress_printer

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
        # This formats data as a pandas dataframe before returning it
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

        # This turns the date column into a column of datetime objects based on timestamps gotten from the symbol + '_time' column (which is dropped before returning the dataframe)
        df.insert(loc=0, column='date', value=[datetime.fromtimestamp(d) for d in df[symbol + '_time']])

        if return_time_stamp:
            # If this option is used the first timestamp is returned for reference, can help with varying time zones
            time_stamps = df[symbol + '_time'].values
            time_stamp = time_stamps[0]
            df = df.drop(columns=[symbol + '_time'])
            return df, time_stamp

        df = df.drop(columns=[symbol + '_time']) # Drop this because the unix timestamp column is no longer needed
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

    def coin_snapshot_full_by_id(self, symbol, symbol_id_dict={}):

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

    def __init__(self, date_from, date_to, ticker, sym_list=None, time_units='min', suppression=False, news_hourly_offset=5):
        if sym_list is None:
            sym_list = ['BTC', 'LTC']

        sym_list.insert(0, ticker)

        self.sym_list = sym_list
        self.ticker = ticker
        self.date_from = date_from
        self.date_to = date_to
        self.time_units = time_units
        self.suppress_output = suppression
        self.news_hourly_offset = news_hourly_offset # news hourly offset is the cutoff (in hours) for where past news is relevant

    def scrape_data(self):
        # If a value for date_to is entered then the object will attempt to create new data to add to an existing dataset
        date_from = self.date_from
        date_to = self.date_to

        fmt = '%Y-%m-%d %H:%M:%S'
        datetime_from = datetime.strptime(date_from, fmt)
        news_datetime = datetime_from-timedelta(hours=self.news_hourly_offset)
        news_date_from = news_datetime.strftime(fmt)

        scraper = DataScraper(date_from=date_from, date_to=date_to)
        news_scraper = DataScraper(date_from=news_date_from, date_to=date_to)

        for sym in self.sym_list:

             current_scrape = scraper.get_historical_price(symbol=sym, unit=self.time_units)
             if self.raw_data is None:
                 self.raw_data = current_scrape

             else:
                 current_scrape = current_scrape.drop(['date'], axis=1)
                 self.raw_data = pd.concat([self.raw_data, current_scrape], axis=1, join_axes=[self.raw_data.index])

        if 'BTC' in self.sym_list:
            self.raw_news = news_scraper.iteratively_scrape_news(self.sym_list)
        else:
            self.raw_news = news_scraper.iteratively_scrape_news(self.sym_list + ['BTC'])

    def format_news_data(self):
        # n is an argitrary number used to scale the news data
        news_sentiment = np.array([])
        news_pub_date = np.array([])

        if self.raw_data is None:
            raise ValueError('raw_data attribute is not defined')

        for news in self.raw_news:
            news_sentiment = np.append(news_sentiment, txb(news['title']).sentiment.polarity)
            news_pub_date = np.append(news_pub_date, news['published_on'])

        return news_sentiment, news_pub_date

    def collect_news_counts_and_sentiments(self, n=4500):

        news_sentiment, news_pub_date = self.format_news_data()

        sentiment_col = np.array([])
        count_col = np.array([])

        for i in range(0, len(self.raw_data.index)):

            progress_printer(len(self.raw_data.index), i, digit_resolution=2, tsk='News Formatting', supress_output=self.suppress_output)

            t = self.raw_data.date[i]
            current_ts = convert_time_to_uct(t).timestamp()
            cutoff_ts = current_ts - self.news_hourly_offset * 3600
            current_news_mask = np.argwhere((news_pub_date > cutoff_ts) & (news_pub_date < current_ts))

            current_sentiment = 0
            scaled_count = 0
            relevant_sentiments = news_sentiment[current_news_mask]
            relevant_pub_dates = news_pub_date[current_news_mask]
            for j in range(0, len(relevant_pub_dates)):
                coeff = n / (n + current_ts - news_pub_date[j])
                scaled_count += coeff
                current_sentiment += coeff*relevant_sentiments[j]

            sentiment_col = np.append(sentiment_col, current_sentiment)
            count_col = np.append(count_col, scaled_count)

        return sentiment_col, count_col

    def merge_raw_data_frames(self):
        if self.raw_data is None:
            raise ValueError('raw_data attribute is not defined')

        sentiment_col, count_col = self.collect_news_counts_and_sentiments()
        news_data_frame = pd.DataFrame({'Sentiment':sentiment_col, 'Count':count_col})
        self.raw_data = pd.concat([self.raw_data, news_data_frame], axis=1, join_axes=[news_data_frame.index])

    def format_data_for_training_or_testing(self, forecast_offset=30, predicted_quality='high'):

        # Create output for training
        predicted_quality_vec = self.raw_data[self.ticker + '_' + predicted_quality].values
        output_vec = predicted_quality_vec[forecast_offset::]

        # Create input for training
        scaler = StandardScaler()
        temp_input_arr = self.raw_data.drop(columns='date').values
        temp_input_arr = temp_input_arr[0:-(forecast_offset), ::]
        temp_input_arr = scaler.fit_transform(temp_input_arr)
        input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], temp_input_arr.shape[1], 1)

        return output_vec, input_arr

    def format_data_for_train_test_split(self, train_test_split = 0.33, forecast_offset=30, predicted_quality='high'):
        if (train_test_split >= 1) or (train_test_split <= 0):
            raise ValueError('train_test_split must be in (0, 1)')

        output_vec, input_arr = self.format_data_for_training_or_testing(forecast_offset=forecast_offset, predicted_quality=predicted_quality)

        training_length = (int(len(input_arr)*(1-train_test_split)))
        training_input_arr = input_arr[0:training_length, ::, ::]
        test_input_arr = input_arr[training_length::, ::, ::]
        training_output_vec = output_vec[0:training_length]
        test_output_vec = output_vec[training_length::]

        return training_output_vec, test_output_vec, training_input_arr, test_input_arr

    def format_data_for_prediction(self):

        # Create input for training
        scaler = StandardScaler()
        temp_input_arr = self.raw_data.drop(columns='date').values
        temp_input_arr = scaler.fit_transform(temp_input_arr)
        input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], temp_input_arr.shape[1], 1)

        return input_arr

    def format_data(self, data_type, train_test_split = 0.33, forecast_offset=30, predicted_quality='high'):
        if self.raw_data is None:
            raise ValueError('raw_data attribute is not defined')

        pred_data = {'training output':None, 'training input':None, 'output':None, 'input':None}

        data_type = data_type.lower()

        if (data_type == 'test') or (data_type == 'train'):
            output_vec, input_arr = self.format_data_for_training_or_testing(forecast_offset=forecast_offset,
                                                                             predicted_quality=predicted_quality)
            pred_data['input'] = input_arr
            pred_data['output'] = output_vec

        elif (data_type == 'train/test') or (data_type == 'test/train'):
            training_output_vec, test_output_vec, training_input_arr, test_input_arr = self.format_data_for_train_test_split(train_test_split=train_test_split, forecast_offset=forecast_offset, predicted_quality=predicted_quality)
            pred_data['training input'] = training_input_arr
            pred_data['training output'] = training_output_vec
            pred_data['input'] = test_input_arr
            pred_data['output'] = test_output_vec

        elif data_type == 'forecast':
            input_arr = self.format_data_for_prediction()
            pred_data['input'] = input_arr

        return pred_data

    def save_raw_data(self, file_name=None):

        if len(self.sym_list) > 1:
            symbols_str = self.ticker + '_ticker_' + ','.join(self.sym_list[1::])
        else:
            symbols_str = self.sym_list[0]
        tz = get_current_tz()

        if file_name is None:
            file_name = '-' + self.time_units + 'by' + self.time_units + '_symbols_' + symbols_str  + \
                              '_from_' + self.date_from + tz + '_to_' + self.date_to + tz + '.pickle'
            file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet' + file_name.replace(
                ' ', '_')

        with open(file_name, 'wb') as cp_file_handle:
            pickle.dump(self.raw_data, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)

