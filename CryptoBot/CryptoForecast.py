import sys
#sys.path.append("home/rjhii/CryptoNeuralNet/CryptoPredict")
# use the below for AWS
sys.path.append("home/ubuntu/CryptoNeuralNet")
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
from cbpro import PublicClient
from tzlocal import get_localzone
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import backend as K
from sklearn.preprocessing import StandardScaler

#For AWS
# from CryptoBot_Shared_Functions import convert_time_to_uct
# from CryptoBot_Shared_Functions import get_current_tz
# from CryptoBot_Shared_Functions import progress_printer
# from CryptoBot_Shared_Functions import rescale_to_fit
# from CryptoBot_Shared_Functions import num2str

#For Local
from CryptoBot.CryptoBot_Shared_Functions import convert_time_to_uct
from CryptoBot.CryptoBot_Shared_Functions import get_current_tz
from CryptoBot.CryptoBot_Shared_Functions import progress_printer
from CryptoBot.CryptoBot_Shared_Functions import rescale_to_fit
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoBot.CryptoBot_Shared_Functions import multiple_choice_question_with_prompt

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

    # --Methods for the CryptoCompare API--

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
            sym_list = [ticker]
        else:
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
                 self.raw_data = pd.concat([self.raw_data, current_scrape], axis=1).reindex(current_scrape.index)

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

            progress_printer(len(self.raw_data.index), i, digit_resolution=2, tsk='News Formatting', suppress_output=self.suppress_output)

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
        self.raw_data = pd.concat([self.raw_data, news_data_frame], axis=1).reindex(news_data_frame.index)

    def str_list_to_timestamp(self, datetime_str_list):
        fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
        utc = pytz.UTC
        localized_datetime_objects = [utc.localize(datetime.strptime(str, fmt)) for str in datetime_str_list]
        time_stamps = np.array([dt.timestamp() for dt in localized_datetime_objects])

        return time_stamps

    def normalize_order_book_row(self, base_value, full_row):
        i = 6

        normalized_row = full_row.values

        while i < (len(full_row.columns)):
            col_title = str(i)
            order_price = full_row[col_title]
            normalized_order_price = order_price/base_value
            normalized_row[0, i] = normalized_order_price
            i += 3

        normalized_row = full_row.values
        normalized_row = np.delete(normalized_row, [3])

        return  normalized_row

    def normalize_fill_array_and_order_book(self, order_book, fills):
        # This function takes advantage of the Markovian nature of crypto prices and normalizes the fills by the current
        # top bid. This is intended to make the predictions homogeneous no matter what the current price is
        # Note: current setup has prices at every third entry, should change to have identifying headers
        fill_ind = 0
        order_book_ts_vals = order_book.ts.values
        order_book_top_bid_vals = order_book['3'].values # The fills are normalized off the top bid

        fill_ts_vals = fills.time.values
        fill_price_vals = fills.price.values


        current_fill_ts = fill_ts_vals[fill_ind]
        normalized_fills = np.array([])

        for order_book_ind in range(0, len(order_book_ts_vals)):

            ts = order_book_ts_vals[order_book_ind]
            current_bid = order_book_top_bid_vals[order_book_ind]

            while ts > current_fill_ts:
                fill_ind += 1
                if fill_ind == len(fill_price_vals):
                    # If there are more order book states after the last fill than this stops early
                    return normalized_fills, normalized_order_book
                current_fill_ts = fill_ts_vals[fill_ind]

            current_fill = fill_price_vals[fill_ind]
            current_normalized_fill = current_fill/current_bid
            normalized_fills = np.append(normalized_fills, current_normalized_fill)

            current_order_book_row = order_book[order_book.index==order_book_ind]
            current_order_book_row = current_order_book_row.drop(['ts'], axis=1)
            normalized_order_book_row = self.normalize_order_book_row(current_bid, current_order_book_row)
            if order_book_ind == 0:
                normalized_order_book = normalized_order_book_row
            else:
                normalized_order_book = np.vstack((normalized_order_book, normalized_order_book_row))



        return normalized_fills, normalized_order_book

    # TODO edit format functions to accept preexisting output arrays (for use with the orderbook + fill data)?
    def format_data_for_training_or_testing(self, forecast_offset=30, predicted_quality='high'):

        # Create output for training
        predicted_quality_vec = self.raw_data[self.ticker + '_' + predicted_quality].values
        output_vec = predicted_quality_vec[forecast_offset::]

        # Create input for training
        scaler = StandardScaler()
        temp_input_arr = self.raw_data.drop(columns='date').values
        temp_input_arr = temp_input_arr[0:-(forecast_offset), ::]
        temp_input_arr = scaler.fit_transform(temp_input_arr)
        input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], 1, temp_input_arr.shape[1])

        # Create x labels (which are datetime objects)
        x_labels = self.raw_data.date.values[forecast_offset::]

        return output_vec, input_arr, x_labels

    def format_data_for_train_test_split(self, train_test_split = 0.33, forecast_offset=30, predicted_quality='high'):
        if (train_test_split >= 1) or (train_test_split <= 0):
            raise ValueError('train_test_split must be in (0, 1)')

        output_vec, input_arr, x_labels = self.format_data_for_training_or_testing(forecast_offset=forecast_offset, predicted_quality=predicted_quality)

        training_length = (int(len(input_arr)*(1-train_test_split)))
        training_input_arr = input_arr[0:training_length, ::, ::]
        test_input_arr = input_arr[training_length::, ::, ::]
        training_output_vec = output_vec[0:training_length]
        test_output_vec = output_vec[training_length::]
        x_labels = x_labels[training_length::]

        return training_output_vec, test_output_vec, training_input_arr, test_input_arr, x_labels

    def format_data_for_prediction(self):

        # Create input for training
        scaler = StandardScaler()
        temp_input_arr = self.raw_data.drop(columns='date').values
        temp_input_arr = scaler.fit_transform(temp_input_arr)
        input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], 1, temp_input_arr.shape[1])

        return input_arr

    def format_data(self, data_type, train_test_split = 0.33, forecast_offset=30, predicted_quality='high'):
        if self.raw_data is None:
            raise ValueError('raw_data attribute is not defined')

        pred_data = {'training output':None, 'training input':None, 'output':None, 'input':None, 'x labels':None}

        data_type = data_type.lower()

        if data_type in ['train', 'test', 'optimize']:
            output_vec, input_arr, x_labels = self.format_data_for_training_or_testing(forecast_offset=forecast_offset,
                                                                             predicted_quality=predicted_quality)
            pred_data['input'] = input_arr
            pred_data['output'] = output_vec
            pred_data['x labels'] = x_labels

        elif (data_type == 'train/test') or (data_type == 'test/train'):
            training_output_vec, test_output_vec, training_input_arr, test_input_arr, x_labels = self.format_data_for_train_test_split(train_test_split=train_test_split, forecast_offset=forecast_offset, predicted_quality=predicted_quality)
            pred_data['training input'] = training_input_arr
            pred_data['training output'] = training_output_vec
            pred_data['input'] = test_input_arr
            pred_data['output'] = test_output_vec
            pred_data['x labels'] = x_labels

        elif data_type == 'forecast':
            input_arr = self.format_data_for_prediction()
            pred_data['input'] = input_arr

        return pred_data

    def save_raw_data(self, file_name=None):

        if len(self.sym_list) > 1:
            symbols_str = '__ticker_' + self.ticker + '_aux_' + ','.join(self.sym_list[1::]) + '_'
        else:
            symbols_str = self.sym_list[0]
        tz = get_current_tz()

        if file_name is None:
            file_name = self.time_units + 'by' + self.time_units + symbols_str  + \
                              '_from_' + self.date_from + tz + '_to_' + self.date_to + tz + '.pickle'
            file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/' + file_name.replace(
                ' ', '_')

        with open(file_name, 'wb') as cp_file_handle:
            pickle.dump(self.raw_data, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)\

class FormattedCoinbaseProData:

    time_units = 'fill'
    historical_order_books = None
    historical_fills = None

    def __init__(self, historical_order_books_path=None, historical_fills_path=None, suppress_output=False):

        self.suppress_output = suppress_output

        if historical_order_books_path is not None:
            historical_order_books = pd.read_csv(historical_order_books_path)
            if 'Unnamed: 0' in historical_order_books.columns.values:
                # Some data files have the above header containing the indices, this gets rid of it
                self.historical_order_books = historical_order_books.drop(['Unnamed: 0'], axis=1)
            else:
                self.historical_order_books = historical_order_books
        if historical_fills_path is not None:
            historical_fills = pd.read_csv(historical_fills_path)
            if 'Unnamed: 0' in historical_fills.columns.values:
                # Some data files have the above header containing the indices, this gets rid of it
                self.historical_fills = historical_fills.drop(['Unnamed: 0'], axis=1)
            else:
                self.historical_fills = historical_fills

    def str_list_to_timestamp(self, datetime_str_list):
        fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
        utc = pytz.UTC
        total_len = len(datetime_str_list)
        for i in range(0, total_len):
            current_date_str = datetime_str_list[i]
            if '.' not in current_date_str:
                new_date_str = current_date_str[0:-1] + '.000Z'
                datetime_str_list[i] = new_date_str


        localized_datetime_objects = [utc.localize(datetime.strptime(str, fmt)) for str in datetime_str_list]
        time_stamps = np.array([dt.timestamp() for dt in localized_datetime_objects])

        return time_stamps

    def average_orderbook_features(self, start_col, full_row):
        i = start_col
        full_values = np.array([])

        while i < (len(full_row.columns)):
            col_title = str(i)
            full_values = np.append(full_values, full_row[col_title])
            i += 3

        ans = np.mean(full_values)

        return ans

    def price_at_max_order_size(self, full_row):
        i = 1
        full_values = np.array([])

        while i < (len(full_row.columns)):
            col_title = str(i)
            full_values = np.append(full_values, full_row[col_title])
            i += 3

        max_size_ind = np.argmax(full_values)
        ans = full_row[str(3 * (max_size_ind))]

        return ans, max_size_ind

    def normalize_order_book_row(self, price_base_value, full_row):
        i = 0
        # TODO get rid of cringy repition
        all_values = full_row.values
        normalized_row = np.array([])
        _, max_size_ind = self.price_at_max_order_size(full_row)
        normalized_row = np.append(normalized_row, all_values[0, 0:2])
        normalized_row = np.append(normalized_row, all_values[0, 60:62])
        normalized_row = np.append(normalized_row, all_values[0, max_size_ind:(max_size_ind+2)])
        normalized_row = np.array([normalized_row])

        # size_base_value = self.average_orderbook_features(1, full_row)
        # num_orders_base_value = self.average_orderbook_features(2, full_row)
        #
        #
        # while i < (len(full_row.columns)):
        #     price_col_title = str(i)
        #     size_col_title = str(i + 1)
        #     num_orders_col_title = str(i + 2)
        #
        #     order_price = full_row[price_col_title]
        #     order_size = full_row[size_col_title]
        #     order_num = full_row[num_orders_col_title]
        #
        #     normalized_order_price = order_price / price_base_value
        #     normalized_row[0, i] = normalized_order_price
        #     normalized_size = order_size / size_base_value
        #     normalized_row[0, i + 1] = normalized_size
        #     normalized_order_num = order_num / num_orders_base_value
        #     normalized_row[0, i + 2] = normalized_order_num
        #     i += 3

        #normalized_row = np.delete(normalized_row, [0])

        return  normalized_row

    def normalize_fill_array_and_order_book(self):
        # This function takes advantage of the Markovian nature of crypto prices and normalizes the fills by the current
        # top bid. This is intended to make the predictions homogeneous no matter what the current price is
        # Note: current setup has prices at every third entry, should change to have identifying headers
        fills = self.historical_fills
        normalized_order_book = None

        if fills is not None:
            fill_ind = 0
            fill_ts_vals = self.str_list_to_timestamp(fills.time.values)
            fill_price_vals = fills.price.values

            current_fill_ts = fill_ts_vals[fill_ind]
            current_fill = fill_price_vals[fill_ind]
            normalized_fills = np.array([])
        else:
            fill_ind = 0
            fill_ts_vals = None
            fill_price_vals = None

            current_fill_ts = None
            current_fill = None
            normalized_fills = np.array([])

        order_book = self.historical_order_books
        order_book_ts_vals = order_book.ts.values
        order_book_top_bid_vals = order_book['0'].values # The fills are normalized off the top bid
        order_book_top_ask_vals = order_book['60'].values  # The fills are normalized off the top bid

        last_current_bid = 0
        last_current_ask = 0

        test_books = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/current_test_books_031919.pickle'
        test_fills = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/current_test_fills_031919.pickle'

        for order_book_ind in range(0, len(order_book_ts_vals)):

            progress_printer(len(order_book_ts_vals), order_book_ind, suppress_output=self.suppress_output)

            ts = order_book_ts_vals[order_book_ind]
            current_bid = order_book_top_bid_vals[order_book_ind]
            current_ask = order_book_top_ask_vals[order_book_ind]
            if (current_bid == last_current_bid) and (current_ask == last_current_ask):
                continue

            if fills is not None:
                while (ts > current_fill_ts) or (current_fill-np.abs(fill_price_vals[fill_ind-1]) < 0.0005*current_fill):
                    fill_ind += 1
                    if fill_ind == len(fill_price_vals):
                        # TODO delete this and formalize
                        # with open(
                        #         test_books,
                        #         'wb') as cp_file_handle:
                        #     pickle.dump(normalized_order_book, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        #
                        # with open(
                        #         test_fills,
                        #         'wb') as cp_file_handle:
                        #     pickle.dump(normalized_fills, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        # If there are more order book states after the last fill than this stops early
                        return normalized_fills, normalized_order_book
                    current_fill_ts = fill_ts_vals[fill_ind]
                    current_fill = fill_price_vals[fill_ind]

                fill_base_val = current_bid #self.price_at_max_order_size(current_order_book_row)
                #price_base_val = self.average_orderbook_features(0, current_order_book_row)

                current_fill = fill_price_vals[fill_ind]
                current_normalized_fill = current_fill#/current_bid - 1
                normalized_fills = np.append(normalized_fills, current_normalized_fill)

            current_order_book_row = order_book[order_book.index == (order_book_ind + order_book.index[0])]
            current_order_book_row = current_order_book_row.drop(['ts'], axis=1)

            normalized_order_book_row = current_order_book_row.values #self.normalize_order_book_row(1, current_order_book_row)#current_order_book_row #

            if order_book_ind == 0:
                normalized_order_book = normalized_order_book_row
            else:
                normalized_order_book = np.vstack((normalized_order_book, normalized_order_book_row))

            last_current_bid = current_bid
            last_current_ask = current_ask

        # TODO delete this and formalize
        # with open(test_books, 'wb') as cp_file_handle:
        #     pickle.dump(normalized_order_book, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(test_fills, 'wb') as cp_file_handle:
        #     pickle.dump(normalized_fills, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        return normalized_fills, normalized_order_book

    def format_data_for_training_or_testing(self):
        # output_vec, temp_input_arr = self.normalize_fill_array_and_order_book()
        # TODO delete this and formalize, also uncomment the above
        with open('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/granular_test_books.pickle', 'rb') as ds_file:
            temp_input_arr = pickle.load(ds_file)

        with open('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/granular_test_fills.pickle', 'rb') as ds_file:
            output_vec = pickle.load(ds_file)

        #output_vec = (output_vec - np.min(output_vec))/(np.max(output_vec) - np.min(output_vec))

        output_mask = np.abs(np.diff(output_vec)) > 0.02
        output_mask = np.append(np.array([True]), output_mask)
        temp_input_arr = temp_input_arr#[output_mask, ::]
        output_vec = output_vec#[output_mask]

        scaler = StandardScaler()

        # Create x labels (which are datetime objects)
        x_axis_time_stamps = self.historical_order_books.ts.values[0:len(output_vec)]
        x_labels = np.array([datetime.fromtimestamp(ts) for ts in x_axis_time_stamps])

        # Use standard scalar on input
        temp_input_arr = scaler.fit_transform(temp_input_arr)
        input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], 1, temp_input_arr.shape[1])

        return output_vec, input_arr, x_labels

    def format_data_for_train_test_split(self, train_test_split):
        if (train_test_split >= 1) or (train_test_split <= 0):
            raise ValueError('train_test_split must be in (0, 1)')

        output_vec, input_arr, x_labels = self.format_data_for_training_or_testing()

        training_length = (int(len(input_arr)*(1-train_test_split)))
        training_input_arr = input_arr[0:training_length, ::, ::]
        test_input_arr = input_arr[training_length::, ::, ::]
        training_output_vec = output_vec[0:training_length]
        test_output_vec = output_vec[training_length::]
        x_labels = x_labels[training_length::]

        return training_output_vec, test_output_vec, training_input_arr, test_input_arr, x_labels

    def format_data(self, data_type, train_test_split=0.33):

        if (self.historical_order_books is None):
            raise ValueError('Historical order book data not defined')

        pred_data = {'training output':None, 'training input':None, 'output':None, 'input':None, 'x labels':None}

        data_type = data_type.lower()

        if data_type in ['train', 'test', 'optimize']:
            output_vec, input_arr, x_labels = self.format_data_for_training_or_testing()
            pred_data['input'] = input_arr
            pred_data['output'] = output_vec
            pred_data['x labels'] = x_labels

        elif (data_type == 'train/test') or (data_type == 'test/train'):
            training_output_vec, test_output_vec, training_input_arr, test_input_arr, x_labels = self.format_data_for_train_test_split(train_test_split=train_test_split)
            pred_data['training input'] = training_input_arr
            pred_data['training output'] = training_output_vec
            pred_data['input'] = test_input_arr
            pred_data['output'] = test_output_vec
            pred_data['x labels'] = x_labels

        elif data_type == 'forecast':
            _, temp_input_arr = self.normalize_fill_array_and_order_book()
            input_arr = temp_input_arr.reshape(temp_input_arr.shape[0], 1, temp_input_arr.shape[1])
            pred_data['input'] = input_arr

        return pred_data

class CryptoPriceModel:

    model = None
    data_obj = None
    prediction_length = 30
    optimization_scheme="adam"
    loss_func="mean_absolute_percentage_error"
    bitinfo_list = None

    def __init__(self, date_from, date_to, prediction_ticker, forecast_offset=30, sym_list=None, epochs=500, activ_func='relu', time_units='min', is_leakyrelu=True, suppress_output=False, model_path=None):

        self.date_from = date_from
        self.date_to = date_to
        self.prediction_length = forecast_offset
        self.prediction_ticker = prediction_ticker
        self.time_units = time_units
        self.epochs = epochs
        self.bitinfo_list = sym_list
        self.activation_func = activ_func
        self.is_leakyrelu=is_leakyrelu
        self.suppression = suppress_output

        if model_path is not None:
            self.model = keras.models.load_model(model_path)

    def create_formatted_data_obj(self, save_data=False, data_set_path=None, hourly_time_offset=None):

        if hourly_time_offset is not None:
            # This is for making current predictions
            fmt = '%Y-%m-%d %H:%M:00'
            date_to = datetime.now(get_localzone()).strftime(fmt)
            date_from = (datetime.now(get_localzone()) - timedelta(hours=hourly_time_offset)).strftime(fmt)
            self.date_to = date_to
            self.date_from = date_from
            self.data_obj = FormattedData(date_from, date_to, self.prediction_ticker,
                                          sym_list=self.bitinfo_list, time_units='min', suppression=self.suppression)

            #Shouls Always scrape data for predictions
            data_set_path = None
        else:
            self.data_obj = FormattedData(self.date_from, self.date_to, self.prediction_ticker,
                                          sym_list=self.bitinfo_list, time_units='min', suppression=self.suppression)

        if data_set_path is None:
            self.data_obj.scrape_data()
            self.data_obj.merge_raw_data_frames()
        else:
            with open(data_set_path, 'rb') as ds_file:
                saved_raw_data = pickle.load(ds_file)

            # The below block removes any extra dates from the saved table
            fmt = '%Y-%m-%d %H:%M:%S'
            date_from_object = datetime.strptime(self.date_from, fmt)
            date_to_object = datetime.strptime(self.date_to, fmt)
            dates_list = saved_raw_data.date

            start_ind = (dates_list == date_from_object).argmax()
            stop_ind = (dates_list == date_to_object).argmax() + 1

            saved_raw_data = saved_raw_data[start_ind:stop_ind]
            saved_raw_data.index = np.arange(len(saved_raw_data))

            self.data_obj.raw_data = saved_raw_data

        if save_data:
            self.data_obj.save_raw_data()

    def update_formatted_data(self, date_to=None, save_data=False):
        date_from = self.date_to
        fmt = '%Y-%m-%d %H:%M:00'
        if date_to is None:
            date_to = datetime.now().strftime(fmt)

        fmt = '%Y-%m-%d %H:%M:%S'

        if date_to != date_from:
            data_obj_for_update = FormattedData(date_from, date_to, self.prediction_ticker,
                                 sym_list=self.bitinfo_list, time_units='min', suppression=self.suppression)
            data_obj_for_update.scrape_data()
            data_obj_for_update.merge_raw_data_frames()
            dates_list = self.data_obj.raw_data.date
            date_from_object = datetime.strptime(date_from, fmt)
            date_to_object = datetime.strptime(date_to, fmt)

            if np.sum(dates_list == date_to_object) == 0:
                #This if statement ensures it can't update with old data
                start_ind = len(dates_list) - (dates_list == date_from_object).argmax()
                new_raw_data = data_obj_for_update.raw_data[start_ind::]
                new_raw_data.index = new_raw_data.index + np.max(self.data_obj.raw_data.index.values) + start_ind

                # Update values
                self.data_obj.raw_data = self.data_obj.raw_data.append(new_raw_data)
                self.data_obj.date_to = date_to
                self.date_to = date_to

        if save_data:
            self.data_obj.save_raw_data()

    def custom_loss_func(self, y_true, y_pred):

        all_diffs = K.abs(y_true) - K.abs(y_pred)
        in_range_diffs = all_diffs*K.cast(K.greater_equal(all_diffs, 0), 'float32')
        out_range_diffs = all_diffs*K.cast(K.less(all_diffs, 0), 'float32')

        err = K.abs(K.sum(in_range_diffs) + K.constant(2)*K.sum(out_range_diffs)/K.mean(y_true))

        return err

    def build_model(self, inputs, neurons, output_size=1, dropout=0.25, layer_count=3):
        is_leaky = self.is_leakyrelu
        activ_func = self.activation_func
        loss = self.loss_func
        optimizer = self.optimization_scheme
        self.model = Sequential()

        self.model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        self.model.add(Dropout(dropout))
        self.model.add(Activation('sigmoid'))

        if is_leaky:
            for i in range(0, layer_count):
                self.model.add(Dense(units=neurons, activation="linear", kernel_initializer='normal'))
                self.model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, layer_count):
                self.model.add(Dense(units=neurons, activation=activ_func, kernel_initializer='normal'))

        self.model.add(Dense(units=output_size, activation="linear"))
        self.model.compile(loss=self.loss_func, optimizer=optimizer) #loss, optimizer=optimizer)

    def train_model(self, training_input, training_output, neuron_count=200, save_model=False, train_saved_model=False, layers=3, batch_size=96, training_patience=2):
        if train_saved_model:
            print('re-trianing model')
            self.model.reset_states()
        else:
            self.build_model(training_input, neurons=neuron_count, layer_count=layers)

        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=training_patience, verbose=0, mode='auto')

        hist = self.model.fit(training_input, training_output, epochs=self.epochs,
                              batch_size=batch_size, verbose=2,
                              shuffle=True, validation_split=0.25, callbacks=[estop])

        if self.is_leakyrelu: #TODO add more detail to saves
            file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/' + self.prediction_ticker.upper() + '/' + self.prediction_ticker + 'model_'+ str(layers) + 'layers_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_'+ str(neuron_count) + 'neurons_' + str(np.max(hist.epoch)) +'epochs' + str(datetime.now().timestamp()) + '.h5'

        else:
            file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/' + self.prediction_ticker.upper() + '/' + self.prediction_ticker + 'model_' + str(layers) + 'layers_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(neuron_count) + 'neurons_' + str(np.max(hist.epoch)) +'epochs_' + str(layers) + 'layers' + str(datetime.now().timestamp()) + '.h5'

        if save_model:
            self.model.save(file_name)

        return hist, file_name

    def update_model_training(self, input, output):
        #This is for live model weight updates
        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        self.model.reset_states()
        self.model.fit(input, output, epochs=self.epochs, batch_size=96, verbose=2,
                              shuffle=False, validation_split=0.25, callbacks=[estop])

    def test_model(self, test_input, test_output, show_plots=True, x_indices=None):
        prediction = self.model.predict(test_input)
        prediction = prediction[::, 0] # For some reason the predictions come out 2D (e.g. [[p1,...,pn]] vs [p1,...,pn]]

        if show_plots:
            # Plot the price and the predicted price vs time
            prediction_for_plots = rescale_to_fit(prediction, test_output)
            if x_indices is None:
                plt.plot(prediction_for_plots, 'rx--')
                plt.plot(test_output, 'b.--')
                plt.xlabel('Time (min)')
            else:
                df = pd.DataFrame(data={'Actual': test_output, 'Predicted': prediction_for_plots}, index=x_indices)
                df.Predicted.plot(style='rx--')
                df.Actual.plot(style='b.--')
                plt.xlabel('Date/Time')

            plt.title('Predicted Price and Actual Price')
            plt.ylabel('Price (USD)')

            # Plot the correlation between price and predicted
            plt.figure()
            plt.plot(test_output, prediction_for_plots, 'b.')
            plt.xlabel('measured price')
            plt.ylabel('predicted price')
            plt.title('Correlation Between Predicted and Actual Prices')

            plt.show()


        return {'predicted': prediction, 'actual':test_output}

    def optimize_model(self, training_input, training_output, neuron_grid, layer_grid, batch_size_grid, save_model=True, patience=2):

        hist = np.zeros((len(layer_grid), len(neuron_grid)))
        # The "single" arrays are meant to be used once per run
        single_models = []
        single_val_losses = np.array([])
        single_file_names = []

        for i in range(0, len(layer_grid)):
            for j in range(0, len(neuron_grid)):
                for batch_size in batch_size_grid:
                    layers = layer_grid[i]
                    neuron_count = neuron_grid[j]
                    for k in range(0, 3):
                        current_hist, current_file_name = self.train_model(training_input, training_output, neuron_count=neuron_count, save_model=False, train_saved_model=False, layers=layers, batch_size=batch_size, training_patience=patience)
                        single_models.append(self.model)
                        single_val_losses = np.append(single_val_losses, current_hist.history['val_loss'][-2])
                        single_file_names.append(current_file_name)
                    best_val_loss_ind = np.argmin(single_val_losses)
                    hist[i, j] = (np.min(single_val_losses[best_val_loss_ind]))
                    if save_model:
                        single_models[best_val_loss_ind].save(single_file_names[best_val_loss_ind])

                    # Reset the "single" arrays after each run
                    single_models = []
                    single_val_losses = np.array([])
                    single_file_names = []

        x_axis_labels = [str(x) for x in neuron_grid]
        y_axis_labels = [str(y) for y in layer_grid]

        fig, ax = plt.subplots()
        cax = ax.imshow(hist)
        ax.set_xticks(np.arange(len(neuron_grid)))
        ax.set_yticks(np.arange(len(layer_grid)))
        # label the ticks
        ax.set_xticklabels(x_axis_labels)
        ax.set_yticklabels(y_axis_labels)

        plt.xlabel('Neuron Count')
        plt.ylabel('Layer Count')
        plt.title(self.prediction_ticker.upper() + ' Model Optimization' + '\n From: ' + self.date_from + ' To: ' + self.date_to)
        fig.colorbar(cax, ticks=[np.min(hist), np.max(hist)])

        # Loop over data dimensions and create text annotations.
        for i in range(len(layer_grid)):
            for j in range(len(neuron_grid)):
                text = ax.text(j, i, num2str(hist[i, j], 3),
                               ha="center", va="center", color="w")

        plt.savefig('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/' + self.prediction_ticker.upper() + '/' + self.prediction_ticker.upper() + ' Optimization Grid From: ' + self.date_from + ' To: ' + self.date_to)
        plt.close()

    def model_actions(self, action, train_test_split = 0.33, forecast_offset=30, predicted_quality='high', show_plots=True, neuron_count=92, save_model=False, train_saved_model=False, layers=3, batch_size=96, neuron_grid=None, batch_size_grid=None, layer_grid=None, hourly_time_offset_for_prediction=2, patience=2):

        if neuron_grid is None:
            neuron_grid = [30, 70, 92, 100]

        if layer_grid is None:
            layer_grid = [1, 3, 5, 9]

        if batch_size_grid is None:
            batch_size_grid = [batch_size]

        if action == 'forecast':
            if self.data_obj is None:
                self.create_formatted_data_obj(hourly_time_offset=hourly_time_offset_for_prediction)
            else:
                self.update_formatted_data()
            data = self.data_obj.format_data(action, forecast_offset=forecast_offset,
                                             predicted_quality=predicted_quality)
        else:
            data = self.data_obj.format_data(action, train_test_split=train_test_split, forecast_offset=forecast_offset,
                                         predicted_quality=predicted_quality)

        if action == 'train':
            hist, _ = self.train_model(data['input'], data['output'], neuron_count=neuron_count, save_model=save_model, train_saved_model=train_saved_model, layers=layers, batch_size=batch_size, training_patience=patience)
            return hist

        elif action == 'test':
            test_data = self.test_model(data['input'], data['output'], show_plots=show_plots, x_indices=data['x labels'])
            return test_data

        elif action == 'train/test':
            self.train_model(data['training input'], data['training output'], neuron_count=neuron_count, save_model=save_model, train_saved_model=train_saved_model, layers=layers, batch_size=batch_size, training_patience=patience)

            self.test_model(data['input'], data['output'], show_plots=show_plots, x_indices=data['x labels'])
            return None

        elif action == 'forecast':
            prediction = self.model.predict(data['input'])
            return prediction

        elif action == 'optimize':
            self.optimize_model(data['input'], data['output'], neuron_grid, layer_grid, batch_size_grid, patience=patience)
            return None

class CryptoFillsModel(CryptoPriceModel):

    def __init__(self, prediction_ticker, epochs=500, activ_func='relu', is_leakyrelu=True, suppress_output=False, model_path=None):

        super(CryptoFillsModel, self).__init__('2000-01-01 00:00:00', '2001-01-01 00:00:00', prediction_ticker, 30, None, epochs, activ_func, 'min', is_leakyrelu, suppress_output, model_path)

    def create_formatted_cbpro_data(self, order_book_path=None, fill_path=None):
        self.data_obj = FormattedCoinbaseProData(historical_order_books_path=order_book_path, historical_fills_path=fill_path, suppress_output=self.suppression)

        if order_book_path is not None:
            from_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'  # Format given from coinbase
            to_fmt = '%Y-%m-%d %H:%M:%S'  # Format to save

            # TODO allow a from and to date to in __init__ and filter orderbook based upon that
            # Format in UTC time
            utc = pytz.UTC
            utc_date_from_obj = utc.localize(datetime.strptime(self.data_obj.historical_fills.time.values[0], from_fmt))
            utc_date_to_obj = utc.localize(datetime.strptime(self.data_obj.historical_fills.time.values[-1], from_fmt))

            # Cast to local time zone
            tz = get_localzone()
            date_from_obj = utc_date_from_obj.astimezone(tz)
            date_to_obj = utc_date_to_obj.astimezone(tz)

            # Save as str
            self.date_from = date_from_obj.strftime(to_fmt)
            self.date_to = date_to_obj.strftime(to_fmt)

    def test_model(self, test_input, test_output, show_plots=True, x_indices=None):
        prediction = self.model.predict(test_input)
        prediction = prediction[::, 0] # For some reason the predictions come out 2D (e.g. [[p1,...,pn]] vs [p1,...,pn]]

        if show_plots:
            # Plot the price and the predicted price vs time
            prediction_for_plots = prediction #rescale_to_fit(prediction, test_output) #
            if x_indices is None:
                plt.plot(prediction_for_plots, 'rx--')
                plt.plot(test_output, 'b.--')
                plt.xlabel('Time (min)')
            else:
                df = pd.DataFrame(data={'Actual': test_output, 'Predicted': prediction_for_plots}, index=x_indices)
                df.Predicted.plot(style='rx--')
                df.Actual.plot(style='b.--')
                plt.xlabel('Date/Time')

            plt.title('Predicted Price and Actual Price')
            plt.ylabel('Price (USD)')

            # Plot the correlation between price and predicted
            plt.figure()
            plt.plot(test_output, prediction_for_plots, 'b.')
            plt.xlabel('measured price')
            plt.ylabel('predicted price')
            plt.title('Correlation Between Predicted and Actual Prices')

            plt.show()


        return {'predicted': prediction, 'actual':test_output}

    def model_actions(self, action, train_test_split = 0.33, forecast_offset=30, predicted_quality='high', show_plots=True, neuron_count=92, save_model=False, train_saved_model=False, layers=3, batch_size=96, neuron_grid=None, batch_size_grid=None, layer_grid=None, hourly_time_offset_for_prediction=2, patience=2):

        if neuron_grid is None:
            neuron_grid = [30, 70, 92, 100]

        if layer_grid is None:
            layer_grid = [1, 3, 5, 9]

        if batch_size_grid is None:
            batch_size_grid = [batch_size]

        # TODO add ability to update data for forecasting
        data = self.data_obj.format_data(action, train_test_split=train_test_split)

        if action == 'train':
            hist, _ = self.train_model(data['input'], data['output'], neuron_count=neuron_count, save_model=save_model, train_saved_model=train_saved_model, layers=layers, batch_size=batch_size, training_patience=patience)
            return hist

        elif action == 'test':
            test_data = self.test_model(data['input'], data['output'], show_plots=show_plots, x_indices=data['x labels'])
            return test_data

        elif action == 'train/test':
            self.train_model(data['training input'], data['training output'], neuron_count=neuron_count, save_model=save_model, train_saved_model=train_saved_model, layers=layers, batch_size=batch_size, training_patience=patience)

            self.test_model(data['input'], data['output'], show_plots=show_plots, x_indices=data['x labels'])
            return None

        elif action == 'forecast':
            raw_prediction = self.model.predict(data['input'])
            prediction = raw_prediction[::, 0]
            return prediction

        elif action == 'optimize':
            self.optimize_model(data['input'], data['output'], neuron_grid, layer_grid, batch_size_grid, patience=patience)
            return None

# --Useful Scripts Based on This CryptoPriceModel--

def increase_saved_dataset_length(original_ds_path, sym, date_to=None, forecast_offset=30):
    date_from_search = re.search(r'^.*from_(.*)_to_.*$', original_ds_path).group(1)
    date_from = date_from_search.replace('_', ' ')
    date_from = date_from.replace('EST', '')
    date_to_search = re.search('^.*to_(.*).pickle.*$', original_ds_path).group(1)
    og_to_date = date_to_search.replace('_', ' ')
    og_to_date = og_to_date.replace('EST', '')

    model_obj = CryptoPriceModel(date_from, og_to_date, sym, forecast_offset=forecast_offset)
    model_obj.create_formatted_data_obj(data_set_path=original_ds_path)
    model_obj.update_formatted_data(save_data=True, date_to=date_to)

if __name__ == '__main__':

    obj_type = 'CoinPriceModel'

    # For CoinPriceModel
    if obj_type == 'CoinPriceModel':
        should_use_existing_data_set_path = False
        should_use_existing_model = False
        file_name = None
        model_path = None

        if should_use_existing_data_set_path:
            file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/minbyminETH_from_2018-12-08_22:00:00EST_to_2018-12-15_21:00:00EST.pickle'
        if should_use_existing_model:
            model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/BTC/BTCmodel_3layers_30min_leakyreluact_adamopt_mean_absolute_percentage_errorloss_92neurons_2epochs1545192453.197662.h5'

        date_from = '2021-10-26_20:00:00'.replace('_', ' ')
        date_to = '2021-10-26_09:11:00'.replace('_', ' ')
        sym_list = ['ETH']#['BCH', 'BTC', 'ETC', 'ETH', 'LTC', 'ZRX']

        for sym in sym_list:
            model_obj = CryptoPriceModel(date_from, date_to, sym, forecast_offset=30, model_path=model_path)
            model_obj.create_formatted_data_obj()
            pred = model_obj.model_actions('test')

    elif obj_type == 'CoinFillModel':

        date_from = '2018-12-30_18:24:00'.replace('_', ' ')
        date_to = '2018-12-18_23:00:00'.replace('_', ' ')
        sym_list = ['ETH']#['BCH', 'BTC', 'ETC', 'ETH', 'LTC', 'ZRX']
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/ETH/ETHmodel_1layers_30fill_leakyreluact_adamopt_mean_absolute_percentage_errorloss_60neurons_9epochs1550020276.369253.h5'

        for sym in sym_list:
            # order_books = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/' + sym + '_historical_order_books_granular.csv'
            # fills = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/' + sym + '_fills_granular.csv'
            order_books = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_historical_order_books_granular_031919.csv'
            fills = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_fills_granular_031919.csv'
            model_obj = CryptoFillsModel(sym, model_path=model_path, epochs=5)
            model_obj.create_formatted_cbpro_data(order_book_path=order_books, fill_path=fills)
            pred = model_obj.model_actions('train/test', neuron_count=60, layers=3, batch_size=256, save_model=True, train_test_split=0.1, patience=10)