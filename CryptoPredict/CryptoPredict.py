import requests
from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
import pytz
import dill
import pickle
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from cryptory import Cryptory
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage



def convert_time_to_uct(naive_date_from):
    est = pytz.timezone('America/New_York')
    est_date_from = est.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = est_date_from.astimezone(utc)
    return utc_date

class cryptocompare:

    comparison_symbols = ['USD']
    exchange = ''
    aggregate = 1

    def __init__(self, comparison_symbols=None, exchange=None, date_from=None, date_to=None):

        if comparison_symbols:
            self.comparison_symbols = comparison_symbols

        if exchange:
            self.exchange = exchange

        fmt = '%Y-%m-%d %H:%M:%S %Z'
        naive_date_from = datetime.strptime(date_from, fmt)
        est = pytz.timezone('America/New_York')
        est_date_from = est.localize(naive_date_from)
        utc = pytz.UTC
        self.date_from = est_date_from.astimezone(utc)

        if date_to:
            naive_date_to = datetime.strptime(date_to, fmt)
            est_date_to = est.localize(naive_date_to)

        self.date_to = est_date_to.astimezone(utc)

    def datedelta(self, units):
        d1_ts = time.mktime(self.date_from.timetuple())
        if self.date_to:
            d2_ts = time.mktime(self.date_to.timetuple())
        else:
            d2_ts = time.mktime(datetime.utcnow().timetuple())

        if units == "days":
            del_date = int((d2_ts-d1_ts)/86400)
        elif units == "hours":
            del_date = int((d2_ts - d1_ts) / 3600)
        elif units == "minutes":
            del_date = int((d2_ts - d1_ts) / 60)

        return del_date

    def price(self, symbol='LTC'):
        comparison_symbols = self.comparison_symbols
        exchange = self.exchange

        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}' \
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()
        return data


    def create_data_frame(self, url, symbol='LTC', return_time_stamp=False):
        page = requests.get(url)
        symbol = symbol.upper()
        data = page.json()['Data']
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


    def daily_price_historical(self, symbol='LTC', all_data=True, limit=1):
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

    def hourly_price_historical(self, symbol = 'LTC'):
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

    def minute_price_historical(self, symbol='LTC'): #TODO this method returns one hour less than it should, reason unkown. Fix this
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("minutes")
        first_lim = limit

        if limit > 2001:
            first_lim = 2001

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

        loop_len = int(np.ceil(limit/2001))
        if limit > 2001: #This if statement is to allow the gathering of historical minute data beyond 2000 points (the limit)
            df, time_stamp = self.create_data_frame(url, symbol, return_time_stamp=True)
            for num in range(1, loop_len):
                toTs = time_stamp - 60 # have to subtract a value of 60 had to be added to avoid repeated indices
                url_new = temp_url + '&toTs={}'.format(toTs)
                if num == (loop_len - 1):
                    url_new = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                        .format(symbol.upper(), comparison_symbol.upper(), limit - num*2001, self.aggregate, toTs)
                df_to_append, time_stamp = self.create_data_frame(url_new, symbol, return_time_stamp=True)
                df = df_to_append.append(df, ignore_index=True) #The earliest data goes on top
            return df

        df = self.create_data_frame(url, symbol)
        return df

    def coin_list(self):
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def coin_snapshot_full_by_id(self, symbol='LTC', symbol_id_dict={}):

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

    def news(self, symbol, date_before=None):
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        naive_date_before = datetime.strptime(date_before, fmt)
        est = pytz.timezone('America/New_York')
        est_date_before = est.localize(naive_date_before)
        utc = pytz.UTC
        date_before = est_date_before.astimezone(utc)
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={},Blockchain,Mining,Trading,Market&lTs={}" \
        .format(symbol.upper(), int(date_before.timestamp()))
        page = requests.get(url)
        data = page.json()['Data']
        return data

class DataSet:
    prediction_length=1


    def __init__(self, date_from, date_to, days=None, bitinfo_list = None, prediction_ticker = 'ltc', time_units='hours', fin_table=None, aggregate=1):
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        cryp_obj = cryptocompare(date_from=date_from, date_to=date_to)
        cryp_obj.aggregate = aggregate
        self.cryp_obj = cryp_obj

        temp_fin_table = fin_table
        #This adds the price data from the bitinfo_list currencies to the DataFrame
        sym = bitinfo_list[0]

        if time_units == 'days':
            fin_table = cryp_obj.daily_price_historical(symbol=sym)
            price_func = lambda symbol: cryp_obj.daily_price_historical(symbol=symbol)
        elif time_units == 'hours':
            fin_table = cryp_obj.hourly_price_historical(symbol=sym)
            price_func = lambda symbol: cryp_obj.hourly_price_historical(symbol=symbol)
        else:
            fin_table = cryp_obj.minute_price_historical(symbol=sym)
            price_func = lambda symbol: cryp_obj.minute_price_historical(symbol=symbol)

        self.price_func = price_func

        if temp_fin_table is not None:
            self.fin_table = temp_fin_table
        else:
            for num in range(1,len(bitinfo_list)):
                #cryp_obj.symbol = sym
                sym = bitinfo_list[num]
                temp_table = price_func(symbol=sym)
                temp_table = temp_table.drop(columns='date')
                fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

            #This section adds the news data
            total_len = len(fin_table)
            news_sentiment = []
            news_count = []
            iterations_complete = 0

            rate_limit_url = url = 'https://min-api.cryptocompare.com/stats/rate/limit'
            page = requests.get(rate_limit_url)
            for current_dt in fin_table.date.values:
                current_dt = pd.to_datetime(current_dt)
                current_minute_limit = page.json()['Minute']['CallsLeft']['News']
                current_hour_limit = page.json()['Hour']['CallsLeft']['News']
                if current_minute_limit < 2:
                    time.sleep(60)

                if current_hour_limit < 2:
                    while current_hour_limit < 2:
                        time.sleep(60)
                        current_hour_limit = page.json()['Hour']['CallsLeft']['News']


                current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' EST')
                current_sentiment = [txb(news['title']).sentiment.polarity for news in current_news]

                sentiment_sum = np.sum(current_sentiment)
                news_sentiment.append(sentiment_sum)

                utc_current_dt = convert_time_to_uct(current_dt)
                delta_ts = utc_current_dt.timestamp() - 5 * 3600
                current_news_count = np.sum([news['published_on'] > delta_ts for news in current_news])
                news_count.append(current_news_count)

                iterations_complete += 1
                print('news scraping ' + str(round(100 * iterations_complete / total_len, 2)) + '% complete')

            temp_table = pd.DataFrame({'Sentiment': news_sentiment, 'News Frequency': news_count}, index=fin_table.index)
            fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

            #This section adds the relevat data to the DataSet
            self.fin_table = fin_table
        self.prediction_ticker = prediction_ticker
        self.date_to = date_to
        if days is not None:
            self.prediction_length = days
        self.time_units = time_units

    def create_price_prediction_columns(self):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        temp_prediction_table = self.price_func(symbol=sym)
        prediction_table = temp_prediction_table.drop(columns=['date', sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_high', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])

        fin_table = pd.concat([self.fin_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        data_frame = fin_table.set_index('date')
        self.final_table = data_frame[(data_frame.index <= self.date_to)]

    def create_difference_prediction_columns(self):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        price_table = self.price_func(symbol=sym)
        price_table = price_table.drop(
            columns=['date', sym.upper() + '_low', sym.upper() + '_high',
                     sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])
        close_values = price_table[sym.upper() + '_close'].values
        open_values = price_table[sym.upper() + '_open'].values
        del_values = close_values - open_values
        prediction_table = pd.DataFrame(data=del_values, index=self.fin_table.index)
        fin_table = pd.concat([self.fin_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        data_frame = fin_table.set_index('date')
        self.final_table = data_frame[(data_frame.index <= self.date_to)]

    def get_nth_hr_block(self, test_data, start_time, n=12, time_unit='hours'):
        if time_unit == 'days':
            time_delta = timedelta(days=n)
        elif time_unit == 'hours':
            time_delta = timedelta(hours=n)
        elif time_unit == 'minutes':
            time_delta = timedelta(minutes=n)

        start_date_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S %Z')
        data_block = test_data[
            (test_data.index >= start_date_time) & (test_data.index <= (start_date_time + time_delta))]
        return data_block

    def upper_bound_trade_loop(self, x_point, y_point, m, max_iterations, iterable, delta, upper_bound):
        #This finds the best move available before the upper bound
        sym = self.prediction_ticker
        if x_point > (upper_bound - delta):
            n = 1
            for n in range(m, max_iterations):
                y_point = np.partition(iterable.values.T, n)[::, n][0]
                x_point = iterable.nsmallest(max_iterations - 1, columns=sym.upper() + '_open').index[-1]
                if x_point < (upper_bound - delta):
                    break
            return x_point, y_point, n
        return x_point, y_point, 1

    def lower_bound_trade_loop(self, x_point, y_point, max_iterations, iterable, delta, lower_bound):
        sym = self.prediction_ticker
        if x_point < (lower_bound + delta):
            n = 1
            for n in range(1, max_iterations):
                y_point = np.partition(iterable.values.T, -n)[::, -n][0]
                x_point = iterable.nlargest(max_iterations - 1, columns=sym.upper() + '_open').index[-1]
                if x_point > (lower_bound + delta):
                    break
            return x_point, y_point, n
        return x_point, y_point, 1

    def find_trades(self, data_block, trade_data_frame=None, n=4, max_iterations=5, time_unit='hours'):
        sell_y_point = np.max(data_block.values) #This finds the initial sell points (local max)
        sell_x_point = data_block.idxmax()[0]

        if time_unit == 'days': #TODO replace this recurring if block with a function
            time_delta = pd.Timedelta(days=n)
            buy_sell_delta = pd.Timedelta(days=n)
        elif time_unit == 'hours':
            time_delta = pd.Timedelta(hours=n)
            buy_sell_delta = pd.Timedelta(hours=n)
        else:
            time_delta = pd.Timedelta(minutes=n)
            buy_sell_delta = pd.Timedelta(minutes=n)

        sell_x_point, sell_y_point, n_sell = self.lower_bound_trade_loop(sell_x_point, sell_y_point, len(data_block), data_block,
                                                                        time_delta, data_block.index[0]) #len(data_block was used for the max iterations because the small number of choices led to no trades being found)
        if sell_x_point < (np.min(data_block.index) + time_delta):
            return False, 0, 0

        data_before_sell = data_block[data_block.index < sell_x_point]
        buy_y_point = np.min(data_before_sell.values)
        buy_x_point = data_before_sell.idxmin()[0]

        buy_x_point, buy_y_point, n_buy = self.upper_bound_trade_loop(buy_x_point, buy_y_point, 1, len(data_before_sell),
                                                                      data_before_sell,
                                                                      buy_sell_delta, sell_x_point)
        df = pd.DataFrame([[buy_x_point, buy_y_point, sell_x_point, sell_y_point]],
                          columns=['Buy_X', 'Buy_Y', 'Sell_X', 'Sell_Y'])
        if trade_data_frame is not None:
            final_data_frame = trade_data_frame.append(df, ignore_index=True)
        else:
            final_data_frame = df
        return final_data_frame, buy_y_point, sell_y_point

    def timedelta_strunits(self, n, time_unit):
        if time_unit == 'days':
            time_delta = timedelta(days=n)
        elif time_unit == 'hours':
            time_delta = timedelta(hours=n)
        else:
            time_delta = timedelta(minutes=n)

        return time_delta

    def find_all_trades(self, data_frame, time_unit='hours', n_0=12, m=3, max_iterations=5):

        start_time_str = datetime.strftime(data_frame.index[0], '%Y-%m-%d %H:%M:%S')
        start_time = data_frame.index[0]

        end_time = data_frame.index[-1]
        block_trades = None
        n = n_0
        time_delta_0 = self.timedelta_strunits(n, time_unit)
        search_range = start_time + time_delta_0
        coeff = 1
        while search_range < (end_time - time_delta_0):
            data_frame_block = self.get_nth_hr_block(data_frame, start_time_str + ' EST', n=n, time_unit=time_unit)
            #data_frame_block = data_block.to_frame()
            current_block_trades, current_buy, current_sell = self.find_trades(data_frame_block,
                                                                          trade_data_frame=block_trades, n=coeff*m,
                                                                          max_iterations=coeff*max_iterations,
                                                                          time_unit=time_unit)
            if (current_sell - current_buy) < 0.005 * current_buy or type(current_block_trades) is not pd.DataFrame:
                n = n + n_0
                coeff = int(n/n_0)
                time_delta = self.timedelta_strunits(n, time_unit)
                search_range = search_range + time_delta
            else:
                time_delta = self.timedelta_strunits(n_0, time_unit)
                start_time_str = datetime.strftime(search_range, '%Y-%m-%d %H:%M:%S')
                search_range = search_range + time_delta
                block_trades = current_block_trades
                n = n_0
                coeff = 1

        return block_trades

    def create_single_prediction_column(self, price_data_frame, trade_data_frame, buy_or_sell, time_units='hours'):

        if time_units == 'days':
            time_norm_constant = 24 * 3600
        elif time_units == 'hours':
            time_norm_constant = 3600
        else:
            time_norm_constant = 60

        loop_len = price_data_frame.count()
        trade_ind = 0
        trade_time = trade_data_frame[trade_ind]
        time_arr = None
        price_time = price_data_frame.index[0]
        scale_time = (trade_time - price_time).total_seconds() / time_norm_constant
        full_scale = np.max(price_data_frame.values) - np.min(price_data_frame.values) #This gives the full width scale to determine the size of the jumps

        for price_ind in range(0, loop_len[0]):
            price_time = price_data_frame.index[price_ind]
            if price_time > trade_time:
                trade_ind += 1
                if trade_ind >= trade_data_frame.count():
                    return time_arr
                trade_time = trade_data_frame[trade_ind]
                scale_time = (trade_time - price_time).total_seconds() / time_norm_constant
            if scale_time == 0 & trade_ind > 1:
                return time_arr #TODO fix error that happens when scale_time is 0 because the first trade is at the first index
            #scale the data

            time_to_trade = 0.05*((-1)**buy_or_sell)*trade_ind + ((price_data_frame.values[price_ind]- np.mean(price_data_frame.values))/full_scale) + 1
            #0.1*((-1)**buy_or_sell)*trade_ind +

            # transformation_coeff = ((-1) ** trade_ind) / full_scale
            #transformation_coeff = ((-1) ** trade_ind) * np.pi / (scale_time)
            #unscaled_time_to_trade = (trade_time - price_time).total_seconds() * np.pi / time_norm_constant
            #transformation_coeff = ((-1) ** trade_ind) * np.pi / (scale_time)
            #time_to_trade = np.sin(transformation_coeff*unscaled_time_to_trade) #+ 0.45*(((-1)**trade_ind) > 0)
            #time_to_trade = np.exp(transformation_coeff * unscaled_time_to_trade - (((-1)**trade_ind) > 0))

            if time_arr is None:
                time_arr = np.array([time_to_trade])
            else:
                time_arr = np.vstack((time_arr, np.array([time_to_trade])))
        return time_arr

    def create_buy_sell_prediction_frame(self, n, m, max_iterations):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        price_data_frame = self.price_func(symbol=sym)
        price_data_frame = price_data_frame.drop(
            columns=[sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_high',
                     sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])
        price_data_frame = price_data_frame.set_index('date')
        buy_sell_data_frame = self.find_all_trades(price_data_frame, time_unit=self.time_units, n_0=n, m=m, max_iterations=max_iterations)
        buy_column = self.create_single_prediction_column(price_data_frame, buy_sell_data_frame.Buy_X, 1, self.time_units)
        sell_column = self.create_single_prediction_column(price_data_frame, buy_sell_data_frame.Sell_X, 0, self.time_units)
        cutoff_len = len(buy_column)
        frame_length = range(0, cutoff_len)
        sell_column = sell_column[frame_length]
        final_indxs = price_data_frame.index[frame_length]

        prediction_frame = pd.DataFrame(data=(buy_column + sell_column), index=final_indxs, columns=['Buy and Sell'])
        data_frame = self.fin_table.set_index('date')
        data_frame = data_frame.head(cutoff_len)
        self.final_table = pd.concat([data_frame, prediction_frame], axis=1, join_axes=[prediction_frame.drop_duplicates().index])

    def create_arrays(self, model_type='buy&sell', time_block_length=24, min_distance_between_trades=3):
        if model_type == 'price':
            self.create_price_prediction_columns()
            n = -1
            m = n
        elif model_type == 'difference':
            self.create_difference_prediction_columns()
            n = -1
            m = n
        elif model_type == 'buy&sell':
            self.create_buy_sell_prediction_frame(time_block_length, min_distance_between_trades, 5) #TODO remove max_iterations from find_all trades as new fixes have made it redundant
            n = -1 #This should be -2 for two columns
            m = n #This should be -3 for two columns

        temp_input_table = self.final_table#.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.prediction_length:-1, n:])
        temp_input_array = np.array(temp_array[0:-(self.prediction_length + 1), 0:m])
        scaler = StandardScaler()
        temp_input_array = scaler.fit_transform(temp_input_array)
        self.input_array = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)

    def create_prediction_arrays(self):
        data_frame = self.fin_table.set_index('date')
        temp_input_array = data_frame.values
        scaler = StandardScaler()
        temp_array = temp_input_array[np.logical_not(np.isnan(np.sum(temp_input_array, axis=1))), ::]
        temp_array = scaler.fit_transform(temp_array)
        self.input_array = temp_array.reshape(temp_array.shape[0], temp_array.shape[1], 1)
        self.output_array = data_frame[self.prediction_ticker.upper() + '_open'].values
        self.final_table=data_frame

class CoinPriceModel:

    model = None
    training_array_input = None
    training_array_output = None
    data_obj = None
    prediction_length = 1
    activation_func=None
    optimization_scheme="adam"
    loss_func="mean_absolute_percentage_error"
    epochs = None
    bitinfo_list = None
    google_list=None

    def __init__(self, date_from, date_to, model_path=None, days=None, bitinfo_list=None, google_list=None,
                 prediction_ticker='ltc', epochs=50, activ_func='relu', time_units='hours', is_leakyrelu=False, need_data_obj=True, data_set_path=None, aggregate_val=1):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        if google_list is None:
            google_list = ['Litecoin']
        if days is not None:
            self.prediction_length = days
        self.prediction_ticker = prediction_ticker

        if need_data_obj:
            if data_set_path:
                with open(data_set_path, 'rb') as ds_file:
                    saved_table = pickle.load(ds_file)
            else:
                saved_table=None

            self.data_obj = DataSet(date_from=date_from, date_to=date_to, days=self.prediction_length,
                                    bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, fin_table=saved_table, aggregate=aggregate_val)


        self.epochs = epochs
        self.bitinfo_list = bitinfo_list
        self.google_list = google_list
        self.activation_func = activ_func
        self.is_leakyrelu=is_leakyrelu

    def build_model(self, inputs, neurons, output_size=1,
                    dropout=0.25):  #TODO make output_size someing editable outside the class
        is_leaky = self.is_leakyrelu
        activ_func = self.activation_func
        loss = self.loss_func
        optimizer = self.optimization_scheme
        self.model = Sequential()

        self.model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        self.model.add(Dropout(dropout))

        if is_leaky:
            for i in range(0, 3):
                self.model.add(Dense(units=neurons, activation="linear", kernel_initializer='normal'))
                self.model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, 3):
                self.model.add(Dense(units=neurons, activation=activ_func, kernel_initializer='normal'))

        self.model.add(Dense(units=output_size, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer)

    def create_arrays(self, time_block_length=24, min_distance_between_trades=3, model_type='buy&sell'):
        self.data_obj.create_arrays(model_type, time_block_length, min_distance_between_trades)
        self.training_array_input = self.data_obj.input_array
        self.training_array_output = self.data_obj.output_array

    #TODO create method to save Data Frames without testing the models

    def train_model(self, neuron_count=200, time_block_length=24, min_distance_between_trades=3, model_type='buy&sell', save_model=False):
        self.create_arrays(time_block_length, min_distance_between_trades, model_type=model_type)
        if model_type == 'buy&sell':
            self.build_model(self.training_array_input, neurons=neuron_count, output_size=1)
        else:
            self.build_model(self.training_array_input, neurons=neuron_count)

        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        hist = self.model.fit(self.training_array_input, self.training_array_output, epochs=self.epochs, batch_size=96, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])

        if self.is_leakyrelu & save_model: #TODO add more detail to saves
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/' + self.prediction_ticker + 'model_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
                np.max(hist.epoch)) + 'epochs_' + str(neuron_count) + 'neuron' + str(datetime.utcnow().timestamp()) + '.h5')
        elif save_model:
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/' + self.prediction_ticker + 'model_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
                np.max(hist.epoch)) + 'epochs_' + str(neuron_count) + 'neurons' + str(datetime.utcnow().timestamp()) + '.h5')

        return hist

    def test_model(self, from_date, to_date, time_units='hours', model_type='price'):
        test_data = DataSet(date_from=from_date, date_to=to_date, days=self.prediction_length, bitinfo_list=self.bitinfo_list,
                            prediction_ticker=self.prediction_ticker, time_units=time_units)
        test_data.create_arrays(model_type=model_type)
        test_input = test_data.input_array
        test_output = test_data.output_array
        prediction = self.model.predict(test_input)


        zerod_output = test_output[::, 0] - np.mean(test_output[::, 0])
        zerod_prediction = prediction[::, 0] - np.mean(prediction[::, 0])
        #TODO uncomment plots
        #plt.plot(zerod_output/(np.max(zerod_output)), 'bo--')
        #plt.plot(zerod_prediction/(np.max(zerod_prediction)), 'rx--')
        #plt.title('Prediction')
        #plt.show()

    def create_standard_dates(self):
        utc_to_date = datetime.utcnow()
        utc = pytz.UTC
        est = pytz.timezone('America/New_York')
        utc_to_date = utc.localize(utc_to_date)
        to_date = utc_to_date.astimezone(est)
        return to_date

    def predict(self, time_units='hours', show_plots=True):
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        to_date = self.create_standard_dates()

        if time_units == 'minutes':
            delta = timedelta(minutes=self.prediction_length)
            from_delta = timedelta(hours=1)
        elif time_units == 'hours':
            delta = timedelta(hours=self.prediction_length)
            from_delta = timedelta(days=3)
        elif time_units == 'days':
            delta = timedelta(days=self.prediction_length)
            from_delta = timedelta(days=30)

        from_date = to_date - from_delta
        test_data = DataSet(date_from=from_date.strftime(fmt), date_to=to_date.strftime(fmt), days=self.prediction_length, bitinfo_list=self.bitinfo_list,
                            prediction_ticker=self.prediction_ticker, time_units=time_units)
        test_data.create_prediction_arrays()
        prediction_input = test_data.input_array #do not use the create array methods here because the output is not needed
        prediction = self.model.predict(prediction_input)
        price_array = test_data.output_array

        zerod_price = price_array - np.min(price_array)
        scaled_price = zerod_price/np.max(zerod_price)
        scaled_prediction = (prediction[::,0] - np.min(prediction))/np.max(prediction - np.min(prediction))
        zerod_prediction = scaled_prediction + scaled_price[-1] - scaled_prediction[-self.prediction_length-1]
        columstr = 'Predicted ' + time_units
        prediction_table = pd.DataFrame({columstr: zerod_prediction}, index=test_data.final_table.index + delta)
        price_table = pd.DataFrame({'Current': scaled_price},index=test_data.final_table.index)

        if show_plots:
            ax1 = prediction_table.plot(style='rx--')
            price_table.plot(style='bo--', ax=ax1)
            plt.title(self.prediction_ticker.upper() + ' ' + str(self.prediction_length) + ' ' + time_units + ' Prediction')
            plt.show()
        else:
            return prediction_table, price_table

class BaseTradingBot:

    image_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Images//'
    last_true_countdown = 6

    def __init__(self, hourly_model, minute_model, hourly_len=6, minute_len=15, prediction_ticker='ETH'):

        temp = "2018-05-05 00:00:00 EST"
        self.hourly_cp = CoinPriceModel(temp, temp, days=hourly_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='hours', model_path=hourly_model, need_data_obj=False)

        self.minute_cp = CoinPriceModel(date_from, date_to, days=minute_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=minute_model, need_data_obj=False)

        self.hour_length = hourly_len
        self.minute_length = minute_len
        self.prediction_ticker = prediction_ticker

    def find_data(self):

        t_0 = datetime.utcnow().timestamp()
        full_minute_prediction, full_minute_price = self.minute_cp.predict(time_units='minutes', show_plots=False)
        self.minute_prediction = full_minute_prediction[-(self.minute_length + 10)::]
        self.minute_price = full_minute_price[-10::]
        t_f = datetime.utcnow().timestamp()

        full_hourly_prediction, full_hourly_price = self.hourly_cp.predict(time_units='hours', show_plots=False)
        self.hourly_prediction = full_hourly_prediction[-(self.hour_length + 4)::]
        self.hourly_price = full_hourly_price[-4::]

    def prepare_images(self):
        #This method creates, saves, and closes the figures
        #Create minute by minute image
        minute_ax = self.minute_prediction.plot(style='rx--')
        self.minute_price.plot(style='bo--', ax=minute_ax)
        minute_fig_title = self.prediction_ticker.upper() + ' ' + str(self.minute_length) + 'min' + ' Prediction'
        plt.title(minute_fig_title)

        self.current_minute_fig_filename = self.image_path + str(datetime.utcnow().timestamp()) + minute_fig_title.replace(' ', '') + '.png'
        plt.savefig(self.current_minute_fig_filename)

        #Create hourly
        hourly_ax = self.hourly_prediction.plot(style='rx--')
        self.hourly_price.plot(style='bo--', ax=hourly_ax)
        hourly_fig_title = self.prediction_ticker.upper() + ' ' + str(self.hour_length) + 'hrs' + ' Prediction'
        plt.title(hourly_fig_title)

        self.current_hourly_fig_filename = self.image_path + str(datetime.utcnow().timestamp()) + hourly_fig_title.replace(' ', '') + '.png'
        plt.savefig(self.current_hourly_fig_filename)

        plt.close('all')

    def send_data(self):
        #Create the message

        altMsgText = MIMEText('Graphs not displaying properly!' + '\n' + str(round(self.minute_prediction, 3)) + '\n' + str(round(self.hourly_prediction, 3)))

        # Create the root message and fill in the from, to, and subject headers
        msg_root = MIMEMultipart('related')
        msg_root['From'] = 'rogeh2@gmail.com'
        msg_root['To'] = 'rogeh2@gmail.com'
        msg_root['Subject'] = 'Ethereum Prediction From Your Digital Broker'

        # Encapsulate the plain and HTML versions of the message body in an
        # 'alternative' part, so message agents can decide which they want to display.
        msg_alternative = MIMEMultipart('alternative')
        msg_root.attach(msg_alternative)
        msg_alternative.attach(altMsgText)

        # We reference the image in the IMG SRC attribute by the ID we give it below
        msg_text = MIMEText('Crypto currency prediction plots below.<br><img src="cid:imagem"><br><br><img src="cid:imageh"><br>', 'html')
        msg_alternative.attach(msg_text)

        self.prepare_images()

        minute_fp = open(self.current_minute_fig_filename, 'rb')
        minute_image = MIMEImage(minute_fp.read())
        minute_fp.close()

        hourly_fp = open(self.current_hourly_fig_filename, 'rb')
        hourly_image = MIMEImage(hourly_fp.read())
        hourly_fp.close()


        # Define the image's ID as referenced above
        minute_image.add_header('Content-ID', '<imagem>')
        msg_root.attach(minute_image)
        hourly_image.add_header('Content-ID', '<imageh>')
        msg_root.attach(hourly_image)

        # Setup the SMTP server
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('rogeh2@gmail.com', 'Neutrino#0')
        s.sendmail('rogeh2@gmail.com', ['rogeh2@gmail.com'], msg_root.as_string())

    def trade_logic(self, last_bool):
        current_price = self.hourly_prediction.values[-(self.hour_length + 1)]
        next_price = self.hourly_prediction.values[-self.hour_length]
        check_price = self.hourly_prediction.values[-(self.hour_length - 1)]
        val_price = self.hourly_prediction.values[-(self.hour_length - 2)]

        #This tells the bot to send an email if the next predicted price is an inflection point for the next two hours
        if (next_price > current_price) & (next_price > check_price) & (next_price > (val_price + 0.01)):
            self.last_true_countdown = 6
            return True
        elif (next_price < current_price) & (next_price < check_price) & (next_price < (val_price - 0.01)):
            self.last_true_countdown = 6
            return True
        else:
            #If the next price is no longer an inflection point but the last one was keep sending until this hour passes
            if (self.last_true_countdown > 0) & last_bool:
                self.last_true_countdown -= 1
                return True

            return False


    def continuous_monitoring(self):
        current_time = datetime.utcnow().timestamp()
        last_check = 0
        cutoff_time = current_time + 14*3600
        should_send_email = False

        while current_time < cutoff_time:
            if current_time > (last_check + 10*60):
                self.find_data()
                last_check = current_time
                should_send_email = self.trade_logic(should_send_email)
                if should_send_email:
                    self.send_data()
            else:
                time.sleep(1)
            current_time = datetime.utcnow().timestamp()


# TODO try using a classifier Neural Net
# TODO eliminate unnecessary legacy variables from run_neural_net and CryptoPredict

def run_neural_net(date_from, date_to, test_date_from, test_date_to, prediction_length, epochs, prediction_ticker, bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, time_block_length, min_distance_between_trades, model_path, model_type='price', use_type='test', data_set_path=None, save_test_model=True):

    #This creates a CoinPriceModel and saves the data
    if (data_set_path is not None) & (use_type != 'predict'):
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                        time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu, data_set_path=data_set_path)

    elif use_type != 'predict':
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                            time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu)

        table_file_name = '_' + time_unit + '_' + model_type + '_from_' + date_from + '_to_' + date_to + '.pickle'
        table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet' + table_file_name.replace(' ', '_')
        with open(table_file_name, 'wb') as cp_file_handle:
            pickle.dump(cp.data_obj.fin_table, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 'test' will train a model with given conditions then test it, 'optimize' optimizes neuron count by evaluation data loss, 'predict' predicts data
    if use_type == 'test':
        cp.train_model(neuron_count=neuron_count, time_block_length=time_block_length, min_distance_between_trades=min_distance_between_trades, model_type=model_type, save_model=save_test_model)
        cp.test_model(from_date=test_date_from, to_date=test_date_to, time_units=time_unit,
                      model_type=model_type)

    elif use_type == 'optimize':
        hist = []
        neuron_grid = [2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
        for neuron_count in neuron_grid:
            current_hist = cp.train_model(neuron_count=neuron_count, time_block_length=time_block_length,
                                          min_distance_between_trades=min_distance_between_trades, model_type='price', save_model=True)
            hist.append(current_hist.history['val_loss'][-1])

        plt.plot(neuron_grid, hist, 'bo--')
        plt.show()

    elif use_type == 'predict':
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, prediction_ticker=prediction_ticker,
                            bitinfo_list=bitinfo_list, time_units=time_unit,
                            model_path=model_path,
                            need_data_obj=False)
        cp.predict(time_units=time_unit, show_plots=True)


if __name__ == '__main__':

    date_from = "2018-05-11 18:30:00 EST"
    date_to = "2018-05-15 06:30:00 EST"
    test_date_from = "2018-05-15 06:31:00 EST"
    test_date_to = "2018-05-15 06:50:00 EST"
    prediction_length = 15
    epochs = 5000
    prediction_ticker = 'ETH'
    bitinfo_list = ['eth']
    time_unit = 'minutes'
    activ_func = 'relu'
    isleakyrelu = True
    neuron_count = 200
    time_block_length = 60
    min_distance_between_trades = 5
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_4epochs_200neuron1526404187.251125.h5'
    model_type = 'price' #Don't change this
    use_type = 'predict' #valid options are 'test', 'optimize', 'predict'. See run_neural_net for description
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_price_from_2018-05-11_18:30:00_EST_to_2018-05-15_06:30:00_EST.pickle'
    test_model_save_bool = False

    #run_neural_net(date_from, date_to, test_date_from, test_date_to, prediction_length, epochs, prediction_ticker,
    #              bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, time_block_length,
    #               min_distance_between_trades, model_path, model_type, use_type, data_set_path=pickle_path, save_test_model=test_model_save_bool)

    hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_125epochs_50neuron1526405263.252535.h5'
    minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_4epochs_200neuron1526404187.251125.h5'

    naive_bot = BaseTradingBot(hourly_model=hour_path, minute_model=minute_path)
    naive_bot.continuous_monitoring()

    #The below code would make a great unit test
    # fake_prediction = pd.DataFrame({'Test':np.array([0.1, 0.2, 0.1, 0, 0.1, 0.1, 0.1])})
    # naive_bot.hourly_prediction = fake_prediction
    # ans = naive_bot.trade_logic(True)
    # print(str(ans))
    # fake_prediction = pd.DataFrame({'Test': np.array([0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1])})
    # naive_bot.hourly_prediction = fake_prediction
    # ans = naive_bot.trade_logic(True)
    # print(str(ans))