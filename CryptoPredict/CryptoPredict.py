import requests
from requests.adapters import HTTPAdapter

from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
import pytz
import pickle
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import backend as K
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

class CryptoCompare:

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
                    url_new = 'https://min-api.CryptoCompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
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

    def coin_snapshot_full_by_id(self, symbol='LTC', symbol_id_dict={}):#TODO fix the thid argument mutability

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


    def __init__(self, date_from, date_to, days=None, bitinfo_list = None, prediction_ticker = 'ltc', time_units='hours', fin_table=None, aggregate=1, news_hourly_offset=5):
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        cryp_obj = CryptoCompare(date_from=date_from, date_to=date_to)
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

        self.price_func = price_func #TODO eliminate  need for price_func when a saved pickle table is used. Price_func depends on the url which won't work for minutes after 7 days

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

            rate_limit_url = 'https://min-api.cryptocompare.com/stats/rate/limit'
            page = requests.get(rate_limit_url)
            date_len = len(fin_table.date.values)
            last_news = None

            for i in range(1, date_len + 1):
                ind = date_len - i
                current_dt = fin_table.date.values[ind]
                current_dt = pd.to_datetime(current_dt)
                utc_current_dt = convert_time_to_uct(current_dt)
                delta_ts = utc_current_dt.timestamp() - news_hourly_offset * 3600

                current_minute_limit = page.json()['Minute']['CallsLeft']['News']
                current_hour_limit = page.json()['Hour']['CallsLeft']['News']
                if current_minute_limit < 2:
                    time.sleep(60)
                if current_hour_limit < 2:
                    while current_hour_limit < 2:
                        time.sleep(60)
                        current_hour_limit = page.json()['Hour']['CallsLeft']['News']

                if last_news is not None:
                    last_news_publication_times = [news['published_on'] < utc_current_dt.timestamp() for news in last_news]
                    if all(last_news_publication_times):
                        current_news = last_news
                    else:
                        current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' EST')
                else:
                    current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' EST')

                last_news = current_news


                current_full_sentiment = [txb(news['title']).sentiment.polarity for news in current_news]

                current_news_count = np.sum([news['published_on'] > delta_ts for news in current_news])
                news_count.insert(0, current_news_count)

                current_sentiment = current_full_sentiment[0:current_news_count]

                sentiment_sum = np.mean(current_sentiment)
                news_sentiment.insert(0, sentiment_sum)

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
        prediction_table = temp_prediction_table.drop(columns=['date', sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_open', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])

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

    def find_trade_strategy_value(self, buy_bool, sell_bool, all_prices):
        #This finds how much money was gained from a starting value of $100 given a particular strategy
        usd_available = 100
        all_buys = all_prices[buy_bool]
        all_sells = all_prices[sell_bool]
        for i in range(0,len(all_buys)):
            eth_available = usd_available/all_buys[i]
            usd_available = all_sells[i]*eth_available

        value = usd_available - 100

        return value

    def create_single_prediction_column(self, price_data_frame, n, show_plots=False):
        #This finds the optimal trading strategy
        all_times = price_data_frame.index
        all_prices = price_data_frame.values
        eps = 1000.0 #Needs to be float
        iter = 0 #Counts number of iterations in loop
        sell_arr = np.zeros((len(price_data_frame), 1))
        buy_arr = np.zeros((len(price_data_frame), 1))
        val = 0 #This is the money gained by the chosen strategy
        should_sell = True #If true the script should look for max, if False it looks for min
        has_checked_block_counter = n
        last_sell = 0
        target_eps = 0.03 #0.03 chosen as the cutoff for eps because that is the highest GDAX fee and because it seems like a reasonable number to get from one buy-sell cycle due to past experience
        from_n = len(all_times) - n
        val_arr = np.array([])
        strategy_dict = {}

        #This loop keeps making new trade strategies until the change in money earned is under eps
        while (eps > target_eps) & (iter < 1000):
            if iter > 0:
                #calculate new final sell
                ind = np.argwhere(1==buy_arr)[-1][0]
                s_ind = np.argwhere(1==sell_arr)[-2][0]
                current_price_block = all_prices[ind::]
                last_sell = np.max(current_price_block)
                sell_arr = np.zeros((len(price_data_frame), 1))
                from_n = np.argmax(current_price_block) + ind
                sell_arr[from_n] = 1

                #calculate new final sell
                buy_arr = np.zeros((len(price_data_frame), 1))
                current_price_block = all_prices[s_ind:from_n]
                from_n = np.argmin(current_price_block) + s_ind
                buy_arr[from_n] = 1
                should_sell = True


            for i in range(len(all_times) - from_n, len(all_times) - 2*n):
                if has_checked_block_counter > 0:
                    has_checked_block_counter -= 1
                    continue

                ind = len(all_times) - i #To optimize we need to start at the end and go back
                current_price_block = all_prices[(ind-n):ind]
                next_price_block = all_prices[(ind-2*n):(ind-n)]

                # del_percent_check = (np.max(current_price_block) - np.min(current_price_block))/np.max(current_price_block) #This number needs to be less than 0.03 for this to be considered low volatility enough to trade
                #
                # if del_percent_check > 2*target_eps: #From past trades it has been determined that 0.03 is a reasonable number from one trade
                #     continue

                if should_sell: #This says that the bot should sell at the max and buy at the min
                    current_sell = np.max(current_price_block)
                    next_sell = np.max(next_price_block)

                    if (current_sell - next_sell) > 2*target_eps:
                        sell_arr[np.argmax(current_price_block) + (ind-n)] = 1
                        last_sell = np.max(current_price_block)
                        should_sell = False
                        has_checked_block_counter = n
                else:
                    buy_price = np.min(current_price_block)
                    next_buy = np.min(next_price_block)
                    if ((last_sell - buy_price) > 2*target_eps) & ((next_buy - buy_price) > 2*target_eps):
                        should_sell = True
                        buy_arr[np.argmin(current_price_block) + (ind-n)] = 1 #These statements do not have off by one errors because python indexes from 0
                        has_checked_block_counter = n

            sell_arr[0:np.argmax(buy_arr)] = 0 #This always starts with a buy and ends with a sell

            sell_bool = [x[0] == 1 for x in sell_arr] #must add [0] or else each x will be a seperate array and unindexable
            buy_bool = [x[0] == 1 for x in buy_arr]
            val_new = self.find_trade_strategy_value(buy_bool, sell_bool, all_prices) #This calculates the money earned
            val_arr = np.hstack((val_arr, val_new))
            strategy_dict[iter] = {'buy':buy_arr, 'sell':sell_arr}
            eps = np.abs(val_new - val)
            print('on iteration ' + str(iter) + ' with trade value of ' + str(val_new))
            val = val_new
            iter += 1

        best_strategy_ind = val_arr.argmax()
        fin_buy_arr = strategy_dict[best_strategy_ind]['buy']
        fin_sell_arr = strategy_dict[best_strategy_ind]['sell']

        if show_plots:
            #set show_plots to true for debug only
            #TODO make plot for all the data show minutes
            plt.plot(all_times[sell_bool], all_prices[sell_bool], 'rx')
            plt.plot(all_times[buy_bool], all_prices[buy_bool], 'gx')
            plt.plot(all_times, all_prices, 'b--')
            plt.show()
        else:
            return fin_sell_arr, fin_buy_arr

    def create_buy_sell_prediction_frame(self, m):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        price_data_frame = self.price_func(symbol=sym)
        price_data_frame = price_data_frame.drop(
            columns=[sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_open',
                     sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])
        price_data_frame = price_data_frame.set_index('date')

        sell_column, buy_column = self.create_single_prediction_column(price_data_frame, m)

        #This for loop spreads out decisions so that trades that are coming within five minutes are seen
        # m = 5
        # for i in range(0, len(sell_column) - m):
        #     if np.sum(sell_column[i:(m+i)]) == 1:
        #         sell_column[i] = 1
        #
        #     if np.sum(buy_column[i:(m+i)]) == 1:
        #         buy_column[i] = 1


        prediction_frame = pd.DataFrame(data=np.hstack((buy_column, sell_column)), index=price_data_frame.index, columns=['Buy', 'Sell'])

        #TODO check on this
        self.create_price_prediction_columns()

        self.final_table = pd.concat([self.final_table, prediction_frame], axis=1, join_axes=[prediction_frame.index])

    def create_arrays(self, model_type='price', min_distance_between_trades=5):
        if model_type == 'price':
            self.create_price_prediction_columns()
            n = -1
            m = n
        elif model_type == 'difference':
            self.create_difference_prediction_columns()
            n = -1
            m = n
        elif model_type == 'buy&sell':
            self.create_buy_sell_prediction_frame(min_distance_between_trades)
            n = -1 #This should be -2 for two columns
            m = n #This should be -3 for two columns

        temp_input_table = self.final_table#.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.prediction_length:-1, n:])
        temp_input_array = np.array(temp_array[0:-(self.prediction_length + 1), 0:m])
        self.input_array_times = temp_input_table.index[0:-(self.prediction_length + 1)]
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
        self.output_array = data_frame[self.prediction_ticker.upper() + '_high'].values
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
                 prediction_ticker='ltc', epochs=500, activ_func='relu', time_units='hours', is_leakyrelu=True, need_data_obj=True, data_set_path=None, aggregate_val=1):
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
            if saved_table is None:
                table_file_name = '_' + time_units + '_from_' + date_from + '_to_' + date_to + '.pickle'
                table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet' + table_file_name.replace(
                    ' ', '_')
                with open(table_file_name, 'wb') as cp_file_handle:
                    pickle.dump(self.data_obj.fin_table, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)


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

    def create_arrays(self, min_distance_between_trades=5, model_type='price'):
        self.data_obj.create_arrays(model_type, min_distance_between_trades)
        training_length = (int(2*len(self.data_obj.input_array)/3))

        self.input = self.data_obj.input_array
        self.output = self.data_obj.output_array
        self.times = self.data_obj.input_array_times

        self.training_array_input = self.data_obj.input_array[0:training_length]
        self.training_array_output = self.data_obj.output_array[0:training_length]
        self.training_times = self.data_obj.input_array_times[0:training_length]

        self.test_array_input = self.data_obj.input_array[training_length::]
        self.test_array_output = self.data_obj.output_array[training_length::]
        self.test_times = self.data_obj.input_array_times[training_length::]

    #TODO create method to save Data Frames without testing the models

    def train_model(self, neuron_count=200, min_distance_between_trades=5, model_type='price', save_model=False):
        self.create_arrays(min_distance_between_trades, model_type=model_type)
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

    def test_model(self, did_train=True, show_plots=True, min_distance_between_trades=5, model_type='price'):
        if did_train:
            test_input = self.test_array_input
            test_output = self.test_array_output
        else:
            self.create_arrays(min_distance_between_trades, model_type=model_type)
            test_input = self.input
            test_output = self.output


        prediction = self.model.predict(test_input)

        zerod_output = test_output[::, 0] - np.mean(test_output[::, 0])
        zerod_prediction = prediction[::, 0] - np.mean(prediction[::, 0])

        if show_plots:
            plt.plot(zerod_output/(np.max(zerod_output)), 'bo--')
            plt.plot(zerod_prediction/(np.max(zerod_prediction)), 'rx--')
            plt.title('Prediction')
            plt.show()
        else:
            return prediction, test_output

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

class CryptoTradeStrategyModel(CoinPriceModel):
    #TODO add all data to the strategy fin table and have one column for prediction and one for price (instead of n), because this is an LSTM

    strategy_loss_fun = 'binary_crossentropy'

    strategy_is_leakyrelu = True

    #TODO move the create methods to the DataSet class

    def create_test_price_columns(self, should_train=True, min_distance_between_trades=5, n=10):
        #TODO delete extraneous code
        #This creates a table with 2*n columns that contains n columns of the price for the past n units of time and prediction for the next n units. This is meant to train the strategy model

        if should_train:
            val_loss = 30
            iter = 0
            while (val_loss > 10) & (iter < 20):
                hist = self.train_model(model_type='price', neuron_count=30)
                val_loss = hist.history['val_loss'][-1]
                iter += 1
                if val_loss > 10:
                    print('val_loss is too great, retraining')
                else:
                    print('valid model detected')

            prediction, price = self.test_model(show_plots=False)
        else:
            self.create_arrays(min_distance_between_trades, model_type='price')
            prediction, price = self.test_model(did_train=False, show_plots=False, min_distance_between_trades=min_distance_between_trades, model_type='price')

        all_times = self.test_times
        column_len = len(prediction) - 2*n
        prediction_columns = np.zeros((column_len, n))
        time_columns = []

        prediction_data = prediction.T #This makes it easier to add data to columns

        for ind in range(0, column_len):
            prediction_columns[ind, ::] = prediction_data[0, (ind + n):(ind + 2*n)]
            time_columns.insert(0, all_times[ind+n])

        return prediction, all_times #TODO add ability to return dates as well remember that each prediction is at ind+n

    def create_strategy_prediction_frame(self, n, min_distance_between_trades=5, show_plots=False): #Set show_plots to True for debug only
        predicted_price, price_time = self.create_test_price_columns(n=n)
        self.data_obj.create_buy_sell_prediction_frame(min_distance_between_trades)
        strategy_input_frame = pd.DataFrame(data=predicted_price, index=price_time)

        sym = self.data_obj.prediction_ticker.upper()
        buy_frame = self.data_obj.final_table.drop(columns=['Sell'])
        sell_frame = self.data_obj.final_table.drop(columns=['Buy'])

        #TODO delete extraneous code
        buy_strategy_frame = pd.merge(strategy_input_frame, buy_frame, left_index=True, right_index=True)
        sell_strategy_frame = pd.merge( strategy_input_frame, sell_frame, left_index=True, right_index=True)

        #self.strategy_frame = pd.concat([strategy_input_frame, buy_frame, sell_frame], axis=1, join_axes=[strategy_input_frame.index])

        if show_plots:
            ax1 = sell_strategy_frame[self.prediction_ticker.upper() + '_high'].plot(style='b--')
            buy_strategy_frame[self.prediction_ticker.upper() + '_high'][buy_strategy_frame['Buy'].values == 1].plot(style='gx', ax=ax1)
            sell_strategy_frame[self.prediction_ticker.upper() + '_high'][sell_strategy_frame['Sell'].values == 1].plot(style='rx', ax=ax1)
            plt.show()
        else:
            return buy_frame, sell_frame

    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        loss_val = possible_positives/(true_positives + K.epsilon())
        return loss_val

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.clip((1 - y_true) * (1 - y_pred), 0, 1))
        possible_negatives = K.sum(K.clip(1 - y_true, 0, 1))
        loss_val = possible_negatives / (true_negatives + K.epsilon())

        return loss_val

    def custom_loss_func(self, y_true, y_pred):
        loss_val = self.specificity(y_true, y_pred) + self.sensitivity(y_true, y_pred)
        return loss_val

    #TODO make model methods more modular for simpler integration of new models (like the strategy model)
    def build_strategy_model(self, inputs, neurons, strategy_activ_func = 'relu', output_size=1, dropout=0.5, layers=3):
        is_leaky = self.strategy_is_leakyrelu
        activ_func = strategy_activ_func
        loss = self.strategy_loss_fun
        optimizer = self.optimization_scheme
        strategy_model = Sequential()

        strategy_model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        strategy_model.add(Dropout(dropout))

        if is_leaky:
            for i in range(0, layers):
                strategy_model.add(Dense(units=neurons, activation="linear", kernel_initializer='normal'))
                strategy_model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, layers):
                strategy_model.add(Dense(units=neurons, activation=activ_func, kernel_initializer='normal'))

        strategy_model.add(Dense(units=output_size, activation="sigmoid"))
        strategy_model.compile(loss=self.custom_loss_func, optimizer=keras.optimizers.adam(lr=0.0001))

        return strategy_model

    def prep_arrays_for_model(self, arr, training_len, should_reshape):

        if should_reshape:
            scaler = StandardScaler()
            temp_input_array = scaler.fit_transform(arr)
            output_arr = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)
        else:
            output_arr = arr

        training_output = output_arr[0:training_len]
        test_output = output_arr[training_len::]

        return training_output, test_output

    def train_strategy_model(self, neuron_count=30, min_distance_between_trades=5, save_model=False, t = 0.9, layers=1):
        buy_frame, sell_frame = self.create_strategy_prediction_frame(min_distance_between_trades=min_distance_between_trades, n=10)

        buy_values = buy_frame['Buy'].values
        sell_values = sell_frame['Sell'].values

        n_trades = np.sum(buy_values)
        n_holds = len(buy_values) - n_trades
        class_weight = {
            1: (n_holds/n_trades)*t,
            0: (n_holds/n_trades)*(1 - t)
        }#This weights the trades because they are few relative to the holds

        input_values = buy_frame.drop(columns=['Buy'])

        training_length = (int(2 * len(buy_values) / 3))

        training_input, test_input = self.prep_arrays_for_model(input_values, training_length, should_reshape=True)
        training_buy_output, test_buy_output = self.prep_arrays_for_model(sell_values, training_length, should_reshape=False)
        training_sell_output, test_sell_output = self.prep_arrays_for_model(buy_values, training_length,should_reshape=False)



        self.buy_model = self.build_strategy_model(training_input, neurons=neuron_count, layers=layers)
        self.sell_model = self.build_strategy_model(training_input, neurons=neuron_count, layers=layers)

        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        self.buy_model.fit(training_input, training_buy_output, epochs=1000, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop], class_weight=class_weight)

        self.sell_model.fit(training_input, training_sell_output, epochs=1000, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop], class_weight=class_weight)

        return test_buy_output, test_input, test_sell_output

    def test_strategy_model(self, did_train=True, show_plots=True, min_distance_between_trades=5):
        # if did_train:
        #     test_input = self.test_array_input
        #     test_output = self.test_array_output
        # else:
        #     self.create_arrays(min_distance_between_trades, model_type=model_type)
        #     test_input = self.input
        #     test_output = self.output

        test_buy_output, test_input, test_sell_output = self.train_strategy_model()
        buy_prediction = self.buy_model.predict(test_input)
        sell_prediction = self.sell_model.predict(test_input)

        if show_plots:
            plt.plot(test_buy_output, 'bo--')
            plt.plot(buy_prediction, 'rx--')
            plt.title('Buy Predictions')

            plt.figure()

            plt.plot(test_sell_output, 'bo--')
            plt.plot(sell_prediction, 'rx--')
            plt.title('Sell Predictions')

            plt.show()
        else:
            return buy_prediction, sell_prediction


class BaseTradingBot:

    #TODO create method to train and optimize models (including during continuous usage)
    image_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Images//'
    last_true_countdown = 2

    def __init__(self, hourly_model, minute_model, hourly_len=6, minute_len=15, prediction_ticker='ETH', bitinfo_list = ['eth']):
        temp = "2018-05-05 00:00:00 EST"
        self.hourly_cp = CoinPriceModel(temp, temp, days=hourly_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='hours', model_path=hourly_model, need_data_obj=False)

        self.minute_cp = CoinPriceModel(temp, temp, days=minute_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=minute_model, need_data_obj=False)

        self.hour_length = hourly_len
        self.minute_length = minute_len
        self.prediction_ticker = prediction_ticker

    def find_data(self):

        full_minute_prediction, full_minute_price = self.minute_cp.predict(time_units='minutes', show_plots=False)
        self.minute_prediction = full_minute_prediction[-(self.minute_length + 10)::]
        self.minute_price = full_minute_price[-10::]

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
        s.login('rogeh2@gmail.com', 'Mechelo#0-9')
        s.sendmail('rogeh2@gmail.com', ['rogeh2@gmail.com'], msg_root.as_string())

    def send_err(self):
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('rogeh2@gmail.com', 'Neutrino#0')
        msg = MIMEText('Error detected, will try again in 10min')
        msg['Subject'] = 'Ethereum Prediction Error From Your Digital Broker'
        msg['From'] = 'rogeh2@gmail.com'
        s.sendmail('rogeh2@gmail.com', ['rogeh2@gmail.com'], msg.as_string())

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
            if current_time > (last_check + 15*60):
                try:
                    self.find_data()
                    should_send_email = self.trade_logic(should_send_email)
                except:
                    self.send_err()
                    should_send_email = False
                last_check = current_time
                if should_send_email:
                    self.send_data()
            else:
                time.sleep(1)
            current_time = datetime.utcnow().timestamp()
# TODO try using a classifier Neural Net
# TODO eliminate unnecessary legacy variables from run_neural_net and CryptoPredict

def run_neural_net(date_from, date_to, prediction_length, epochs, prediction_ticker, bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, min_distance_between_trades, model_path, model_type='price', use_type='test', data_set_path=None, save_test_model=True, test_saved_model=False):

    #This creates a CoinPriceModel and saves the data
    if (data_set_path is not None) & (use_type != 'predict') & (not test_saved_model):

        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs,
                            prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                        time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu, data_set_path=data_set_path)

    elif (use_type != 'predict')  & (not test_saved_model):
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                            time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu)


    # 'test' will train a model with given conditions then test it, 'optimize' optimizes neuron count by evaluation data loss, 'predict' predicts data
    if use_type == 'test':
        if test_saved_model:
            cp = CoinPriceModel(date_from, date_to, days=prediction_length, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list, time_units=time_unit, model_path=model_path, need_data_obj=True, data_set_path=data_set_path)
            cp.test_model(did_train=False)
        else:
            cp.train_model(neuron_count=neuron_count, min_distance_between_trades=min_distance_between_trades, model_type=model_type, save_model=save_test_model)
            cp.test_model()

    elif use_type == 'optimize':
        hist = []
        neuron_grid = [2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
        for neuron_count in neuron_grid:
            current_hist = cp.train_model(neuron_count=neuron_count,
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

    code_block = 1
    # 1 for test recent code
    # 2 run_neural_net
    # 3 BaseTradingBot

    if code_block == 1:
        date_from = "2018-05-20 8:50:00 EST"
        date_to = "2018-05-23 8:50:00 EST"
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'minutes'
        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-05-20_8:50:00_EST_to_2018-05-23_8:50:00_EST.pickle'

        strategy_model = CryptoTradeStrategyModel(date_from, date_to, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, data_set_path=pickle_path)
        strategy_model.test_strategy_model()

    elif code_block == 2:

        date_from = "2018-05-23 10:00:00 EST"
        date_to = "2018-05-23 11:00:00 EST"
        prediction_length = 15
        epochs = 500
        prediction_ticker = 'ETH'
        bitinfo_list = ['eth']
        time_unit = 'minutes'
        activ_func = 'relu'
        isleakyrelu = True
        neuron_count = 25
        time_block_length = 60
        min_distance_between_trades = 5
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_6epochs_200neuron1527096914.695041.h5'
        model_type = 'price' #Don't change this
        use_type = 'test' #valid options are 'test', 'optimize', 'predict'. See run_neural_net for description
        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-05-23_11:00:00_EST_to_2018-05-23_12:00:00_EST.pickle'
        test_model_save_bool = True

        run_neural_net(date_from, date_to, prediction_length, epochs, prediction_ticker, bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, min_distance_between_trades, model_path, model_type, use_type, data_set_path=pickle_path, save_test_model=test_model_save_bool, test_saved_model=True)

    elif code_block == 3:

        hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_62epochs_30neuron1527097308.228338.h5'
        minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_6epochs_200neuron1527096914.695041.h5'

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