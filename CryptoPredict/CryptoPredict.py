import requests
from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from cryptory import Cryptory
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler



class cryptocompare:

    comparison_symbols = ['USD']
    exchange = ''

    def __init__(self, comparison_symbols=None, exchange=None, date_from=None):

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


    def create_data_frame(self, url, symbol='LTC'):
        page = requests.get(url)
        symbol = symbol.upper()
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df = df.add_prefix(symbol + '_')
        df.insert(loc=0, column='date', value=[datetime.fromtimestamp(d) for d in df[symbol + '_time']])
        df = df.drop(columns=[symbol + '_time'])
        return df


    def daily_price_historical(self, symbol='LTC', all_data=True, limit=1, aggregate=1):
        comparison_symbol = self.comparison_symbols[0]
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

        df = self.create_data_frame(url, symbol)
        return df

    def hourly_price_historical(self, symbol = 'LTC', aggregate=1):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("hours")

        url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

        df = self.create_data_frame(url, symbol)
        return df

    def minute_price_historical(self, symbol='LTC', aggregate = 1):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("minutes")

        url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

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

    def live_social_status(self, symbol, symbol_id_dict={}):

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

class DataSet:
    days_out=1


    def __init__(self, date_from, date_to, days=None, bitinfo_list = None, google_list = None, prediction_ticker = 'ltc', time_units='hours'):
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        if google_list is None:
            google_list = ['Litecoin']
        cryp_obj = cryptocompare(date_from=date_from)
        self.cryp_obj = cryp_obj

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
        for sym in bitinfo_list[1:-1]:
            cryp_obj.symbol = sym
            temp_table = price_func(symbol=sym)
            temp_table = temp_table.drop(columns=['date'])
            fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

        self.fin_table = fin_table
        self.prediction_ticker = prediction_ticker
        self.date_to = date_to
        if days is not None:
            self.days_out = days
        self.time_units = time_units

    def create_price_prediction_columns(self):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        prediction_table = self.price_func(symbol=sym)
        prediction_table = prediction_table.drop(columns=['date', sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_high', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])

        # if google_list:
        #     google_table = cryp_obj.get_google_trends(google_list)
        #     self.final_table = pd.concat([fin_table, google_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        # else:
        #     self.final_table = pd.concat([fin_table, prediction_table], axis=1,
        #                                  join_axes=[prediction_table.index])

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
        if x_point > (upper_bound - delta):
            n = 1
            for n in range(m, max_iterations):
                y_point = np.partition(iterable.values.T, n)[::, n][0]
                x_point = iterable.nsmallest(max_iterations - 1, columns='LTC_open').index[-1] #TODO replace with general ticker based on input
                if x_point < (upper_bound - delta):
                    break
            return x_point, y_point, n
        return x_point, y_point, 1

    def lower_bound_trade_loop(self, x_point, y_point, max_iterations, iterable, delta, lower_bound):
        if x_point < (lower_bound + delta):
            n = 1
            for n in range(1, max_iterations):
                y_point = np.partition(iterable.values.T, -n)[::, -n][0]
                x_point = iterable.nlargest(max_iterations - 1, columns='LTC_open').index[-1] #TODO replace with general ticker based on input
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
            if (current_sell - current_buy) < 0.001 * current_buy or type(current_block_trades) is not pd.DataFrame:
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

    def create_single_prediction_column(self, price_data_frame, trade_data_frame, time_units='hours'):

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
            if scale_time == 0:
                return time_arr #TODO fix error that happens when scale_time is 0 because the first trade is at the first index
            #scale the data

            transformation_coeff = ((-1) ** trade_ind) / full_scale
            time_to_trade = 0.2*(((-1)**trade_ind) > 0)*full_scale + (price_data_frame.values[price_ind] - np.mean(price_data_frame.values))/full_scale + 1 #This line is meant to create jumps at trade points

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
        buy_column = self.create_single_prediction_column(price_data_frame, buy_sell_data_frame.Buy_X, self.time_units)
        sell_column = self.create_single_prediction_column(price_data_frame, buy_sell_data_frame.Sell_X, self.time_units)
        cutoff_len = len(buy_column)
        frame_length = range(0, cutoff_len)
        sell_column = sell_column[frame_length]
        final_indxs = price_data_frame.index[frame_length]
        prediction_frame = pd.DataFrame(data=np.hstack((buy_column, sell_column)), index=final_indxs, columns=['Buy', 'Sell'])
        data_frame = self.fin_table.set_index('date')
        data_frame = data_frame.head(cutoff_len)
        self.final_table = pd.concat([data_frame, prediction_frame], axis=1, join_axes=[prediction_frame.index])

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
            n = -2
            m = -3

        temp_input_table = self.final_table#.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.days_out:-1, n:])
        temp_input_array = np.array(temp_array[0:-(self.days_out+1), 0:m])
        scaler = StandardScaler()
        temp_input_array = scaler.fit_transform(temp_input_array)
        self.input_array = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)

class CoinPriceModel:

    model = None
    training_array_input = None
    training_array_output = None
    dataObj = None
    days = 1
    activation_func=None
    optimization_scheme="adam"
    loss_func="mae"
    epochs = None
    bitinfo_list = None
    google_list=None

    def __init__(self, date_from, date_to, model_path=None, days=None, bitinfo_list=None, google_list=None,
               prediction_ticker='ltc', epochs=50, activ_func='sigmoid', time_units='hours', is_leakyrelu=True):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        if google_list is None:
            google_list = ['Litecoin']
        if days is not None:
            self.days = days
        self.prediction_ticker = prediction_ticker
        self.dataObj = DataSet(date_from=date_from, date_to=date_to, days=self.days, bitinfo_list=bitinfo_list,
                               google_list=google_list, prediction_ticker=prediction_ticker, time_units=time_units)
        self.epochs = epochs
        self.bitinfo_list = bitinfo_list
        self.google_list = google_list
        self.activation_func = activ_func
        self.is_leakyrelu=is_leakyrelu

    def build_model(self, inputs, neurons, output_size=2,
                    dropout=0.25):  #TODO make output_size someing editable outside the class
        is_leaky = self.is_leakyrelu
        activ_func = self.activation_func
        loss = self.loss_func
        optimizer = self.optimization_scheme
        self.model = Sequential()

        self.model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        self.model.add(Dropout(dropout))

        if is_leaky:
            for i in range(0, 1):
                self.model.add(Dense(units=neurons, activation="linear"))
                self.model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, 10):
                self.model.add(Dense(units=neurons, activation=activ_func))

        self.model.add(Dense(units=output_size, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer)

    def create_arrays(self, time_block_length=24, min_distance_between_trades=3, model_type='buy&sell'):
        self.dataObj.create_arrays(model_type, time_block_length, min_distance_between_trades)
        self.training_array_input = self.dataObj.input_array
        self.training_array_output = self.dataObj.output_array

    def train_model(self, neuron_count=200, time_block_length=24, min_distance_between_trades=3, model_type='buy&sell'):
        self.create_arrays(time_block_length, min_distance_between_trades, model_type=model_type)
        self.build_model(self.training_array_input, neurons=neuron_count)
        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        hist = self.model.fit(self.training_array_input, self.training_array_output, epochs=self.epochs, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])
        return hist
        # if self.is_leakyrelu:
        #     self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
        #         self.days) + 'dys_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
        #         self.epochs) + 'epochs_'+ str(neuron_count) + 'neuron'+'.h5')
        # else:
        #     self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
        #         self.days) + 'dys_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
        #         self.epochs) + 'epochs_' + str(neuron_count) + 'neurons' + '.h5')

    def test_model(self, from_date, to_date, time_units='hours', model_type='buy&sell'):
        test_data = DataSet(date_from=from_date, date_to=to_date, days=self.days, bitinfo_list=self.bitinfo_list,
                               google_list=self.google_list, prediction_ticker=self.prediction_ticker, time_units=time_units)
        test_data.create_arrays(model_type=model_type)
        test_input = test_data.input_array
        test_output = test_data.output_array
        prediction = self.model.predict(test_input)

        print(np.mean(test_output) - np.mean(prediction))

        #TODO get rid of hack method for plotting only one line here
        plt.plot(test_input[::, -3] - np.mean(test_input[::, -3]), 'bo--')
        plt.plot(prediction[::, 1] - np.mean(prediction[::, 1]), 'rx--')
        #plt.plot(test_output[::, 0], 'bo--')
        plt.show()

# TODO add a prediction (for future data) method
# TODO try using a classifier Neural Net

if __name__ == '__main__':
    time_unit = 'minutes'
    cp = CoinPriceModel("2018-05-07 00:00:00 EST", "2018-05-07 14:00:00 EST", days=15, epochs=400,
                        google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['ltc'],
                        time_units=time_unit, activ_func='relu', is_leakyrelu=True)

    cp.train_model(neuron_count=3, time_block_length=60, min_distance_between_trades=5, model_type='buy&sell')
    cp.test_model(from_date="2018-05-07 14:05:00 EST", to_date="2018-05-07 23:30:00 EST", time_units=time_unit, model_type='buy&sell')
    # hist = []
    #
    # neuron_grid = [2,3,5,10,20,50,100,200,500,1000,5000]
    # neuron_grid = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
    #
    # for neuron_count in neuron_grid:
    #     current_hist = cp.train_model(neuron_count=neuron_count, time_block_length=60, min_distance_between_trades=5)
    #     hist.append(current_hist.history['val_loss'][-1])
    #
    # plt.plot(neuron_grid, hist, 'bo--')
    # plt.show()


    #cp = CoinPriceModel("2018-04-07 20:00:00 EST", "2018-04-25 20:00:00 EST", days=3, epochs=400, google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['btc', 'eth', 'bnb', 'ltc'], time_units=time_unit, activ_func='sigmoid', is_leakyrelu=True)#for hours
    #cp = CoinPriceModel("2017-04-12", "2017-09-03", model_path='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/ltcmodel_7dys_leakyreluact_adamopt_maeloss_150epochs_200neuron.h5')
    #cp.test_model(from_date="2018-04-20 20:00:00 EST", to_date="2018-05-02 21:00:00 EST", time_units=time_unit)#for hours