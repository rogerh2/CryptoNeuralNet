import requests
from datetime import datetime
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

    def create_arrays(self, type='price'):
        if type == 'price':
            self.create_price_prediction_columns()
        elif type == 'difference':
            self.create_difference_prediction_columns()

        temp_input_table = self.final_table#.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.days_out:-1, -1:])
        temp_input_array = np.array(temp_array[0:-(self.days_out+1), ::])
        scaler = StandardScaler()
        temp_input_array = scaler.fit_transform(temp_input_array)
        self.input_array = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)

# TODO add a prediction_columns method (and supporting methods) for peaks and valleys that should be buy and sell points

class CoinPriceModel:

    model = None
    training_array_input = None
    training_array_output = None
    dataObj = None
    days = 1
    activation_func=None
    is_leakyrelu=True
    optimization_scheme="adam"
    loss_func="mae"
    epochs = None
    bitinfo_list = None
    google_list=None

    def __init__(self, date_from, date_to, model_path=None, days=None, bitinfo_list=None, google_list=None,
               prediction_ticker='ltc', epochs=50, activ_func='sigmoid', time_units='hours'):
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

    def build_model(self, inputs, neurons, output_size=1,
                    dropout=0.25):
        is_leaky = self.is_leakyrelu
        activ_func = self.activation_func
        loss = self.loss_func
        optimizer = self.optimization_scheme
        self.model = Sequential()

        self.model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        self.model.add(Dropout(dropout))

        if is_leaky:
            self.model.add(Dense(units=neurons, activation="linear"))
            self.model.add(LeakyReLU(alpha=0.1))
            self.model.add(Dense(units=neurons, activation="linear"))
            self.model.add(LeakyReLU(alpha=0.1))
        else:
            self.model.add(Dense(units=neurons, activation=activ_func))
            self.model.add(Dense(units=neurons, activation=activ_func))

        self.model.add(Dense(units=output_size, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer)

    def create_arrays(self):
        self.dataObj.create_arrays()
        self.training_array_input = self.dataObj.input_array
        self.training_array_output = self.dataObj.output_array

    def train_model(self, neuron_count=200):
        self.create_arrays()
        self.build_model(self.training_array_input, neurons=neuron_count)
        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        self.model.fit(self.training_array_input, self.training_array_output, epochs=self.epochs, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])

        # if self.is_leakyrelu:
        #     self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
        #         self.days) + 'dys_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
        #         self.epochs) + 'epochs_'+ str(neuron_count) + 'neuron'+'.h5')
        # else:
        #     self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
        #         self.days) + 'dys_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
        #         self.epochs) + 'epochs_' + str(neuron_count) + 'neurons' + '.h5')

    def test_model(self, from_date, to_date, time_units='hours'):
        test_data = DataSet(date_from=from_date, date_to=to_date, days=self.days, bitinfo_list=self.bitinfo_list,
                               google_list=self.google_list, prediction_ticker=self.prediction_ticker, time_units=time_units)
        test_data.create_arrays()
        test_input = test_data.input_array
        test_output = test_data.output_array
        prediction = self.model.predict(test_input)

        plt.plot(test_output - np.mean(test_output), 'bo--')
        plt.plot(prediction - np.mean(prediction), 'rx--')
        plt.show()

# TODO add a prediction (for future data) method

if __name__ == '__main__':
    time_unit = 'hours'
    cp = CoinPriceModel("2018-04-12 21:00:00 EST", "2018-04-20 12:00:00 EST", days=3, epochs=400, google_list=['Etherium'], prediction_ticker='ltc', bitinfo_list=['btc', 'eth', 'ltc'], time_units=time_unit)
    cp.train_model(neuron_count=20)
    #cp = CoinPriceModel("2017-04-12", "2017-09-03", model_path='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/ltcmodel_7dys_leakyreluact_adamopt_maeloss_150epochs_200neuron.h5')
    cp.test_model(from_date="2018-04-20 12:01:00 EST", to_date="2018-04-25 19:00:00 EST", time_units=time_unit)