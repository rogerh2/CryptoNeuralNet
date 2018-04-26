import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
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

class DataSet:
    days_out=1


    def __init__(self, date_from, date_to, days=None, bitinfo_list = None, google_list = None, prediction_ticker = 'ltc'):
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        if google_list is None:
            google_list = ['Litecoin']
        cryp_obj = Cryptory(from_date=date_from, to_date=date_to)
        fin_table = cryp_obj.extract_bitinfocharts(bitinfo_list[0])

        if len(bitinfo_list) > 1:
            for i in range(1,len(bitinfo_list)):
                temp_table = cryp_obj.extract_bitinfocharts(bitinfo_list[i])
                temp_table = temp_table.drop(columns='date')
                fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

        prediction_table = cryp_obj.extract_bitinfocharts(prediction_ticker)
        prediction_table = prediction_table.drop(columns='date')

        if google_list:
            google_table = cryp_obj.get_google_trends(google_list)
            self.final_table = pd.concat([fin_table, google_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        else:
            self.final_table = pd.concat([fin_table, prediction_table], axis=1,
                                         join_axes=[prediction_table.index])

        if days is not None:
            self.days_out = days

    def create_arrays(self):
        temp_input_table = self.final_table.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.days_out:-1, -1:])
        temp_input_array = np.array(temp_array[0:-(self.days_out+1), ::])
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
    is_leakyrelu=True
    optimization_scheme="adam"
    loss_func="mae"
    epochs = None
    bitinfo_list = None
    google_list=None

    def __init__(self, date_from, date_to, model_path=None, days=None, bitinfo_list=None, google_list=None,
               prediction_ticker='ltc', epochs=50, activ_func='sigmoid'):
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
                               google_list=google_list, prediction_ticker=prediction_ticker)
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
        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(self.training_array_input, self.training_array_output, epochs=self.epochs, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])

        if self.is_leakyrelu:
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
                self.days) + 'dys_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
                self.epochs) + 'epochs_'+ str(neuron_count) + 'neuron'+'.h5')
        else:
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/'+self.prediction_ticker + 'model_' + str(
                self.days) + 'dys_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(
                self.epochs) + 'epochs_' + str(neuron_count) + 'neurons' + '.h5')

    def test_model(self, from_date, to_date):
        test_data = DataSet(date_from=from_date, date_to=to_date, days=self.days, bitinfo_list=self.bitinfo_list,
                               google_list=self.google_list, prediction_ticker=self.prediction_ticker)
        test_data.create_arrays()
        test_input = test_data.input_array
        test_output = test_data.output_array
        prediction = self.model.predict(test_input)

        plt.plot(test_output - np.mean(test_output), 'bo--')
        plt.plot(prediction - np.mean(prediction), 'rx')
        plt.show()


cp = CoinPriceModel("2018-01-04", "2018-04-01", days=2, epochs=400, google_list=['Etherium'], prediction_ticker='eth', bitinfo_list=['btc', 'ltc'])
cp.train_model(neuron_count=300)
#cp = CoinPriceModel("2017-04-12", "2017-09-03", model_path='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/ToyModels/ltcmodel_7dys_leakyreluact_adamopt_maeloss_150epochs_200neuron.h5')
cp.test_model(from_date="2018-04-01", to_date="2018-04-12")