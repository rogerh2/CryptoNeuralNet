import numpy as np
import pickle
from time import time
from matplotlib import pyplot as plt
from CryptoBot.CryptoBot_Shared_Functions import LSTM_NN, format_data_for_propogator, find_outliers
from CryptoBot.CryptoBot_Shared_Functions import progress_printer
from ToyScripts.playground_v5 import construct_piecewise_polynomial_for_data
from ToyScripts.playground_v5 import piece_wise_fit_eval
from ToyScripts.playground_v5 import top_N_real_fourier_coefficients
from pandas import DataFrame
import tensorflow as tf
from keras import backend as K
from CryptoPredict.CryptoPredict import CryptoCompare
from sklearn.preprocessing import StandardScaler
from ToyScripts.playground_v5 import evaluate_fourier_coefficients
from ODESolvers.PSM import create_multifrequency_propogator_from_data, MultiFrequencySystem

SYM_LIST = ('ATOM',
            'OXT',
            'LTC',
            'LINK',
            'ZRX',
            'XLM',
            'ALGO',
            'ETH',
            'EOS',
            'ETC',
            'XRP',
            'XTZ',
            'BCH',
            'DASH',
            'REP',
            'BTC',
            'KNC')
MODEL_SAVE_DIR = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/models//'
PRICE_DATA_SAVE_DIR = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data'

def find_trade_strategy_value(buy_bool, sell_bool, all_prices, return_value_over_time=False, fee=0.0035):
    #This finds how much money was gained from a starting value of $100 given a particular strategy
    usd_available = 100
    eth_available = 0

    all_buys = all_prices[buy_bool]
    all_sells = all_prices[sell_bool]
    b = 0
    s = 0
    trade_fee_correction = 1 - fee

    portfolio_value_over_time = np.array([])

    for i in range(0,len(all_prices)):
        if i in buy_bool:
            if (usd_available > 0):
                eth_available = trade_fee_correction*usd_available/(all_buys[b])
                usd_available = 0
            b += 1
        elif i in sell_bool:
            if (eth_available > 0):
                usd_available = trade_fee_correction*all_sells[s]*eth_available
                eth_available = 0
            s += 1

        portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    if usd_available > 0:
        value = usd_available - 100
    else:
        value = all_prices[-1]*eth_available - 100

    portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    if return_value_over_time:
        return value, portfolio_value_over_time
    else:
        return value

def find_omegas(data, freq_num):
    t = np.arange(0, len(data))  # Time in minutes
    poly_len = 5000  # Length of the polynomial approximation (certain size needed for frequency resolution
    poly_t = np.linspace(0, len(data), poly_len)  # Time stamps for polynomials

    coeff = construct_piecewise_polynomial_for_data(data, 5, t=t)
    pfit, start_stop = piece_wise_fit_eval(coeff, t=poly_t)
    poly_fit_t = np.linspace(coeff[1][2], coeff[-1][1], len(pfit))

    periodic_poly_fit_t = poly_fit_t
    periodic_pfit = pfit
    for j in range(0, 20):
        periodic_poly_fit_t = np.append(periodic_poly_fit_t, poly_fit_t + np.max(periodic_poly_fit_t))
        periodic_pfit = np.append(periodic_pfit, pfit)
    a0_i, a_i, b_i, omega_i = top_N_real_fourier_coefficients(periodic_pfit, periodic_poly_fit_t, freq_num)

    return omega_i, a0_i, a_i, b_i

def find_batch_omegas(data_set, freq_num):
    omegas = []
    a0s = []
    a_coeffs = []
    b_coeffs = []
    for data in data_set:
        a0_i, a_i, b_i, omega_i = find_omegas(data, freq_num)
        omegas.append(omega_i)
        a0s.append(a0_i)
        a_coeffs.append(a_i)
        b_coeffs.append(b_i)

    return omegas, a0s, a_coeffs, b_coeffs

def tf_polyval(coeffs, x, name=None):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + p * x
    return p

class OptimalStrategy:
    def __init__(self, data, fee=0.0035, min_profit_coeff=0.002):
        self.fee = fee
        self.min_profit = min_profit_coeff
        self.prices = data

    def calculate_profit(self, cost, profit):
        total_fees = self.fee * (cost + profit)
        net_profit = profit - cost - total_fees
        return net_profit

    def find_next_trade(self, hold, start_ind):
        # This finds the next trade that would return an above threshold profit
        current_price = self.prices[start_ind]
        trade_ind = 0
        for i in range(start_ind, 0, -1):
            test_price = self.prices[i]
            if hold:
                profit = current_price
                cost = test_price
            else:
                profit = test_price
                cost = current_price
            current_value = self.calculate_profit(cost, profit)
            if current_value > self.min_profit * cost:
                trade_ind = i
                break
        return trade_ind

    def find_intermediary_trades(self, start_ind, t, hold):
        test_t = t
        test_ind = start_ind - 10
        for test_t in range(t, start_ind, -1):
            test_ind = self.find_next_trade(hold, test_t)
            if test_ind > start_ind:
                break
        return test_ind, test_t

    def find_positive_profit_interval(self):
        data = self.prices
        hold_arr = np.zeros(len(data))
        t = len(data)-1
        last_t = 0
        buy_ls = []
        sell_ls = []

        while last_t != t:
            last_t = t
            progress_printer(len(data), len(data)-t, tsk='Calculating Strategy')
            # loop backwards through the prices to determine the optimal trade strategy
            buy_ind = self.find_next_trade(True, t)
            sell_ind = self.find_next_trade(False, t)
            if sell_ind < buy_ind:
                test_sell_ind, _ = self.find_intermediary_trades(buy_ind, t, False)
                if test_sell_ind > buy_ind:
                    t = test_sell_ind
                else:
                    for i in range(buy_ind, t):
                        hold_arr[i] = 1
                    t = buy_ind

            else:
                test_buy_ind, test_t = self.find_intermediary_trades(buy_ind, t, False)
                if test_buy_ind > sell_ind:
                    test_t = np.argmax(data[test_buy_ind:test_t]) + test_buy_ind
                    for i in range(test_buy_ind, test_t):
                        hold_arr[i] = 1
                    t = test_buy_ind
                else:
                    t = sell_ind
        for ind in range(1, len(hold_arr)):
            hold = hold_arr[ind]
            last_hold = hold_arr[ind-1]
            if hold == last_hold:
                continue
            elif hold:
                buy_ls.append(ind)
            else:
                sell_ls.append(ind)

        return np.array(buy_ls).astype(int), np.array(sell_ls).astype(int)

    def find_next_trade_diff(self, arr):
        # This finds the difference between the current price and the next trade price
        price = self.prices
        price_deltas = []
        last_buy_ind = 0
        for ind in arr:
            for j in range(last_buy_ind, ind):
                buy_diff = price[ind] - price[j]
                price_deltas.append(buy_diff)
            last_buy_ind = ind
        return price_deltas


def custom_sell_loss_func(y_true, y_pred):
    t = K.arange(0.0, 30.0)
    predicted_val = tf.math.polyval(tf.split(y_pred, y_pred.shape[1].value, axis=1), t)
    true_val = tf.math.polyval(tf.split(y_true, 5, axis=1), t)
    residuals = predicted_val - true_val
    rmse = K.sqrt(K.sum(K.square(residuals))/30)

    return rmse

class FourierPolynomial:

    def __init__(self, omega_list, a0, a_list, b_list):
        self.omegas = omega_list
        self.a0 = a0
        self.a_list = a_list
        self.b_list = b_list

    def find_polynomials(self, order, t0, tf):
        t = np.linspace(t0, tf, 30)
        data = evaluate_fourier_coefficients(self.a0, self.a_list, self.b_list, self.omegas, t)
        coeff = np.polyfit(t, data, order)

        return coeff

class PSM_Model:

    def __init__(self, freq_num=None, F_weights=None, model_path=None):
        if model_path is not None:
            self.F_weights = pickle.load(model_path)
        else:
            self.F_weights = F_weights
        self.num_peaks = freq_num

    # def train(self, training_data, training_curves):

class FourierNeuralNetworkGenerator:

    def __init__(self, data,  train_len, prediction_len, freq_num, model_path=None):
        self.model = LSTM_NN(model_path=model_path)
        self.model.loss_func = custom_sell_loss_func
        self.data = data
        self.train_len = train_len
        self.prediction_len = prediction_len
        self.freq_num = freq_num

    def data_instances(self):
        return self.data, self.train_len, self.prediction_len, self.freq_num

    def create_training_data(self, feture_type, wave_num, step_size):
        # This method creates training data to predict either the a_coefficients, b_coefficients, or omegas for the
        # specified frequency  or a0. It predicts using the a's b's and omega's from the training data
        raw_data, train_len, prediction_len, num_frequencies = self.data_instances()
        train_predict_offset = train_len + prediction_len
        training_columns = None
        prediction_array = np.array([])

        for i in range(0, len(raw_data) - train_predict_offset, step_size):
            # Create Fourier coefficients
            training_data = raw_data[i:i+train_len]
            training_data = (training_data - training_data[-1]) / (np.max(training_data) - np.min(training_data))
            prediction_data = raw_data[i+train_len:i+train_predict_offset]
            prediction_data = (prediction_data - prediction_data[0]) / (np.max(prediction_data) - np.min(prediction_data))
            t = np.arange(0, len(prediction_data))
            omega_train, a0_train, a_train, b_train = find_omegas(np.flip(training_data), num_frequencies)
            omega_predict, a0_predict, a_predict, b_predict = find_omegas(prediction_data, num_frequencies)
            prediction_coeff = np.polyfit(t, prediction_data, 4)


            # Create prediction enctry
            if feture_type == 'a0':
                prediction_feature = a0_predict
            elif feture_type == 'a':
                prediction_feature = a_predict[wave_num]
            elif feture_type == 'b':
                prediction_feature = b_predict[wave_num]
            elif feture_type == 'omega':
                prediction_feature = omega_predict[wave_num]
            elif feture_type == 'polyval':
                prediction_feature = prediction_coeff
            else:
                raise ValueError('Selected Prediction Feature Not Available')


            # Create training row
            training_row = np.concatenate([omega_train, np.array([a0_train]), a_train, b_train])

            if training_columns is None:
                training_columns = np.array([training_row])
            else:
                training_columns = np.vstack((training_columns, training_row))

            if feture_type == 'polyval':
                if len(prediction_array) == 0:
                    prediction_array = prediction_feature
                else:
                    prediction_array = np.vstack((prediction_array, prediction_feature))
            else:
                prediction_array = np.append(prediction_array, prediction_feature)

        scalar = StandardScaler()
        temp_input_arr = scalar.fit_transform(training_columns)

        return temp_input_arr.reshape(training_columns.shape[0], training_columns.shape[1], 1), prediction_array

    def create_model(self, feture_type, wave_num, step_size=30, patience=2):
        training_data, prediction_arr = self.create_training_data(feture_type, wave_num, step_size)
        self.model.build_model(training_data, 20, output_size=len(prediction_arr[0,::]), layer_count=6)

        ts = str(time()).split('.')[0]
        self.model.train_model(training_data, prediction_arr, 70,file_name=MODEL_SAVE_DIR + ts + '_' + feture_type + '_' + str(wave_num) + '.h5',
                               training_patience=patience, batch_size=96)

    def test_model(self, feture_type, wave_num, step_size=30):
        training_data, prediction_arr = self.create_training_data(feture_type, wave_num, step_size)
        self.model.test_model(training_data, prediction_arr)

class PSMLSTM:

    def __init__(self, data, train_len, prediction_len, sym, step_size=5, order=12, model_path=None):
        # This class handles the creation of a LSTM neural net for predicting buy and sell prices
        self.model = LSTM_NN(model_path=model_path)
        # self.model.loss_func = custom_sell_loss_func
        self.prices = data
        self.train_len = train_len
        self.prediction_len = prediction_len
        self.sym = sym
        self.step_size = step_size
        self.order = order

    def data_instances(self):
        return self.prices, self.train_len, self.prediction_len, self.sym

    def get_prediction_features(self, sym, t, coeff_list, shift_list, zero, solver:MultiFrequencySystem):
        psm_step_size = self.step_size
        psm_order= self.order
        ind = solver.identifiers.index(sym)
        coeff = coeff_list[ind]
        shift = shift_list[ind]
        x_fit, t_fit = solver.evaluate_nth_polynomial(t, psm_step_size, psm_order, n=ind + 1, verbose=False)
        err, _, x_reversed = solver.err(psm_step_size, psm_order, ind + 1, coeff, shift, verbose=False, data_len=30)

        x_fit = x_fit - x_fit[0] + zero
        x_reversed = x_reversed - x_reversed[-1] + zero

        min_future_price = np.min(x_fit)
        max_future_price = np.max(x_fit)
        min_past_price = np.min(x_reversed)
        max_past_price = np.max(x_reversed)

        return min_future_price, max_future_price, min_past_price, max_past_price, err

    def aggreagate_price_data(self, start_ind, step_size, data:DataFrame):
        # This method aggregates a slice of data to represent a slice for a different period of time
        data_to_aggregate = data[start_ind:start_ind+step_size]
        aggregate = []
        for header in data.keys():
            data_column = data_to_aggregate[header]
            if 'close' in header:
                aggregate.append(data_column.iloc[-1])
            elif 'high' in header:
                aggregate.append(np.max(data_column.values))
            elif 'low' in header:
                aggregate.append(np.min(data_column.values))
            elif 'open' in header:
                aggregate.append(data_column.iloc[0])
            elif 'volume' in header:
                aggregate.append(np.sum(data_column.values))
        return np.array(aggregate)

    def create_training_data(self, feture_type, step_size, all_currencies_exchange_data, syms, training_data_path=None):
        # This method creates training data for the neural net, this includes predicting the price every step size
        #
        # These are the variables needed to make the prediction
        sym_prices, train_len, prediction_len, sym = self.data_instances()
        train_predict_offset = train_len + prediction_len
        all_prices = format_data_for_propogator(all_currencies_exchange_data)

        # This is the initialization of the variables needed to create the training data
        training_columns = None
        prediction_array = np.array([])
        strategy = OptimalStrategy(sym_prices)

        # These are the answer to the prediction
        buy_arr, sell_arr = strategy.find_positive_profit_interval()
        buy_price_differences = strategy.find_next_trade_diff(buy_arr)
        sell_price_differences = strategy.find_next_trade_diff(sell_arr)

        for i in range(step_size, len(buy_price_differences) - train_predict_offset - 10, step_size):
            progress_printer(len(buy_price_differences) - train_predict_offset, i, digit_resolution=3, start_ind=step_size, tsk='training data creation for ' + sym)
            start_ind = i + train_len
            # Create Fourier coefficients
            if training_data_path is None:
                psm_training_data = [raw_price[i:i + train_len] for raw_price in all_prices]
                system_fit, coeff_list, shift_list = create_multifrequency_propogator_from_data(psm_training_data, syms)
                min_future_price, max_future_price, min_past_price, max_past_price, err = self.get_prediction_features(sym, np.arange(0, 30), coeff_list, shift_list, sym_prices[i+train_len], system_fit)
                current_price_data = self.aggreagate_price_data(start_ind, step_size, all_currencies_exchange_data[sym])

                # Create training row
                training_row = np.append(current_price_data, np.array(
                    [min_future_price, max_future_price, min_past_price, max_past_price, err]))

                if training_columns is None:
                    training_columns = np.array([training_row])
                else:
                    training_columns = np.vstack((training_columns, training_row))

            # Create prediction enctry
            if feture_type == 'buy':
                prediction_feature = buy_price_differences[i+train_len+step_size+1]
            elif feture_type == 'sell':
                prediction_feature = sell_price_differences[i+train_len+step_size+1]
            else:
                raise ValueError('Must predict either buys or sells')

            prediction_array = np.append(prediction_array, prediction_feature)

        if training_data_path is None:
            scalar = StandardScaler()
            training_data = scalar.fit_transform(training_columns)
            ts = str(time()).split('.')[0]
            data_name = ts + '_' + self.sym + '_' + feture_type
            with open('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ManualTradeHelper/Data//' + data_name + '.pickle',
                      'wb') as f:
                pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            training_data = pickle.load(open(training_data_path, "rb"))


        for i in range(0, 12):
            inds_to_delete_1 = find_outliers(training_data[::, -1])
            inds_to_delete_2 = find_outliers(training_data[::, -4])
            inds_to_delete = inds_to_delete_1 + inds_to_delete_2
            training_data = np.delete(training_data, inds_to_delete, axis=0)
            prediction_array = np.delete(prediction_array, inds_to_delete)
            scalar = StandardScaler()
            training_data = scalar.fit_transform(training_data)

        return training_data.reshape(training_data.shape[0], training_data.shape[1], 1), (prediction_array - np.mean(prediction_array))/np.std(prediction_array)

    def create_model(self, feture_type, all_prices, syms, step_size=30, patience=2, training_data_path=None):
        # Create the training data and format
        training_data, prediction_arr = self.create_training_data(feture_type, step_size, all_prices, syms, training_data_path=training_data_path)

        # Build the model and train
        ts = str(time()).split('.')[0]
        data_name = ts + '_' + self.sym + '_' + feture_type
        self.model.build_model(training_data, 20, layer_count=2)

        self.model.train_model(training_data, prediction_arr, 100,file_name=MODEL_SAVE_DIR + data_name + '.h5',
                               training_patience=patience, batch_size=96)

    def test_model(self, feture_type, all_prices, syms, step_size=30, training_data_path=None):
        training_data, prediction_arr = self.create_training_data(feture_type, step_size, all_prices, syms, training_data_path=training_data_path)

        self.model.test_model(training_data, prediction_arr)

if __name__ == "__main__":
    saved_prices_path = "/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data/Minutely Historical Data from 2020-05-09 11:00:00 EST to 2020-05-16 10:00:00 EST"

    RAW_DATA_DICT = pickle.load(open(saved_prices_path,"rb"))
    RAW_DATA = format_data_for_propogator(RAW_DATA_DICT)
    model_path = None#'/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/models/1588044392_a_2.h5'
    price = RAW_DATA[12]
    run_type = 'train'

    if run_type == 'test_strategy':
        strategy = OptimalStrategy(price)
        buy_arr, sell_arr = strategy.find_positive_profit_interval()
        value, portfolio_value_over_time = find_trade_strategy_value(buy_arr, sell_arr, price, return_value_over_time=True)
        norm_price = 100*price/price[0]
        plt.plot(norm_price)
        plt.plot(portfolio_value_over_time)
        plt.plot(buy_arr, norm_price[buy_arr], 'gx')
        plt.plot(sell_arr, norm_price[sell_arr], 'rx')
        plt.title('Optimal Strategy Returns')
        plt.ylabel('Returns (%)')

        buy_price_differences = strategy.find_next_trade_diff(buy_arr)
        sell_price_differences = strategy.find_next_trade_diff(sell_arr)

        plt.figure()
        plt.plot(price, np.array(buy_price_differences), 'k.')
        plt.show()

    elif run_type == 'train':
        i = 0
        # for i in range(0, len(RAW_DATA)):
        price = RAW_DATA[i]
        generator = PSMLSTM(price, 480, 30, SYM_LIST[i])
        generator.create_model('buy', RAW_DATA_DICT, SYM_LIST, patience=70, step_size=8, training_data_path='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ManualTradeHelper/Data/1589654689_ATOM_buy.pickle')
    else:
        i=0
        RAW_DATA_DICT = pickle.load(open(
            "/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data/Minutely Historical Data from 2020-05-16 11:00:00 EST to 2020-05-17 08:00:00 EST",
            "rb"))
        RAW_DATA = format_data_for_propogator(RAW_DATA_DICT)
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/models/1589738917_ATOM_buy.h5'
        price = RAW_DATA[i]
        generator = PSMLSTM(price, 480, 30, SYM_LIST[i], model_path=model_path)
        generator.test_model('buy', RAW_DATA_DICT, SYM_LIST, step_size=8, training_data_path='/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ManualTradeHelper/Data/1589720898_ATOM_buy.pickle')