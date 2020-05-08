import numpy as np
import pickle
from time import time
from matplotlib import pyplot as plt
from CryptoBot.CryptoBot_Shared_Functions import LSTM_NN
from ToyScripts.playground_v5 import construct_piecewise_polynomial_for_data
from ToyScripts.playground_v5 import piece_wise_fit_eval
from ToyScripts.playground_v5 import top_N_real_fourier_coefficients
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from ToyScripts.playground_v5 import evaluate_fourier_coefficients

MODEL_SAVE_DIR = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/models//'

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

if __name__ == "__main__":
    RAW_DATA = pickle.load(open("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/SolversUnitTests/PriceData/price_data_2020-04-21_100000est_to_20200427_105700est.pickle","rb"))
    model_path = None#'/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/models/1588044392_a_2.h5'

    # omega_train, a0_train, a_train, b_train = find_omegas(RAW_DATA[0][0:1200], 5)
    # poly = FourierPolynomial(omega_train, a0_train, a_train, b_train, 5)
    # poly.find_polynomials(10, 0, 320)

    generator = FourierNeuralNetworkGenerator(RAW_DATA[0], 120, 30, 5, model_path=model_path)
    generator.create_model('polyval', 4, patience=70, step_size=3)
    generator.test_model('polyval', 4, step_size=30)