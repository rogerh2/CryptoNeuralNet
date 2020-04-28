import numpy as np
import pickle
from matplotlib import pyplot as plt
from CryptoBot.CryptoBot_Shared_Functions import BaseNN
from ToyScripts.playground_v5 import construct_piecewise_polynomial_for_data
from ToyScripts.playground_v5 import piece_wise_fit_eval
from ToyScripts.playground_v5 import top_N_real_fourier_coefficients
from ToyScripts.playground_v5 import evaluate_fourier_coefficients

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

class FourierNeuralNetworkGenerator:

    def __init__(self, data,  train_len, prediction_len, freq_num):
        self.model = BaseNN()
        self.data = data
        self.train_len = train_len
        self.prediction_len = prediction_len
        self.freq_num = freq_num

    def data_instances(self):
        return self.data, self.train_len, self.prediction_len, self.freq_num

    def create_training_data(self, feture_type, wave_num, step_size=60):
        # This method creates training data to predict either the a_coefficients, b_coefficients, or omegas for the
        # specified frequency  or a0. It predicts using the a's b's and omega's from the training data
        raw_data, train_len, prediction_len, num_frequencies = self.data_instances()
        train_predict_offset = train_len + prediction_len
        training_columns = None

        for i in range(0, len(raw_data) - train_predict_offset, step_size):
            # Create Fourier coefficients
            training_data = raw_data[i:i+train_len]
            prediction_data = raw_data[i+train_len:i+train_predict_offset]
            omega_train, a0_train, a_train, b_train = find_omegas(training_data, num_frequencies)
            omega_predict, a0_predict, a_predict, b_predict = find_omegas(prediction_data, num_frequencies)

            # Create prediction enctry
            if feture_type == 'a0':
                prediction_feature = a0_predict
            elif feture_type == 'a':
                prediction_feature = a_predict[wave_num]
            elif feture_type == 'b':
                prediction_feature = b_predict[wave_num]
            elif prediction_feature == 'omega':
                prediction_feature = omega_predict[wave_num]


            # Create training row
            training_row = np.concatenate([omega_train, np.array([a0_train]), a_train, b_train])
            row_entry = np.append(training_row, prediction_feature)
            if training_columns is None:
                training_columns = np.array([row_entry])
            else:
                training_columns = np.vstack((training_columns, row_entry))

        return training_columns

    # def create_model(self, feture_type, wave_num, step_size=60):
    #     pass