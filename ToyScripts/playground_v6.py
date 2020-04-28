import numpy as np
import pickle
from matplotlib import pyplot as plt
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
    sol = evaluate_fourier_coefficients(a0_i, a_i, b_i, omega_i, poly_fit_t)
    start_pt = evaluate_fourier_coefficients(a0_i, a_i, b_i, omega_i, t[-1])
    data_set = data
    # plt.plot(poly_fit_t, sol)
    # plt.plot(t[-1], start_pt, 'rx')
    # plt.plot(t, data_set)
    # plt.plot(poly_fit_t, pfit)
    # plt.show()

    return omega_i, a0_i, a_i, b_i

# This function investigates the relationship between characteristic frequeincies over time
def find_omega_over_time(data, freq_num, data_slice_len):
    # Breakup the data by time chuncks
    data_slices = [data[i:i+data_slice_len] for i in range(0, len(data)-data_slice_len, data_slice_len)]
    omegas = [np.array([]) for i in range(0, freq_num)]
    for data_slice in data_slices:
        omega_i, _, _, _ = find_omegas(data_slice, freq_num)
        for i in range(0, freq_num):
            omegas[i] = np.append(omegas[i], omega_i[i])

    ls = np.zeros(len(omegas[0]))

    for i in range(0, freq_num):
        ls += omegas[i]**2
    plt.plot(ls, 'rx')
    plt.title(str(i) + ' Greatest Frequency Over Time')
    plt.show()


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

def check_if_buys_are_barred(raw_data, std_coeff=0.75):
    # This method checks if the buy price is at least 1 standard deviation below the mean price
    t = np.arange(0, len(raw_data))
    lin_fit = np.polyfit( t, raw_data, 1)
    mean_prices = np.polyval(lin_fit, t)
    mean_price = mean_prices[-1]
    residuals = raw_data - mean_prices
    expected_price_err = np.mean(residuals) + std_coeff * np.std(residuals)

    return expected_price_err, mean_price


if __name__ == "__main__":
    # Define variables
    dat_len = 480 # length to average
    start_ind = 400 # start time to collect data
    extension_len = 480 # amount of time after collection to check
    sym_list = ['ATOM', 'OXT', 'LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH',
                'REP', 'BTC', 'KNC']

    # Load data
    concat_data_list = pickle.load(
        open("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data/psm_test.pickle", "rb"))
    # concat_data_list = [x[0::] for x in concat_data_list]
    # find_batch_omegas(concat_data_list, 10)
    # find_omega_over_time(concat_data_list[5], 5, 40)

    # Compare moving avrages
    for data in concat_data_list:
        raw_dat = data[start_ind:dat_len+start_ind]
        extended_dat = data[start_ind:dat_len + start_ind+extension_len]
        extension_arr = np.array([dat_len, dat_len + extension_len])

        coeff = np.polyfit(np.arange(0, dat_len), raw_dat, 1)
        fit_data = np.polyval(coeff, np.arange(0, dat_len))
        lo_offset, mean_price = check_if_buys_are_barred(raw_dat)
        hi_offset, _ = check_if_buys_are_barred(raw_dat, std_coeff=-1.5)

        plt.plot(fit_data)
        plt.plot(fit_data-hi_offset, 'r--')
        plt.plot(extension_arr,(mean_price-hi_offset)*np.ones(2), 'r--')
        plt.plot(fit_data-lo_offset, 'r--')
        plt.plot(extension_arr, (mean_price - lo_offset) * np.ones(2), 'r--')
        plt.plot(extended_dat)
        plt.show()