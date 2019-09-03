import numpy as np
from matplotlib import pyplot as plt
from CryptoPredict.CryptoPredict import CryptoCompare

def construct_piecewise_polynomial_for_data(data, order, x=None):
    data_len = len(data)
    fit_len = 15 * order
    if 2 * fit_len > data_len:
       raise ValueError('Not enough data for fit')

    fits = [fit_len]
    if x is None:
        x = np.arange(0, len(data))

    for ind in range(fit_len, data_len - fit_len, fit_len):
        fit_x = x[ind:ind+fit_len]
        y = data[ind:ind+fit_len]
        coeffs = np.polyfit(fit_x, y, order)
        fits.append((coeffs, ind))

    return fits

def piece_wise_fit_eval(coeffs, x=None):
    fit_len = coeffs[0]
    if x is None:
        x = np.arange(0, len(data))
    fit = np.array([])
    start_ind = coeffs[1][1]
    stop_ind = coeffs[-1][1] + fit_len

    for coeff_data in coeffs[1::]:
        fit_x = x[coeff_data[1]:coeff_data[1]+fit_len]
        coeff = coeff_data[0]
        current_fit = np.polyval(coeff, fit_x)
        fit = np.append(fit, current_fit)

    return fit, [start_ind, stop_ind]

def calculate_coefficients_for_second_order(polynomial_coefficients, F):
    # y'' + gamma * y' + omega * y = F
    y = np.flip(polynomial_coefficients[-4::], 0)
    denominator = y[2]*y[0] - y[1]**2
    gamma_numerator = -F*y[1] + y[2]*y[1] + 2*y[3]*y[0]
    omega_numerator = y[2]*F - y[2]**2 - 2*y[3]*y[1]
    gamma = gamma_numerator / denominator
    omega = omega_numerator / denominator

    return gamma, omega

if __name__ == "__main__":
    # cc = CryptoCompare(date_from='2019-08-29 14:39:00 Eastern Standard Time')
    # data = cc.minute_price_historical('ETH')['ETH_close'].values
    decay_rate = 0.05
    freq = np.sqrt(0.9975)
    t = np.arange(0, 30*np.pi, 0.1)
    c = np.pi/2
    data = np.exp(-decay_rate*t)*np.sin(freq*t + c)
    coeff = construct_piecewise_polynomial_for_data(data, 4, x=t)
    poly_fit, start_stop = piece_wise_fit_eval(coeff, x=t)
    fit_t = t[start_stop[0]:start_stop[1]]
    plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
    plt.plot(fit_t, poly_fit, 'rx')
    plt.show()
    for i in range(1, len(coeff)):
        gamma, omega = calculate_coefficients_for_second_order(coeff[i][0], 0)
        print(gamma, omega)
        fit_decay = gamma / 2
        z = gamma / (2 * np.sqrt(omega))
        if np.isnan(z):
            continue
        fit_freq = np.sqrt(omega) * np.sqrt(1 - z**2)
        fit = np.exp(-fit_decay*fit_t)*np.sin(fit_freq*fit_t + c)
        plt.plot(fit_t, data[start_stop[0]:start_stop[1]],'b')
        plt.plot(fit_t, fit,'rx')
        plt.show()