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
        x = np.arange(0, fit_len)
    else:
        x = x[0:fit_len]

    for ind in range(fit_len, data_len - fit_len, fit_len):
        y = data[ind:ind+fit_len]
        coeffs = np.polyfit(x, y, order)
        fits.append((coeffs, ind))

    return fits

def piece_wise_fit_eval(coeffs, x=None):
    fit_len = coeffs[0]
    if x is None:
        x = np.arange(0, fit_len)
    else:
        x = x[0:fit_len]
    fit = np.array([])
    start_ind = coeffs[1][1]
    stop_ind = coeffs[-1][1] + fit_len

    for coeff_data in coeffs[1::]:
        coeff = coeff_data[0]
        current_fit = np.polyval(coeff, x)
        fit = np.append(fit, current_fit)

    return fit, [start_ind, stop_ind]

def calculate_coefficients_for_second_order(polynomial_coefficients, F):
    # y'' + zeta * omega * y' + omega^2 * y = F
    y = np.flip(polynomial_coefficients[-4::], 0)
    denominator = 2 * y[1]**2 - 4 * y[2] * y[0]

    zeta_numerator = -4 * y[2] * y[1] + 12 * y[0] * y[3]

    omega_squared_numerator = 8*y[2]**2 - 12 * y[1] * y[3]

    omega = np.sqrt(omega_squared_numerator / denominator)
    zeta = zeta_numerator / ( 2 * omega * denominator )

    return zeta, omega

if __name__ == "__main__":
    # cc = CryptoCompare(date_from='2019-08-29 14:39:00 Eastern Standard Time')
    # data = cc.minute_price_historical('ETH')['ETH_close'].values
    decay_rate = 0.05
    freq = np.sqrt(0.9975)
    t = np.arange(0, 30*np.pi, 0.01)
    c = np.pi/2
    data = np.exp(-decay_rate*t)*np.sin(freq*t + c)
    coeff = construct_piecewise_polynomial_for_data(data, 10, x=t)
    poly_fit, start_stop = piece_wise_fit_eval(coeff, x=t)
    fit_t = t[start_stop[0]:start_stop[1]]
    plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
    plt.plot(fit_t, poly_fit, 'rx')
    plt.show()
    for i in range(1, len(coeff)):
        zeta, omega = calculate_coefficients_for_second_order(coeff[i][0], 0)
        print(zeta, omega)
        fit_decay = zeta * omega
        if np.isnan(zeta):
            continue
        fit_freq = omega * np.sqrt(1 - zeta ** 2)
        fit = np.exp(-fit_decay*fit_t)*np.sin(fit_freq*fit_t + c)
        plt.plot(fit_t, data[start_stop[0]:start_stop[1]],'b')
        plt.plot(fit_t, fit,'rx')
        plt.show()