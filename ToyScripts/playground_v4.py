import numpy as np
from matplotlib import pyplot as plt
from CryptoBot.CryptoBot_Shared_Functions import num2str
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

def calculate_coefficients_for_second_order(polynomial_coefficients, F0, F1):
    # y'' + zeta * omega * y' + omega^2 * y = F
    y = np.flip(polynomial_coefficients[-4::], 0)
    denominator = -2 * y[0] * y[2] + y[1]**2

    zeta_numerator = F0 * y[1] - 2 * y[2] * y[1] - F1 * y[0] + 6 * y[3] * y[0]

    omega_squared_numerator =  y[1] * F1 - 2 * y[2] * F0 + 4 * y[2]**2 - 6 * y[3] * y[1]

    omega = np.sqrt(omega_squared_numerator / denominator)
    zeta = zeta_numerator / ( 2 * omega * denominator )

    return zeta, omega

def get_Z_and_phi_for_forced_harmonic_oscillator(omega_set, omega_f, zeta_set):
    Z = np.sqrt((2 * omega_set * zeta_set) ** 2 + (1 / (omega_f ** 2)) * (omega_set ** 2 - omega_f ** 2) ** 2)
    phi = np.arctan((2 * omega_f * omega_set * zeta_set) / (omega_f ** 2 - omega_set ** 2))

    return Z, phi

if __name__ == "__main__":
    # cc = CryptoCompare(date_from='2019-08-29 14:39:00 Eastern Standard Time')
    # data = cc.minute_price_historical('ETH')['ETH_close'].values

    # Define intial state
    zeta_set = 0.1
    omega_set = np.sqrt(7)
    freq = omega_set * np.sqrt(1 - zeta_set ** 2)
    omega_f = np.pi/2
    F0 = 5
    t = np.arange(0, 30*np.pi, 0.01)

    # Construct test force
    # F = F0*sin(omega_f*t)
    Z, phi = get_Z_and_phi_for_forced_harmonic_oscillator(omega_set, omega_f, zeta_set)
    print('True Z: ' + num2str(Z, 4) + ' True phi: ' + num2str(phi, 4))
    F = F0 * np.sin(omega_f * t)
    F_coeff = construct_piecewise_polynomial_for_data(F, 15, x=t)
    F_poly_fit, start_stop = piece_wise_fit_eval(F_coeff, x=t)
    fit_t = t[start_stop[0]:start_stop[1]]
    plt.plot(fit_t, F[start_stop[0]:start_stop[1]], 'b')
    plt.plot(fit_t, F_poly_fit, 'rx')
    plt.show()

    # Construct test data
    decay_rate = omega_set * zeta_set
    print('True zeta: ' + num2str(zeta_set, 4) + '\nTrue omega: ' + num2str(omega_set, 4))
    c = np.pi/2
    # data = np.exp(-decay_rate*t)*np.sin(freq*t + c)
    data = (F0/(Z * omega_f)) * np.sin(omega_f*t + phi)

    # Construct funtion fit
    coeff = construct_piecewise_polynomial_for_data(data, 15, x=t)
    poly_fit, start_stop = piece_wise_fit_eval(coeff, x=t)
    fit_t = t[start_stop[0]:start_stop[1]]
    plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
    plt.plot(fit_t, poly_fit, 'rx')
    plt.show()
    for i in range(1, len(coeff)):
        print(F_coeff[i][0][-1])
        zeta, omega = calculate_coefficients_for_second_order(coeff[i][0], F_coeff[i][0][-1], F_coeff[i][0][-2])
        print('Calculated zeta: ' + num2str(zeta, 4) + ' Calculated omega: ' + num2str(omega, 4))
        fit_decay = zeta * omega
        # if np.isnan(zeta):
        #     continue
        # fit_freq = omega * np.sqrt(1 - zeta ** 2)
        # fit = np.exp(-fit_decay*fit_t)*np.sin(fit_freq*fit_t + c)
        #
        Z, phi = get_Z_and_phi_for_forced_harmonic_oscillator(omega, omega_f, zeta)
        print('Calculated Z: ' + num2str(Z, 4) + ' Calculated phi: ' + num2str(phi, 4))
        fit = (F0/(Z * omega_f)) * np.sin(omega_f*fit_t + phi)

        plt.plot(fit_t, data[start_stop[0]:start_stop[1]],'b')
        plt.plot(fit_t, fit,'rx')
        plt.show()