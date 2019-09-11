import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoPredict.CryptoPredict import CryptoCompare

def construct_piecewise_polynomial_for_data(data, order, t=None):
    data_len = len(data)
    fit_len = 15 * order
    if 2 * fit_len > data_len:
       raise ValueError('Not enough data for fit')

    fits = [fit_len]
    if t is None:
        t = np.arange(0, fit_len)
    else:
        t = t[0:fit_len]

    for ind in range(fit_len, data_len - fit_len, fit_len):
        x = data[ind:ind+fit_len]
        coeffs = np.polyfit(t, x, order)
        fits.append((coeffs, ind))

    return fits

def piece_wise_fit_eval(coeffs, t=None):
    fit_len = coeffs[0]
    if t is None:
        t = np.arange(0, fit_len)
    else:
        t = t[0:fit_len]
    fit = np.array([])
    start_ind = coeffs[1][1]
    stop_ind = coeffs[-1][1] + fit_len

    for coeff_data in coeffs[1::]:
        coeff = coeff_data[0]
        current_fit = np.polyval(coeff, t)
        fit = np.append(fit, current_fit)

    return fit, [start_ind, stop_ind]

def calculate_coefficients_for_second_order(polynomial_coefficients, F_coefficients):
    # x'' + zeta * omega * x' + omega^2 * x = F
    x = np.flip(polynomial_coefficients[-4::], 0)
    F = np.flip(F_coefficients[-2::], 0)
    denominator = -2 * x[0] * x[2] + x[1]**2

    zeta_numerator = F[0] * x[1] - 2 * x[2] * x[1] - F[1] * x[0] + 6 * x[3] * x[0]

    omega_squared_numerator =  x[1] * F[1] - 2 * x[2] * F[0] + 4 * x[2]**2 - 6 * x[3] * x[1]

    omega = np.sqrt(omega_squared_numerator / denominator)
    zeta = zeta_numerator / ( 2 * omega * denominator )

    return zeta, omega

def get_Z_and_phi_for_forced_harmonic_oscillator(omega_set, omega_f, zeta_set):
    Z = np.sqrt((2 * omega_set * zeta_set) ** 2 + (1 / (omega_f ** 2)) * (omega_set ** 2 - omega_f ** 2) ** 2)
    phi = np.arctan((2 * omega_f * omega_set * zeta_set) / (omega_f ** 2 - omega_set ** 2))

    return Z, phi

class PSMPolynomialGenerator:

    def __init__(self, x0, y0, zeta, omega, F=None):
        # y' = F - 2 * zeta * omega * y - omega^2 * x
        # x' = y
        self.poly = [np.array([x0, y0]), np.array([y0])]
        self.force = F
        self.zeta = zeta
        self.omega = omega

    def generate_next_order(self):
        # Setup variables
        x = self.poly[0]
        y = self.poly[1]
        n = len(y) - 1 # This represents the order of the last calculated Picard iterate
        if self.force is None:
            F = np.zeros(len(x) + 1)
        else:
            F = F = np.flip(self.force, 0)
        if len(x) > len(F):
            print('Not enough terms in F to iterate further')
        else:
            # calculate next step
            y_next = ( F[n] - 2 * self.zeta * self.omega * y[-1] - (self.omega**2) * x[-2] ) / (n + 1)
            x_next_next = y_next / (n + 2) # for this particular equation, computing the next x is trivial

            # append to polynomial
            self.poly[0] = np.append(x, x_next_next)
            self.poly[1] = np.append(y, y_next)

    def generate_nth_order_polynomial(self, n):
        for i in range(1, n):
            self.generate_next_order()

    def evaluate_polynomial(self, t):
        x = 0
        for n in range(0, len(self.poly[0])):
            x += self.poly[0][n] * t**n

        return x



class ODEFit:

    def __init__(self, response, force, sample_rate=1.0):
        self.response = response
        self.raw_force = force
        self.time = sample_rate * np.arange(0, len(self.response))

    def get_F(self, coeff=1.0):
        F = coeff * self.raw_force
        return F

    def calculate_ode_coeff(self, input_order, ode_func=calculate_coefficients_for_second_order, force_modifier_coeff=1.0):

        response_coeff = construct_piecewise_polynomial_for_data(self.response, input_order, t=self.time)
        F_coeff = construct_piecewise_polynomial_for_data(self.get_F(force_modifier_coeff), input_order, t=self.time)
        ode_coeff = {'zeta':np.array([]), 'omega':np.array([]), 'start_ind':np.array([])}

        for i in range(1, len(coeff)):
            zeta, omega = ode_func(response_coeff[i][0], F_coeff[i][0])
            ode_coeff['zeta'] = np.append(ode_coeff['zeta'], zeta)
            ode_coeff['omega'] = np.append(ode_coeff['omega'], omega)
            ode_coeff['start_ind'] = np.append(ode_coeff['start_ind'], response_coeff[i][1]).astype(int)

        return ode_coeff




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
    F_coeff = construct_piecewise_polynomial_for_data(F, 15, t=t)
    F_poly_fit, start_stop = piece_wise_fit_eval(F_coeff, t=t)
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
    coeff = construct_piecewise_polynomial_for_data(data, 15, t=t)
    poly_fit, start_stop = piece_wise_fit_eval(coeff, t=t)
    fit_t = t[start_stop[0]:start_stop[1]]
    plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
    plt.plot(fit_t, poly_fit, 'rx')
    plt.show()

    # Construct ODE

    fit_finder = ODEFit(data, F, sample_rate=0.01)
    fit_coeff = fit_finder.calculate_ode_coeff(15)

    for i in range(1, len(fit_coeff['start_ind'])-2):
        zeta = fit_coeff['zeta'][i]
        omega = fit_coeff['omega'][i]
        x_0 = coeff[i+2][0][-1]
        y_0 = coeff[i+2][0][-2]
        ind0 = fit_coeff['start_ind'][i]
        indf = fit_coeff['start_ind'][i+1]
        indffit = fit_coeff['start_ind'][i+2]
        print('Calculated zeta: ' + num2str(zeta, 4) + ' Calculated omega: ' + num2str(omega, 4))

        propogator = PSMPolynomialGenerator(x_0, y_0, zeta, omega, F_coeff[i+2][0])
        propogator.generate_nth_order_polynomial(10)
        fit = propogator.evaluate_polynomial(t[indf:indffit] - t[indf])

        # fit_decay = zeta * omega
        # if np.isnan(zeta):
        #     continue
        # fit_freq = omega * np.sqrt(1 - zeta ** 2)
        # fit = np.exp(-fit_decay*fit_t)*np.sin(fit_freq*fit_t + c)
        #
        # Z, phi = get_Z_and_phi_for_forced_harmonic_oscillator(omega, omega_f, zeta)
        # print('Calculated Z: ' + num2str(Z, 4) + ' Calculated phi: ' + num2str(phi, 4))
        # fit = (F0/(Z * omega_f)) * np.sin(omega_f*fit_t + phi)


        plt.plot(fit_t, data[start_stop[0]:start_stop[1]],'b')
        plt.plot(t[indf:indffit], fit,'rx')
        plt.plot(t[ind0:indf], data[ind0:indf], 'bo')
        plt.show()