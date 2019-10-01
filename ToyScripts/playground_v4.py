import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.fftpack import fft
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

def construct_inverse_t_force(data, window_size, sample_rate=1., coeff=1.):
    forcing = np.zeros(window_size)

    for i in range(window_size, len(data)):
        t = np.flip(1 / (sample_rate * np.arange(0, window_size) + 1), axis=0)
        f = coeff * np.sum(data[i-window_size:i] * t)
        forcing = np.append(forcing, f)

    return forcing

def running_mean(data, window_size, coeff=1.):
    forcing = np.zeros(window_size)

    for i in range(window_size, len(data)):
        f = coeff * np.mean(data[i - window_size:i])
        forcing = np.append(forcing, f)

    return forcing

class PSMPolynomialGenerator:

    def __init__(self, x0, y0, zeta, omega, F=None):
        # y' = F - 2 * zeta * omega * y - omega^2 * x
        # x' = y
        self.poly = [np.array([x0, y0]), np.array([y0])]
        self.force = np.flip(F, 0)
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
            F = self.force
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

    def evaluate_derivatives(self, t, poly_ind, derivative_order=0):
        x = 0
        full_poly = np.array([])
        for n in range(derivative_order, len(self.poly[poly_ind])):
            if derivative_order > 0:
                numbers_to_multiply = np.arange(n+1-derivative_order, n+1)
                coeff = np.prod(numbers_to_multiply)
            else:
                coeff = 1
            x += coeff * self.poly[poly_ind][n] * t ** ( n - derivative_order )
            full_poly = np.append(full_poly, coeff * self.poly[poly_ind][n])

        return x, full_poly

    def evaluate_polynomial(self, t, poly_ind=0):
        x, _ = self.evaluate_derivatives(t, poly_ind)

        return x

    def take_n_steps_for_derivative(self, N, n, step_size, derivative_order, f_arr=None):
        # N is the number of steps, n is the polynomial order, step_size is the length of each step

        full_solution = np.array([])
        all_polynomials = []
        for i in range(0, N):
            if not f_arr is None:
                self.force = f_arr[i]
            self.generate_nth_order_polynomial(n)
            # Evaluate solution for first step
            derivative_sol, full_poly = self.evaluate_derivatives(step_size, poly_ind=0, derivative_order=derivative_order)
            sol = self.evaluate_polynomial(step_size)
            dsol_dt = self.evaluate_polynomial(step_size, poly_ind=1)
            full_solution = np.append(full_solution, derivative_sol)
            all_polynomials.append(np.append(full_poly, np.zeros(derivative_order)))

            # Generate polynomial for the next step
            self.poly = [np.array([sol, dsol_dt]), np.array([dsol_dt])]

        return full_solution, step_size*np.arange(0, N), all_polynomials

    def take_n_steps(self, N, n, step_size, f_arr=None):
        # N is the number of steps, n is the polynomial order, step_size is the length of each step

        full_solution = np.array([])
        all_polynomials = []
        for i in range(0, N):
            if not f_arr is None:
                self.force = f_arr[i]
            self.generate_nth_order_polynomial(n)
            # Evaluate solution for first step
            sol = self.evaluate_polynomial(step_size)
            dsol_dt = self.evaluate_polynomial(step_size, poly_ind=1)
            full_solution = np.append(full_solution, sol)
            all_polynomials.append(self.poly[0])

            # Generate polynomial for the next step
            self.poly = [np.array([sol, dsol_dt]), np.array([dsol_dt])]

        return full_solution, step_size*np.arange(0, N), all_polynomials

    def reset_initial_conditions(self, x0, y0):
        self.poly = [np.array([x0, y0]), np.array([y0])]

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

        for i in range(1, len(F_coeff)):
            zeta, omega = ode_func(response_coeff[i][0], F_coeff[i][0])
            ode_coeff['zeta'] = np.append(ode_coeff['zeta'], zeta)
            ode_coeff['omega'] = np.append(ode_coeff['omega'], omega)
            ode_coeff['start_ind'] = np.append(ode_coeff['start_ind'], response_coeff[i][1]).astype(int)

        return ode_coeff

def test_func(x, b):
    return np.sin(b * x)

def find_sin_freq(data, t):
    T = t[1] - t[0]
    N = len(t)
    dataf = fft(data)
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))
    guess_freq = f[np.argmax(dataf[0:int(N / 2)])]

    omega, _ = optimize.curve_fit(test_func, t, data, p0=[guess_freq])

    return omega[0]

def nominal_magnitude(x0, dx0, omega0):
    x0_norm = np.abs(x0)
    dx0_norm = np.abs(dx0)

    if (x0_norm > 0) and (dx0_norm > 0):
        mag = ( dx0_norm * np.sqrt(1 + omega0 * x0_norm / dx0_norm) ) / omega0
    elif (x0_norm == 0) and (dx0_norm > 0):
        mag = dx0_norm / omega0
    elif (x0_norm > 0) and (dx0_norm == 0):
        mag = x0_norm
    else:
        mag = 0

    return mag


def propogate_n_coupled_oscillators(zetas, omegas, x0s, dx0s, num_steps, step_size, order=15, max_iter_num=100):
    # system =|~~~m1~~~m2 ... ~~~mn, ~~~ denotes spring and damper with coefficients c_i and k_i, m1 = m2 = ... mn = 1
    # setup initial variabls
    t = np.arange(0, num_steps * step_size, step_size)
    force = 0 * t
    fits = [np.sin(omega * t) for omega in omegas]
    dd_fits = [-(omega ** 2) * np.sin(omega * t) for omega in omegas]
    full_force_poly = [np.zeros(order) for i in range(0, len(t))]
    dd_polynomials = [None, None]
    j = 0
    err = np.pi
    epsilon = 0.001
    F_mag = 0

    # iterate and backplug solutions until convergence
    while (err > epsilon) and (j < max_iter_num):
        last_fits = fits.copy()
        j += 1
        for i in range(0, len(dd_fits)):
            current_dd = np.zeros(len(t))
            for dd_partial_fit in current_dd[i::]:
                current_dd = current_dd + dd_partial_fit
            dd_fits[i] = current_dd

        for zeta, omega, x0, dx0, i in zip(zetas, omegas, x0s, dx0s, range(0, len(zetas))):
            # Create an apparent damping coefficient to eliminate exploding solutions
            shift_coeff = 1
            A = nominal_magnitude(x0, dx0, omega)
            z_eff = F_mag / ( 2 * A * omega**2)
            zeta = zeta + z_eff

            # Propogate the current trajectory
            omega_shift = omega * shift_coeff
            f_coeff = np.polyfit(t[0:150], force[0:150], order)

            force_propogator = PSMPolynomialGenerator(x0, dx0, zeta, omega_shift, f_coeff)
            fit_psm_force, _, all_fits_force = force_propogator.take_n_steps(num_steps, order, step_size, f_arr=full_force_poly)

            propogator = PSMPolynomialGenerator(x0, dx0, zeta, omega, f_coeff)
            fit_psm, _, all_fits = propogator.take_n_steps(num_steps, order, step_size, f_arr=full_force_poly)
            propogator.reset_initial_conditions(x0, dx0)
            ddfit_psm, _, all_dd_fits = propogator.take_n_steps_for_derivative(num_steps, order, step_size, 2, f_arr=full_force_poly)
            fits[i] = fit_psm
            dd_fits[i] = ddfit_psm
            dd_polynomials[i] = all_dd_fits

            plt.plot(ddfit_psm)
            plt.plot(fit_psm)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            # coordinates are in the rest frame of the prior oscillating mass, leading to psuedo gravitational forces
            if i > 0:
                psuedo_gravity = dd_fits[i-2]
                psuedo_gravity_polynomials = dd_polynomials[i-2]
                psued0_g_mag = (omegas[i-2] ** 2) * nominal_magnitude(x0s[i-2], dx0s[i-2], omegas[i-2])
            else:
                psuedo_gravity = np.zeros(len(t))
                psuedo_gravity_polynomials = None
                psued0_g_mag = 0

            # Use the current trajectory to create the forces for the next term
            if i < (len(zetas) - 1):
                force = (omega ** 2)*fit_psm_force - psuedo_gravity
                F_mag = (omega ** 2) * nominal_magnitude(x0, dx0, omega) + psued0_g_mag
                if psuedo_gravity_polynomials is not None:
                    full_force_poly = [(omega ** 2)*f - g for f, g in zip(all_fits_force, psuedo_gravity_polynomials)]
                else:
                    full_force_poly = [(omega ** 2) * f for f in all_fits_force]
            else:
                force = psuedo_gravity#(omegas[0]**2)*fit_psm_force
                F_mag = psued0_g_mag
                full_force_poly = [g for g in psuedo_gravity_polynomials]#[(omegas[0] ** 2 ) * f for f in all_fits_force]
                # At the last iteration calculate the forces
                err_arr = np.array([])
                for old_fit, new_fit in zip(last_fits, fits):
                    res = old_fit - new_fit
                    err_arr = np.append( err_arr, np.abs(np.mean(res)) + 2 * np.std(res) )

                err = np.sum(err_arr)

        print(err)

    # for i in range(0, len(fits)):
    #     fit = np.zeros(len(t))
    #     for partial_fit in fits[i::]:
    #         fit = fit + partial_fit
    #
    #     true_fits.append(fit)

    true_fits = fits

    return true_fits, t


def fit_system_of_odes(data, time, order, number_of_hidden_eqautions=5):
    # Format data to range from -1 to 1
    response_poly = construct_piecewise_polynomial_for_data(data, order, t=time)
    inds = [entry[1] for entry in response_poly[1::]]
    ind_list = list(range(0, number_of_hidden_eqautions))
    ind_list.reverse()

    for ind, next_ind in zip(inds[0:-1], inds[1::]):
        # Setup initial guess solution
        y = data[ind:next_ind]
        zero = np.min(y)
        scale = 2 / (np.max(y) - np.min(y))
        y = y - zero
        y = scale * y
        y = y - 1
        y0 = y

        t = time[ind:next_ind]
        force = np.zeros(len(t))
        f_coeff = np.zeros(order)
        fit = np.zeros(len(t))
        omega_guess = find_sin_freq(y, t)
        zeta_guess = 0
        step_size = 1
        polynomial_coefficients = np.polyfit(t - t[0], y, order)
        # Format variables with initial conditions
        omega_arr = []
        zeta_arr = []
        fit_arr = []

        for iteration in range(0, 3):
            y = y0
            for i in ind_list:
                print('omega: ' + num2str(omega_guess) + ', zeta: ' + num2str(zeta_guess) + ', equation: ' + str(i))
                if (zeta_guess == 0) or (iteration == 0):
                    fit_psm = np.sin(omega_guess*t)
                else:
                    propogator = PSMPolynomialGenerator(polynomial_coefficients[-1], polynomial_coefficients[-2], zeta_guess, omega_guess, f_coeff)
                    fit_psm, t_psm, _ = propogator.take_n_steps(len(t), 15, step_size)
                    plt.plot(t_psm, fit_psm)
                    plt.figure()

                # Store for next iteration
                if (iteration == 0):
                    omega_arr.append(omega_guess)
                    zeta_arr.append(zeta_guess)
                    fit_arr.append(fit_psm)
                    fit = fit + fit_psm
                    y = y0 - fit
                else:
                    omega_arr[i] = omega_guess
                    zeta_arr[i] = zeta_guess
                    fit_arr[i] = fit_psm
                    fit = np.zeros(len(fit_psm))
                    fit_force = np.zeros(len(fit_psm))
                    for partial_fit in fit_arr[i::]:
                        fit = fit + partial_fit

                    for partial_fit in fit_arr[0:i]:
                        fit_force = fit_force + partial_fit

                    y = y0 - fit_arr[i]



                plt.plot(fit)
                plt.plot(y)
                plt.show()

                if iteration > 0:
                    polynomial_coefficients = np.polyfit(t - t[0], y, order)
                    F_coefficients = np.polyfit(t - t[0], fit_force, order)
                    zeta_guess, omega_guess = calculate_coefficients_for_second_order(polynomial_coefficients, F_coefficients)
                if np.isnan(zeta_guess) or (iteration == 0):
                    zeta_guess = 0
                    omega_guess = find_sin_freq(y, t)

            plt.plot(t, y0, 'b.')
            plt.plot(t, fit)
            plt.show()

    # for n in range(0, number_of_hidden_eqautions):






if __name__ == "__main__":
    run_type = 'test'
    if run_type == 'full':
        # cc = CryptoCompare(date_from='2019-08-29 14:39:00 Eastern Standard Time')
        # data = cc.minute_price_historical('ETH')['ETH_close'].values

        # # Define intial state
        # zeta_set = 0.01
        # omega_set = np.sqrt(2)
        # freq = omega_set * np.sqrt(1 - zeta_set ** 2)
        # omega_f = np.pi/2
        # F0 = 5
        # t = np.arange(0, 30*np.pi, 0.01)
        #
        # # Construct test force
        # # F = F0*sin(omega_f*t)
        # Z, phi = get_Z_and_phi_for_forced_harmonic_oscillator(omega_set, omega_f, zeta_set)
        # print('True Z: ' + num2str(Z, 4) + ' True phi: ' + num2str(phi, 4))
        # F = F0 * np.sin(omega_f * t)
        # F_coeff = construct_piecewise_polynomial_for_data(F, 15, t=t)
        # F_poly_fit, start_stop = piece_wise_fit_eval(F_coeff, t=t)
        # fit_t = t[start_stop[0]:start_stop[1]]
        # plt.plot(fit_t, F[start_stop[0]:start_stop[1]], 'b')
        # plt.plot(fit_t, F_poly_fit, 'rx')
        # plt.show()
        #
        # # Construct test data
        # decay_rate = omega_set * zeta_set
        # print('True zeta: ' + num2str(zeta_set, 4) + '\nTrue omega: ' + num2str(omega_set, 4))
        # c = np.pi/2
        # data = np.exp(-decay_rate*t)*np.sin(freq*t + c)
        # F = np.zeros(len(data))
        # # data = (F0/(Z * omega_f)) * np.sin(omega_f*t + phi)
        # coeff = construct_piecewise_polynomial_for_data(data, 15, t=t)
        # poly_fit, start_stop = piece_wise_fit_eval(coeff, t=t)
        # fit_t = t[start_stop[0]:start_stop[1]]
        # plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
        # plt.plot(fit_t, poly_fit, 'rx')
        # plt.show()

        # Construct ODE

        cc = CryptoCompare(date_from='2019-09-15 14:39:00 EST')
        data = cc.minute_price_historical('ETH')['ETH_close'].values
        t = np.arange(0, len(data))
        coeff = construct_piecewise_polynomial_for_data(data, 10, t=t)
        poly_fit, start_stop = piece_wise_fit_eval(coeff, t=t)
        fit_t = t[start_stop[0]:start_stop[1]]
        # fit_system_of_odes(np.sin(20*np.arange(0, 10, 0.001)) + 1*np.random.normal(size=len(np.arange(0, 10, 0.001))), np.arange(0, 10, 0.001))
        fit_system_of_odes(data, t, 15)
        plt.plot(fit_t, data[start_stop[0]:start_stop[1]], 'b')
        plt.plot(fit_t, poly_fit, 'rx')
        plt.show()

        # t = fit_t
        # data = poly_fit
        F = construct_inverse_t_force(data, 200, sample_rate=0.1, coeff=0.001)
        F_coeff = construct_piecewise_polynomial_for_data(F, 15, t=t)

        F_poly_fit, start_stop = piece_wise_fit_eval(F_coeff, t=t)
        fit_t = t[start_stop[0]:start_stop[1]]
        plt.plot(fit_t, F[start_stop[0]:start_stop[1]], 'b')
        plt.plot(fit_t, F_poly_fit, 'rx')
        plt.show()

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
            # propogator.generate_nth_order_polynomial(10)
            # fit = propogator.evaluate_polynomial(np.arange(0, 10, 0.01))
            # t_psm = np.arange(0, 10, 0.01)
            fit, t_psm, _ = propogator.take_n_steps(100, 15, 0.1)

            # fit_decay = zeta * omega
            if (np.isnan(zeta)) or (zeta < 1):
                continue
            # fit_freq = omega * np.sqrt(1 - zeta ** 2)
            # fit = np.exp(-fit_decay*fit_t)*np.sin(fit_freq*fit_t + c)
            #
            # Z, phi = get_Z_and_phi_for_forced_harmonic_oscillator(omega, omega_f, zeta)
            # print('Calculated Z: ' + num2str(Z, 4) + ' Calculated phi: ' + num2str(phi, 4))
            # fit = (F0/(Z * omega_f)) * np.sin(omega_f*fit_t + phi)


            plt.plot(t[ind0-30:indffit+30], data[ind0-30:indffit+30],'b')
            plt.plot(t_psm + t[indf], fit,'rx')
            plt.plot(t[ind0:indf], data[ind0:indf], 'bo')
            plt.show()

    else:
        zetas = [0, 0]
        omegas = [np.sqrt(1.808), np.sqrt(0.4)]
        dx0s = [7 / 10, 0]
        x0s = [0, 0.5]
        fit, t = propogate_n_coupled_oscillators(zetas, omegas, x0s, dx0s, 333, 0.3)
        plt.plot(t, fit[1])
        plt.title('Mass 1')
        plt.figure()
        plt.plot(t, fit[0])
        plt.title('Mass 2')
        plt.show()
