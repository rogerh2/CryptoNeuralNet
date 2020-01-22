import numpy as np
import pickle
from time import time
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.fftpack import fft
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoPredict.CryptoPredict import CryptoCompare
from CryptoBot.CryptoBot_Shared_Functions import progress_printer
from CryptoBot.CryptoBot_Shared_Functions import nth_max_ind


# -- Auxillary functions --

# generate a random number between -A and A
def rand_betwee_plus_and_minus(A):
    rand_num = A * np.random.normal()
    while np.abs(rand_num) >= np.abs(A):
        rand_num = A * np.random.normal()
    return rand_num

# construct_piecewise_polynomial_for_data, constructs a piecewise polynomial of desired order to approximate data
def construct_piecewise_polynomial_for_data(data, order, t=None):
    data_len = len(data)
    fit_len = 3 * order
    if 2 * fit_len > data_len:
       raise ValueError('Not enough data for fit')

    fits = [fit_len]
    if t is None:
        fit_t = np.arange(0, fit_len)
    else:
        fit_t = t[0:fit_len]

    for ind in range(0, data_len - fit_len, fit_len):
        if ind >= (data_len - 2*fit_len):
            fit_len = data_len - ind
            fit_t = t[0:fit_len]
        x = data[ind:ind+fit_len]
        coeffs = np.polyfit(fit_t, x, order)
        fits.append((coeffs, t[ind+fit_len-2], t[ind]))

    return fits

# piece_wise_fit_eval, evaulatues piecewise polynomials
def piece_wise_fit_eval(coeffs, t=None):
    fit_len = coeffs[0]
    if t is None:
        t = np.arange(0, coeffs[-1] + fit_len)
    fit = np.array([])
    start_ind = coeffs[1][1]
    stop_ind = coeffs[-1][1] + fit_len

    for coeff_data in coeffs[1::]:
        coeff = coeff_data[0]
        t_max = coeff_data[1]
        t_min = coeff_data[2]
        t_mask = (t>=t_min) * (t<t_max)
        fit_t = t[t_mask] - t_min
        current_fit = np.polyval(coeff, fit_t)
        fit = np.append(fit, current_fit)

    return fit, [start_ind, stop_ind]

# get error with N sigma confidence
def N_sigma_err(data, fit, N=3, norm=True):
    res = data - fit
    err = (np.abs(np.mean(res)) + N * np.std(res))/(np.max(data) - np.min(data))**norm
    return err

def evaluate_fourier_coefficients(a0, a_coeffs, b_coeffs, omegas, t):
    sol = a0 / 2
    for a, b, omega in zip(a_coeffs, b_coeffs, omegas):
        sol += a * np.cos(omega * t) + b * np.sin(omega * t)

    return sol

# simple FFT wrapper
def fourier_transform(data, t, return_type='magnitude'):
    T = t[1] - t[0]
    N = len(t)
    dataf = fft(data)/t.size
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))

    if return_type=='magnitude':
        return np.abs(dataf[0:int(N / 2)]), f
    elif return_type=='complex':
        return dataf[0:int(N / 2)], f
    elif return_type=='real':
        dataf *= 2
        return dataf[0].real, dataf[1:int(N / 2)-1].real, -dataf[1:int(N / 2)-1].imag, f
    else:
        dataf *= 2
        return dataf[0].real, dataf[1:int(N / 2) - 1].real, -dataf[1:int(N / 2) - 1].imag, np.abs(dataf[0:int(N / 2)]), f

def top_N_real_fourier_coefficients(data, t, N):
    a0, a, b, mag, f = fourier_transform(data, t, return_type='all')
    max_inds = [nth_max_ind(mag[1::], n) for n  in np.arange(2, N+1)]
    T = np.max(t) - np.min(t)
    omegas = 2 * np.pi * np.array(max_inds) / T
    a_max = a[max_inds]
    b_max = b[max_inds]

    return a0, a_max, b_max, omegas

# sinusoidal_test_func and find_sin_freq together guess the frequency of the primary sinusoidal component in data
def sinusoidal_test_func(x, b):
    return np.sin(b * x)

def find_sin_freq(data, t, n=1):
    T = t[1] - t[0]
    N = len(t)
    dataf = np.abs(fft(data))
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))
    guess_freq = f[nth_max_ind(dataf[0:int(N / 2)], n)]
    # plt.plot(f, dataf[0:int(N / 2)])

    try:
        omega, _ = optimize.curve_fit(sinusoidal_test_func, t, data, p0=[guess_freq])
        # plt.plot(np.array([omega, omega]), np.array([0, np.max(dataf)]), 'r--')
        # plt.show()
        return omega[0]
    except:
        # plt.plot(np.array([guess_freq, guess_freq]), np.array([0, np.max(dataf)]), 'r--')
        # plt.show()
        return guess_freq

# This calculates the nominal magnitude of a sine wave
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

# -- PSM for N coupled oscillators --

# This class represents the polynomial for a single mass (mass n)
class Polynomial:

    def __init__(self, x0, y0, zeta, omega, omega_plus, F=None):
        self.poly = [np.array([x0, y0]), np.array([y0])]
        if F is None:
            self.force = None
        else:
            self.force = np.flip(F, 0)
        self.zeta = zeta
        self.omega = omega
        self.omega_sq = omega ** 2
        self.omega_plus = omega_plus
        self.omega_plus_sq = omega_plus ** 2

    def generate_next_order(self, x_plus, x_minus):
        # y_n' = F_n - 2 * zeta_n * omega_n * y_n - omega_n^2 * ( x_n - x_(n-1)) - omega_(n+1)^2 * (x_n - x_(n+1))
        # x_n' = y_n
        # x_plus = x_(n+1)
        # x_minus = x_(n-1)
        # Setup variables
        x = self.poly[0][-2]
        y = self.poly[1]
        omega_sq = self.omega_sq
        omega_plus_sq = self.omega_plus_sq
        n = len(y) - 1 # This represents the order of the last calculated Picard iterate

        if self.force is None:
            F = None
        else:
            F = self.force
            if len(x) > len(F):
                print('Not enough terms in F to iterate further')

        # calculate next step
        if (F is not None) and (np.abs(self.zeta) > 0): #If F and zeta are non zero
            y_next = ( F[n] - 2 * self.zeta * self.omega * y[-1] + ( omega_sq - omega_plus_sq) * x + ( omega_plus_sq * x_plus - omega_sq * x_minus)) / (n + 1)
        elif (F is None) and (np.abs(self.zeta) > 0): #If F is zero and zeta is non zero
            y_next = (2 * self.zeta * self.omega * y[-1] + ( omega_sq - omega_plus_sq) * x + ( omega_plus_sq * x_plus - omega_sq * x_minus)) / (n + 1)
        elif (F is not None): #If F nonzero and zeta is zero
            y_next = (F[n] + ( omega_sq - omega_plus_sq) * x + ( omega_plus_sq * x_plus - omega_sq * x_minus)) / (n + 1)
        else: #If F and zeta are both zero
            y_next = (( omega_plus_sq * x_plus + omega_sq * x_minus) - ( omega_sq + omega_plus_sq) * x) / (n + 1)

        x_next_next = y_next / (n + 2) # for this particular equation, computing the next x is trivial

        # append to polynomial
        self.poly[0] = np.append(self.poly[0], x_next_next)
        self.poly[1] = np.append(y, y_next)

    def polynomial(self):
        # returns flipped polynomial for compatability with np.polyval
        return np.flip(self.poly[0], axis=0)

    def derivative(self, derivative_order):
        # returns flipped polynomial for compatability with np.polyval
        if derivative_order <= 1:
            full_poly = self.poly[derivative_order]
        else:
            poly_ind = 0
            full_poly = np.array([])
            for n in range(derivative_order, len(self.poly[poly_ind])):
                numbers_to_multiply = np.arange(n + 1 - derivative_order, n + 1)
                coeff = np.prod(numbers_to_multiply)
                full_poly = np.append(full_poly, coeff * self.poly[poly_ind][n])

        return np.flip(full_poly, axis=0)

    def evaluate_derivatives(self, t, derivative_order=0):
        t = t
        full_poly = self.derivative(derivative_order)
        x = np.polyval(full_poly, t)

        return x, full_poly

    def evaluate_polynomial(self, t):
        x, _ = self.evaluate_derivatives(t, 0)

        return x

    def get_x_and_y_at_t(self, t):
        x = self.evaluate_polynomial(t)
        y, _ = self.evaluate_derivatives(t, 1)

        return x, y

    def __len__(self):
        return len(self.poly[0])

# This class represents a system of ode's (all N masses). It holds the polynomials and evolves them over time
class SystemPropogator:

    def __init__(self, x0s, y0s, omegas, zetas, t0=0, identifiers=None):
        # TODO add dictionary to map symbols to indices (for instance 'BTC' could be mapped to self.N)
        self.t0 = t0
        self.x0s = x0s
        self.y0s = y0s
        self.polynomials = {t0:[]}
        self.N = len(x0s)
        self.omegas = omegas
        self.zetas = zetas
        self.keys = identifiers
        for i in range(0, self.N):
            x0 = x0s[i]
            y0 = y0s[i]
            if i < (self.N - 1):
                omega_plus = omegas[i+1]
            else:
                omega_plus = 0
            self.polynomials[t0].append(Polynomial(x0, y0, zetas[i], omegas[i], omega_plus))

    def t_max(self):
        return np.max(np.array(list(self.polynomials.keys())))

    def generate_next_order(self):
        t = self.t_max()
        polynomial_list = self.polynomials[t]

        for (poly, prev_poly, next_poly) in zip(polynomial_list, [None] + polynomial_list[0:-1], polynomial_list[1::] + [None]):
            # Get values for x_(n-1) and x_(n+1)
            next_polynomial = np.zeros(5)
            prev_polynomial = np.zeros(5)
            if next_poly is None:
                prev_polynomial = prev_poly.polynomial()
            elif prev_poly is None:
                next_polynomial = next_poly.polynomial()
            else:
                prev_polynomial = prev_poly.polynomial()
                next_polynomial = next_poly.polynomial()
            x_minus = prev_polynomial[2] # x_minus is indexed by 2 because prev_polynomial would have already contain the next order
            x_plus = next_polynomial[1]

            # Generate the next order
            poly.generate_next_order(x_plus, x_minus)

    def generate_nth_order_polynomials(self, order):
        for i in range(0, order):
            self.generate_next_order()

    def take_next_step(self, step_size, polynomial_order):
        t_current = self.t_max()
        t_next = self.t_max() + step_size
        self.polynomials[t_next] = []

        for n in range(0, self.N):
            nth_poly = self.polynomials[t_current][n]
            x0, y0 = nth_poly.get_x_and_y_at_t(step_size)
            self.polynomials[t_next].append(Polynomial(x0, y0, nth_poly.zeta, nth_poly.omega, nth_poly.omega_plus))

        self.generate_nth_order_polynomials(polynomial_order)

    def break_up_time(self, time, step_size, polynomial_order, verbose):
        # This method breaks up the time vectors so that the polynoials at each step can be used to evaluate
        t_poly = list(self.polynomials.keys())
        t_poly.sort()
        ind = 0
        i = 0
        max_i = len(t_poly)
        t_dict = {}

        while i < max_i:
            progress_printer(len(time), round(ind, -1), tsk='Propogation', suppress_output=(not verbose))
            t0 = t_poly[i]
            next_i = i + 1
            if (next_i >= max_i) and ((time[-1] - time[ind]) <= step_size):
                t_dict[t0] = time[ind::]
                break
            elif (next_i >= max_i):
                self.take_next_step(step_size, polynomial_order)
                t_poly = list(self.polynomials.keys())
                t_poly.sort()
                max_i = len(t_poly)

            time_mask = (t_poly[i] <= time) * (time <= t_poly[next_i])
            if len(time_mask) == 0:
                time_mask = i
            t_current = time[time_mask]
            t_dict[t0] = t_current
            ind += len(t_current)
            i += 1
            if len(t_current) == 0:
                break

        return t_dict

    def evaluate_nth_polynomial(self, time_arr, step_size, polynomial_order, n=None, verbose=True):
        t = time_arr
        while np.min(np.diff(t)) > step_size:
            t = np.arange(t[0], t[-1], np.min(np.diff(t))/2)
        if not n:
            n = self.N
        n -= 1
        x = np.array([])
        t_fit = np.array([])
        self.generate_nth_order_polynomials(polynomial_order)
        t_dict = self.break_up_time(t, step_size, polynomial_order, verbose)


        for t0 in self.polynomials.keys():
            if t0 not in t_dict.keys():
                break
            time = t_dict[t0]
            if len(time) == 0:
                break
            polynomial = self.polynomials[t0][n]
            current_x = polynomial.evaluate_polynomial(time - time[0])
            x = np.append(x, current_x)
            t_fit = np.append(t_fit, time)

        x_arr = np.array([])
        t_arr = np.array([])
        for x_current, t_current in zip(x, t_fit):
            if np.any(np.isclose(t_current, time_arr)):
                x_arr = np.append(x_arr, x_current)
                t_arr = np.append(t_arr, t_current)

        return x_arr, t_arr

    def evaluate(self, time_arr, step_size, polynomial_order, verbose=True):
        x = []
        for i in range(0, self.N):
            x_fit, t_fit = self.evaluate_nth_polynomial(time_arr, step_size, polynomial_order, n=i + 1, verbose=verbose)
            x.append(x_fit)

        return x, t_fit

    def reset(self, x0s=None, y0s=None):
        # Erase the past data for repropagation
        self.polynomials = {self.t0: []}
        self.N = len(self.omegas)
        omegas = self.omegas
        if not x0s is None:
            self.x0s = x0s
        if not y0s is None:
            self.y0s = y0s

        for i in range(0, self.N):
            x0 = self.x0s[i]
            y0 = self.y0s[i]
            if i < (self.N - 1):
                omega_plus = omegas[i + 1]
            else:
                omega_plus = 0
            self.polynomials[self.t0].append(Polynomial(x0, y0, zetas[i], omegas[i], omega_plus))

    def simulate(self, data_sets, t, step_size, poly_order, eval_lens=None, del_len=10):
        # This method evaluates the polynomials at each point in eval_lens then puts them together in a continuous way
        if eval_lens is None:
            eval_lens = np.array([30])
        ind0 = del_len
        indf = ind0 + np.max(eval_lens)
        x_sets = {}
        single_x_sets = [np.array([]) for i in range(0, self.N)]
        eval_t = {}
        for eval_len in eval_lens:
            # There is a different data set for each length of time
            x_sets[eval_len] = [single_x_sets.copy(), single_x_sets.copy()]
            eval_t[eval_len] = single_x_sets

        while indf < len(t):
            progress_printer(len(data_sets[0]), ind0, start_ind=del_len, tsk='Simulation')
            x0s = [x[ind0] for x in data_sets]
            y0s = [np.mean(np.diff(x[ind0-del_len:ind0])) for x in data_sets]
            current_t = t[ind0:indf] - t[ind0]
            self.reset(x0s=x0s, y0s=y0s)
            x_fit, t_fit = self.evaluate(current_t, step_size, poly_order, verbose=False)
            for key in x_sets.keys():
                eval_len = key - 2
                for i in range(0, self.N):
                    x_sets[key][0][i] = np.append(x_sets[key][0][i], x_fit[i][eval_len]) # TODO taylor eval_len using t_fit
                    x_sets[key][1][i] = np.append(x_sets[key][1][i], data_sets[i][ind0 + eval_len])
                    eval_t[key][i] = np.append(eval_t[key][i], t[ind0 + eval_len])

            ind0 += 1
            indf = ind0 + np.max(eval_lens)

        return x_sets, eval_t

    def plot_simulation(self, data_sets, t, step_size, poly_order, coefficients=None, shifts=None, eval_lens=None, del_len=10):
        # TODO allow plotting of different N's (instead of only the last one)
        x_sets, eval_t = self.simulate(data_sets, t, step_size, poly_order, eval_lens=eval_lens, del_len=del_len)
        for eval_len in x_sets.keys():
            plt.figure()
            plt.plot(coefficients[-1] * x_sets[eval_len][0][-1] + shifts[-1])
            plt.plot(coefficients[-1] * x_sets[eval_len][1][-1] + shifts[-1])
            plt.title(str(eval_len) + ' Minute Prediction')
            plt.legend(('true', 'fit'))
            plt.xlabel('Time (min)')
            plt.ylabel('Price ($)')

        plt.show()

    def save(self, file_path):
        model_topology = {'keys':self.keys, 'omega':self.omegas, 'zeta':self.zetas}
        pickle.dump(model_topology, open(file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        model_topology = pickle.load(open(file_path, "rb"))
        self.keys = model_topology['keys']
        self.omegas = model_topology['omega']
        self.zetas = model_topology['zeta']

    def __getitem__(self, n):
        N = len(self.polynomials[self.t0])
        if (n > 0) and (n < (N + 1)):
            poly = self.polynomials[self.t0][n-1]
            omega = poly.omega
            omegaplus = poly.omega_plus
            zeta = poly.zeta
            X = np.flip(poly.polynomial(), axis=0)
            return X, omega, omegaplus, zeta
        else:
            return np.zeros(len(self.polynomials[self.t0][0])), 0, 0, 0

# -- Fixed Point Iteration to Find All N Omega's and Zeta's --

class Hamiltonian:

    def __init__(self, data, t):
        dataf, f = fourier_transform(data, t)
        self.frequncies = f
        self.fft = dataf
        self.energy_density = self.fft ** 2

    def normalize(self, ref_value, ref_omega):
        self.energy_density = ref_value * self.energy_density / ref_omega

    def __getitem__(self, omega):
        if type(omega) == int:
            omega_ind = np.argmin( np.abs(self.frequncies - omega) )
            E = self.energy_density[omega_ind] * omega
        else:
            E = np.array([])
            for freq in omega:
                omega_ind = np.argmin(np.abs(self.frequncies - freq))
                E = np.append(E, self.energy_density[omega_ind] * freq)

        return E


class SystemIterator:

    def __init__(self, x0s, y0s, omegas, zetas, change_coeff, epsilon, max_iterations=10, t0=0, identifiers=None):
        self.t0 = t0
        self.propogator = SystemPropogator(x0s, y0s, omegas, zetas, t0, identifiers=identifiers)
        self.min_chang_coeff = change_coeff
        self.epsilon = epsilon
        self.max_iter = max_iterations
        self.keys = identifiers

    def get_nth_initial_conditions(self, omega, H):
        # generate a random intial velocity and position that satifies the energy spectrum at that frequency
        # H = (1/2) * ( y^2 + omega^2 * x^2 )
        # H = self.hamiltonian[omega]
        x = rand_betwee_plus_and_minus(np.sqrt(2 * H / omega**2))
        y = np.sqrt(2 * H - omega ** 2 * x ** 2)

        return x, y

    def get_next_nth_omega_zeta(self, n, X=None, X_next=None, X_prev=None):
        # Upadate omega using fixed point iteration
        _, omega0, omegaplus, zeta0 = self.propogator[n]
        if X is None:
            X, _, _, _ = self.propogator[n]
        if X_next is None:
            X_next, _, _, _ = self.propogator[n + 1]
        if X_prev is None:
            X_prev, _, _, _ = self.propogator[n - 1]

        Feff = omega0**2 * X_prev[0:len(X_next)] + omegaplus * X_next
        omegeff = np.sqrt(omega0 ** 2 + omegaplus ** 2)

        omega_squared_numerator = X[1] * Feff[1] - 2 * X[2] * Feff[0] + 4 * X[2] ** 2 - 6 * X[3] * X[1]
        omega_squared_denominator = X[1]**2 - 2 * X[0] * X[2]
        zeta_numerator = omegeff * (Feff[0] * X[1] - 2 * X[2] * X[1] - Feff[1] * X[0] - 6 * X[3] * X[0])
        zeta_denominator = 2 * omega0**2 * omega_squared_denominator
        omega = np.sqrt(omega_squared_numerator / omega_squared_denominator - omegaplus**2)
        zeta = zeta_numerator / zeta_denominator
        # print('omega' + num2str(omega, 3))
        # TODO add zeta back once you get the no damping case working
        return omega, 0 #zeta

    def update_propogator(self, X_true_poly, order, data_list):
        # Used fixed point iteration to create a new propogator
        initial_polys = [np.polyfit(np.arange(0, len(x)), x, 1) for x in data_list]
        x0s = []
        y0s = []
        omegas = []
        zetas = []

        for n in range(0, self.propogator.N):
            # poly = np.append(initial_polys[n], X_true_poly[n][2:order+1])
            current_coeff = X_true_poly[n]
            omega = self.propogator.omegas[n-1] + (np.random.rand() - 0.5)/50
            zeta = 0#self.propogator.zetas[n-1] + (np.random.rand() - 0.5)/500

            x0 = initial_polys[n][1]
            y0 = initial_polys[n][0]

            x0s.append(x0)
            y0s.append(y0)
            omegas.append(omega)
            zetas.append(zeta)

        return SystemPropogator(x0s, y0s, omegas, zetas, self.t0, identifiers=self.keys)

    def compute_system_err(self, x_list, time, step_size, test_len, psm_order, propogator=None, start_i=0):
        err=0
        if propogator is None:
            propogator = self.propogator
        for j in range(0, self.propogator.N):
            progress_printer(self.propogator.N, j, tsk='Evaluationg Polynomials')
            x_guess, t_guess = propogator.evaluate_nth_polynomial(time, step_size, psm_order, n=j + 1, verbose=j==0)
            x = x_list[j]
            current_err = N_sigma_err(x[start_i:start_i+len(x_guess)], x_guess) / test_len
            if not np.isinf(current_err):
                err += current_err
        err = err / self.propogator.N

        return err, x_guess, t_guess

    def random_walk_optimization(self, t, x_list, step_size, order, propogation_len=60):
        segment_start_ind = 0
        x = x_list[-1]
        _, x_guess, t_guess = self.compute_system_err(x_list, np.arange(0, propogation_len), step_size, propogation_len, order, start_i=segment_start_ind) #TODO add validation data
        last_err = 1
        #TODO add validation data
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, x)
        line, = ax.plot(t_guess, x_guess)
        plt.title('Initial Guess'+ ', Error: ' + num2str(last_err, 4))
        # plt.show()
        plt.draw()
        plt.pause(0.1)
        propogators = [self.propogator]
        errs = [last_err]


        for i in range(0, self.max_iter):
            segment_start_ind += propogation_len
            if (segment_start_ind + propogation_len) >= len(t):
                segment_start_ind = 0
            # Create a new propogator and compare to data
            poly = [np.flip(np.polyfit(np.arange(0, propogation_len), a[segment_start_ind:segment_start_ind+propogation_len], 10), axis=0) for a in x_list]
            input_data = [a[segment_start_ind:segment_start_ind+propogation_len] for a in x_list]
            new_propogator = self.update_propogator(poly, order, input_data)
            err, x_guess, t_guess = self.compute_system_err(x_list, np.arange(0, propogation_len), step_size, propogation_len, start_i=segment_start_ind, propogator=new_propogator)
            line.set_ydata(x_guess)
            line.set_xdata(t_guess + segment_start_ind)
            plt.title('Iteration: ' + str(i + 1) + ', Error: ' + num2str(err, 4))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            # plt.close()
            # Create variables for Monte Carlo
            if not np.isnan(err):
                propogators.append(new_propogator)
                errs.append(err)
                u = np.random.random()
                alpha = last_err / err
                if alpha > u:
                    self.propogator = new_propogator
                    last_err = err
            else:
                # self.propogator = propogators[np.argmin(np.array([errs]))]
                print('Final error: ' + num2str(np.min(np.array(errs)), digits=4))
                break

            if err < self.epsilon:
                break


class MultiFrequencySystem:

    def __init__(self, data_list, identifiers, t0=0):
        self.identifiers = identifiers
        self.t0 = t0
        self.data = data_list
        self.propogators = None

    def find_omegas(self, freq_num):
        data = self.data
        t = np.arange(0, len(data[0]))  # Time in minutes
        poly_len = 5000  # Length of the polynomial approximation (certain size needed for frequency resolution
        poly_t = np.linspace(0, len(data[0]), poly_len) # Time stamps for polynomials
        poly_list = []

        # create the polynomial approximation
        for i in range(0, len(data)):
            coeff = construct_piecewise_polynomial_for_data(data[i], 5, t=t)
            poly_fit, start_stop = piece_wise_fit_eval(coeff, t=poly_t)
            poly_list.append(poly_fit)
            if not i:
                poly_fit_t = np.linspace(coeff[1][2], coeff[-1][1], len(poly_fit))

        omegas = []
        a0s = []
        a_coeffs = []
        b_coeffs = []
        for i in range(0, len(poly_list)):
            pfit = poly_list[i]
            a0_i, a_i, b_i, omega_i = top_N_real_fourier_coefficients(pfit, poly_fit_t, freq_num)
            # sol = evaluate_fourier_coefficients(a0_i, a_i, b_i, omega_i, poly_fit_t)
            # start_pt = evaluate_fourier_coefficients(a0_i, a_i, b_i, omega_i, t[-1])
            # data_set = data[i]
            # plt.plot(poly_fit_t, sol)
            # plt.plot(t[-1], start_pt, 'rx')
            # plt.plot(t, data_set)
            # plt.show()
            omegas.append(omega_i)
            a0s.append(a0_i)
            a_coeffs.append(a_i)
            b_coeffs.append(b_i)

        return omegas, a0s, a_coeffs, b_coeffs

    def find_x_from_y_squared(self, y_sq, omegas, ratios):
        y_sq_arr = np.array(y_sq)
        x2_arr = (omegas**-2) * (ratios - y_sq_arr)
        x_arr = np.sqrt(x2_arr)
        return x_arr

    def score_y_sq(self, y_sq, omegas, ratios, target_xy_ratio):
        x_arr = self.find_x_from_y_squared(y_sq, omegas, ratios)
        score = np.abs(np.sum(x_arr) / np.sum(np.sqrt(y_sq)) - target_xy_ratio)
        return score

    def find_xs_and_ys_for_single_currency(self, x0, y0, omega_list, a0, a_list, b_list, T):
        possible_xs = evaluate_fourier_coefficients(a0, a_list, b_list, omega_list, np.arange(0, T, T / 1000))
        # The possible_ys come from the derivative of the Fourier series
        t_arr = np.arange(0, T, T / 1000)
        possible_ys = evaluate_fourier_coefficients(0, omega_list * b_list, - omega_list * a_list, omega_list, t_arr)
        x_distance = possible_xs / x0 - 1
        y_distance = possible_ys / y0 - 1
        tot_distance_sq = x_distance**2 + y_distance**2
        best_ind = np.argmin(tot_distance_sq)
        eval_t = t_arr[best_ind]
        x = a0 / 2 + a_list * np.cos(omega_list * eval_t) + b_list * np.sin(omega_list * eval_t)
        y = - omega_list * a_list * np.sin(omega_list * eval_t) + omega_list * b_list * np.cos( omega_list * eval_t)

        return x, y

    def find_xs_and_ys(self, freq_num, x0s=None, y0s=None):
        omega_list, a0_list, a_list, b_list = self.find_omegas(freq_num)
        x_list = []
        y_list = []

        # The top loop creates the x's and y's for each individual currency
        eval_t = len(self.data[0])
        for j in range(0, len(self.data)):
            data = self.data[j]
            xy_coeff = np.polyfit(np.arange(0, len(data[-11::])), data[-11::], 1)
            if x0s is None:
                x0 = np.polyval(xy_coeff, 11)
            else:
                x0 = x0s[j]
            if y0s is None:
                y0 = xy_coeff[0]
            else:
                y0 = y0s[j]
            # Setup the Hamiltonian and IC's
            x, y = self.find_xs_and_ys_for_single_currency(x0, y0, omega_list[j], a0_list[j], a_list[j], b_list[j], eval_t)
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list, omega_list

    def reset(self, x0s, y0s):
        freq_num = len(self.propogators)
        x_list, y_list, omega_list = self.find_xs_and_ys(freq_num, x0s=x0s, y0s=y0s)
        for k in range(0, freq_num-1):
            x0s = []
            y0s = []
            for j in range(0, len(self.data)):
                x = x_list[j][k]
                y = y_list[j][k]
                x0s.append(x)
                y0s.append(y)

            self.propogators[k].reset(x0s=x0s, y0s=y0s)

    def create_propogator(self, freq_num, calc_freq_num=None):
        if not calc_freq_num:
            calc_freq_num = freq_num
        x_list, y_list, omega_list = self.find_xs_and_ys(calc_freq_num)
        propagators = []

        for k in range(0, freq_num-1):
            x0s = []
            y0s = []
            omegas = []
            ids = []
            for j in range(0, len(self.data)):
                x = x_list[j][k]
                y = y_list[j][k]
                omega = omega_list[j][k]
                id = self.identifiers[j]
                # if k < (freq_num - 1):
                #     id = self.identifiers[j] + '_' + str(k)
                # else:
                #     id = self.identifiers[j]

                x0s.append(x)
                y0s.append(y)
                omegas.append(omega)
                ids.append(id)

            propagator = SystemPropogator(x0s, y0s, omegas, np.zeros(len(x0s)), identifiers=ids)
            propagators.append(propagator)

        self.propogators = propagators

        return propagators

    def evaluate_nth_polynomial(self, time_arr, step_size, polynomial_order, n=None, verbose=True):
        x_fit = None
        for system_fit in self.propogators:
            # TODO use multiprocessing to evaluate all polynomials at once
            if x_fit is None:
                x_fit, t_fit = system_fit.evaluate_nth_polynomial(time_arr, step_size, polynomial_order, n=n,
                                                                  verbose=verbose)
            else:
                x_fit_omega, t_fit = system_fit.evaluate_nth_polynomial(time_arr, step_size, polynomial_order, n=n,
                                                                        verbose=verbose)
                x_fit += x_fit_omega

        return x_fit, t_fit


if __name__ == "__main__":
    run_type = 'crypto'

    if run_type == 'crypto':
        # -- Get raw data --
        model_save_folder = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/psm_models//'
        use_saved_data = True
        sym_list = ['LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP', 'BTC']
        if not use_saved_data:
            cc = CryptoCompare(date_from='2020-01-20 10:00:00 EST', date_to='2020-01-21 10:57:00 EST', exchange='Coinbase')
            raw_data_list = []
            for sym in sym_list:
                data = cc.minute_price_historical(sym)[sym + '_close'].values
                raw_data_list.append(data)
                print(sym)

            data_len = np.min(np.array([len(x) for x in raw_data_list]))
            concat_data_list = [x[0:data_len] for x in raw_data_list]
            pickle.dump(concat_data_list, open("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data/psm_test.pickle", "wb"))
        else:
            concat_data_list = pickle.load(open( "/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/saved_data/psm_test.pickle", "rb" ))

        data_list = [(x - np.mean(x)) / (np.max(x) - np.min(x)) for x in concat_data_list]
        coeff_list = [(np.max(x) - np.min(x)) for x in concat_data_list]
        shift_list = [np.mean(x) for x in concat_data_list]

        # -- Set Up Initial Guess --
        poly_list = [] # This list will contain polynomial approximations for finding the frequency
        t = np.arange(0, len(data_list[0])) # Time in minutes
        poly_len = 5000 # Length of the polynomial approximation (certain size needed for frequency resolution
        poly_t = np.linspace(0, len(data_list[0]), poly_len) # Time stamps for polynomials
        train_len = 120 # Length of data to be used for training
        test_len = 15
        poly_train_ind = (int(train_len*poly_len/len(t)))# Training length equivalent for the polynomial

        # create the polynomial approximation
        for i in range(0, len(data_list)):
            coeff = construct_piecewise_polynomial_for_data(data_list[i], 15, t=t)
            poly_fit, _ = piece_wise_fit_eval(coeff, t=poly_t)
            poly_list.append(poly_fit)
            if not i:
                poly_fit, start_stop = piece_wise_fit_eval(coeff, t=poly_t)
                poly_t = np.linspace(0, len(data_list[i]), len(poly_fit))


        initial_xs = [x[0] for x in data_list]
        initial_ys = [np.mean(np.diff(x[0:10])) for x in data_list]
        omegas = [find_sin_freq(pfit[0:poly_train_ind], poly_t[0:poly_train_ind]) for pfit in poly_list]
        zetas = [0 for x in data_list]
        psm_step_size = 0.1
        psm_order = 5
        t = np.arange(0, train_len)
        x_list = [x[0:train_len] for x in data_list]

        # Optimize the model and save parameters
        sys_iter = SystemIterator(initial_xs, initial_ys, omegas, zetas, None, 0.005, max_iterations=10, identifiers=sym_list)
        # sys_iter.random_walk_optimization(np.linspace(0, np.max(t), len(t)), x_list, psm_step_size, psm_order, propogation_len=test_len)
        # system_fit = sys_iter.propogator
        # system_fit.save(model_save_folder + 'psm_model_' + str(time()) + ''.join(sym_list) + '.pickle')
        # system_fit.load('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/psm_models/psm_model_1576334162.853399LTCLINKZRXXLMALGOETHEOSETCXRPXTZBCHDASHREPBTC.pickle')
        x_train_list = [(x[0:train_len] - np.mean(x[0:train_len])) / (np.max(x[0:train_len]) - np.min(x[0:train_len])) for x in concat_data_list]
        coeff_list = [(np.max(x[0:train_len]) - np.min(x[0:train_len])) for x in concat_data_list]
        shift_list = [np.mean(x[0:train_len]) for x in concat_data_list]
        system_fit = MultiFrequencySystem(x_train_list, sym_list)
        system_fit.create_propogator(freq_num=5)


        # --  Test Propogator --
        initial_polys = [np.polyfit(np.arange(0, len(x)), x, 2) for x in x_train_list]
        # test_x0s0 = [x[train_len] for x in data_list]
        # test_y0s0 = [np.mean(np.diff(x[train_len-10:train_len])) for x in data_list]
        test_x0s = [np.polyval(x, len(x_train_list[0])) for x in initial_polys]
        test_y0s = [x[0] for x in initial_polys]
        test_t = np.arange(0, test_len)
        all_ids = system_fit.propogators[0].keys
        system_fit.reset(x0s=test_x0s, y0s=test_y0s)

        # system_fit.plot_simulation(x_list, t, psm_step_size, psm_order, coefficients=coeff_list, shifts=shift_list, eval_lens=[10, 15, 20, 30])

        for i in range(0, len(sym_list)):
            coeff = coeff_list[i]
            shift = shift_list[i]
            N = all_ids.index(sym_list[i])
            progress_printer(system_fit.propogators[0].N, i, tsk='Evaluating Polynomials')
            x_fit, t_fit = system_fit.evaluate_nth_polynomial(test_t, psm_step_size, psm_order, n=N + 1, verbose=i==False)

            x_fit_coeff = np.polyfit(np.linspace(0, np.max(t_fit), len(x_fit)), x_fit, 1)
            x_plot_fit = np.polyval(x_fit_coeff, np.linspace(0, np.max(test_t), len(x_fit)))
            minmax = 0.5*np.std(x_fit - x_plot_fit)

            x_raw = concat_data_list[i][train_len-10:train_len+2*test_len]
            shift = x_raw[10] - coeff * x_fit[0]
            plt.figure()
            plt.plot(np.linspace(-10, 2*test_len, len(x_raw)), x_raw)
            plt.plot(t_fit, coeff * x_fit + shift)
            plt.plot(np.linspace(0, np.max(test_t), len(x_plot_fit)), coeff * x_plot_fit + shift)
            plt.plot(np.linspace(0, np.max(test_t), len(x_plot_fit)), coeff * ( x_plot_fit + minmax ) + shift, 'r--')
            plt.plot(np.linspace(0, np.max(test_t), len(x_plot_fit)), coeff * ( x_plot_fit - minmax ) + shift, 'r--')
            plt.legend(('true', 'fit', 'mean', 'bounds'))
            plt.title( sym_list[i] + ' Predicted Price Vs Actual')
            plt.xlabel('Time (min)')
            plt.ylabel('Price ($)')
        plt.show()

    else:
        initial_xs = [0.5, -0.5]
        initial_ys = [0, 0.7]
        omegas = [np.sqrt(0.4), np.sqrt(0.8)]
        zetas = [0, 0]
        t = np.arange(0, 50, 0.05)
        step_size = 0.04
        order = 10
        E_ref = (1/2) * ( initial_ys[1]**2 + omegas[1] ** 2 * initial_xs[1] **2 )

        system = SystemPropogator(initial_xs, initial_ys, omegas, zetas)
        x2, t2 = system.evaluate_nth_polynomial(t, step_size, order, n=2)
        x1, t1 = system.evaluate_nth_polynomial(t, step_size, order, n=1)

        # H = Hamiltonian(x1, np.linspace(0, np.max(t), len(x1)))
        # H.normalize(E_ref, omegas[1])

        # sys_iter = SystemIterator([0.5, -0.5], [0, 0.7], [np.sqrt(0.9), np.sqrt(0.2)], [0, 0], None, 0.0004)
        # sys_iter.random_walk_optimization(np.linspace(0, np.max(t), len(x1)), [x1, x2], step_size, order)
        #
        # system_fit = sys_iter.propogator
        # x2_fit, t2_fit = system_fit.evaluate_nth_polynomial(t, step_size, order, n=2)
        # x1_fit, t1_fit = system_fit.evaluate_nth_polynomial(t, step_size, order, n=1)

        plt.close()
        plt.ioff()
        # plt.plot(H.frequncies, H.energy_density * H.frequncies)
        plt.figure()
        plt.plot(np.linspace(0, np.max(t), len(x2)), x2)
        # plt.plot(np.linspace(0, np.max(t), len(x2_fit)), x2_fit)
        plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
        # plt.legend(('true', 'fit'))
        plt.title('2')
        plt.figure()
        plt.plot(np.linspace(0, np.max(t), len(x1)), x1)
        # plt.plot(np.linspace(0, np.max(t), len(x1_fit)), x1_fit)
        plt.title('1')
        # plt.legend(('true', 'fit'))
        plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
        plt.show()