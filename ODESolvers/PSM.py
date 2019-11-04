import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.fftpack import fft
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoPredict.CryptoPredict import CryptoCompare
from CryptoBot.CryptoBot_Shared_Functions import progress_printer


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
    fit_len = 5 * order
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

# piece_wise_fit_eval, evaulatues piecewise polynomials
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

# get error with N sigma confidence
def N_sigma_err(data, fit, N=3, norm=True):
    res = data - fit
    err = (np.abs(np.mean(res)) + N * np.std(res))/(np.max(data) - np.min(data))**norm
    return err

# simple FFT wrapper
def fourier_transform(data, t):
    T = t[1] - t[0]
    N = len(t)
    dataf = fft(data)
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))

    return dataf[0:int(N / 2)], f

# sinusoidal_test_func and find_sin_freq together guess the frequency of the primary sinusoidal component in data
def sinusoidal_test_func(x, b):
    return np.sin(b * x)

def find_sin_freq(data, t):
    T = t[1] - t[0]
    N = len(t)
    dataf = fft(data)
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))
    guess_freq = f[np.argmax(dataf[0:int(N / 2)])]

    try:
        omega, _ = optimize.curve_fit(sinusoidal_test_func, t, data, p0=[guess_freq])
        return omega[0]
    except:
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
            y_next = ( F[n] - 2 * self.zeta * self.omega * y[-1] - self.omega_sq * ( x[-2] - x_minus) - self.omega_plus_sq * (x[-2] - x_plus) ) / (n + 1)
            x_next_next = y_next / (n + 2) # for this particular equation, computing the next x is trivial

            # append to polynomial
            self.poly[0] = np.append(x, x_next_next)
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

    def __init__(self, x0s, y0s, omegas, zetas, t0=0):
        # TODO add dictionary to map symbols to indices (for instance 'BTC' could be mapped to self.N)
        self.t0 = t0
        self.x0s = x0s
        self.y0s = y0s
        self.polynomials = {t0:[]}
        self.N = len(x0s)
        self.omegas = omegas
        self.zetas = zetas
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
            if t_current in time_arr:
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
            self.polynomials[self.t0].append(Polynomial(x0, y0, self.zetas[i], omegas[i], omega_plus))

    def simulate(self, data_sets, t, step_size, poly_order, eval_lens=None, del_len=5):
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
            progress_printer(len(data_sets[0])-np.max(eval_len), ind0, start_ind=del_len, tsk='Simulation')
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

def create_propogator_from_data(raw_data_list, initial_t=0):
    data_len = np.min(np.array([len(x) for x in raw_data_list]))
    concat_data_list = [x[0:data_len] for x in raw_data_list]
    data_list = [(x - np.mean(x)) / (np.max(x) - np.min(x)) for x in concat_data_list]
    coeff_list = [(np.max(x) - np.min(x)) for x in concat_data_list]
    shift_list = [np.mean(x) for x in concat_data_list]
    poly_list = []  # This list will contain polynomial approximations for finding the frequency
    t = np.arange(0, data_len)  # Time in minutes
    poly_len = 5000  # Length of the polynomial approximation (certain size needed for frequency resolution)
    poly_t = np.linspace(0, len(data_list[0]), poly_len)  # Time stamps for polynomials

    # create the polynomial approximation to be used to find the nominal frequency
    for i in range(0, len(data_list)):
        coeff = construct_piecewise_polynomial_for_data(data_list[i], 15, t=t)
        poly_fit, _ = piece_wise_fit_eval(coeff, t=poly_t)
        poly_list.append(poly_fit)
        if not i:
            poly_fit, start_stop = piece_wise_fit_eval(coeff, t=poly_t)
            poly_t = poly_t[start_stop[0]:start_stop[1]]

    initial_xs = [x[0] for x in data_list]
    initial_ys = [np.mean(np.diff(x[0:10])) for x in data_list]
    omegas = [find_sin_freq(pfit, poly_t) for pfit in poly_list]
    zetas = [0 for x in data_list]

    return SystemPropogator(initial_xs, initial_ys, omegas, zetas, t0=initial_t), coeff_list, shift_list


if __name__ == "__main__":
    use_saved_data = False
    sym_list = ['LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP', 'BTC']
    if not use_saved_data:
        cc = CryptoCompare(date_from='2019-11-03 00:30:00 EST', exchange='Coinbase')
        raw_data_list = []
        for sym in sym_list:
            data = cc.minute_price_historical(sym)[sym + '_close'].values
            raw_data_list.append(data)
            print(sym)

        pickle.dump(raw_data_list, open("psm_test.pickle", "wb"))
    else:
        raw_data_list = pickle.load(
            open("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/psm_test.pickle", "rb"))

    train_len = 500  # Length of data to be used for training
    test_len = 100
    t = np.arange(0, test_len)
    psm_step_size = 0.02
    psm_order = 5
    train_list = [x[0:train_len] for x in raw_data_list]

    system_fit, coeff_list, shift_list = create_propogator_from_data(train_list)

    test_list = [x[train_len:train_len+test_len] for x in raw_data_list]
    norm_test_list = [(x - shift)/(coeff) for x, shift, coeff, in zip(test_list, shift_list, coeff_list)]
    system_fit.plot_simulation(norm_test_list, t, psm_step_size, psm_order, coefficients=coeff_list, shifts=shift_list,
                               eval_lens=[10, 15, 20, 30])