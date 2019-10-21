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
    err = (np.abs(np.max(res)) + N * np.std(res))/(np.max(data) - np.min(data))**norm
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

    # omega, _ = optimize.curve_fit(sinusoidal_test_func, t, data, p0=[guess_freq])

    return guess_freq#omega[0]

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
        self.omega_plus = omega_plus

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
            y_next = ( F[n] - 2 * self.zeta * self.omega * y[-1] - self.omega ** 2 * ( x[-2] - x_minus) - self.omega_plus ** 2 * (x[-2] - x_plus) ) / (n + 1)
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
        self.t0 = t0
        self.x0s = x0s
        self.y0s = y0s
        self.polynomials = {t0:[]}
        self.N = len(x0s)
        self.omegas = omegas
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

            time_mask = (t_poly[i] < time) * (time < t_poly[next_i])
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

    def reset(self, x0s=None, y0s=None):
        # Erase the past data for repropagation
        self.polynomials = {self.t0: []}
        self.N = len(self.omegas)
        self.omegas = omegas
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
        self.energy_density = np.abs(dataf) ** 2
        self.frequncies = f
        self.fft = np.square(np.abs(dataf))

    def normalize(self, ref_value, ref_omega):
        self.energy_density = ref_value * self.energy_density / ref_omega

    def __getitem__(self, omega):
        omega_ind = np.argmin( np.abs(self.frequncies - omega) )
        E = self.energy_density[omega_ind] * omega
        return E

class SystemIterator:

    def __init__(self, x0s, y0s, omegas, zetas, H_spectrum, epsilon, max_iterations=10, t0=0):
        self.t0 = t0
        self.propogator = SystemPropogator(x0s, y0s, omegas, zetas, t0)
        self.hamiltonian = H_spectrum
        self.epsilon = epsilon
        self.max_iter = max_iterations

    def get_nth_initial_conditions(self, omega, H):
        # generate a random intial velocity and position that satifies the energy spectrum at that frequency
        # H = (1/2) * ( y^2 + omega^2 * x^2 )
        # H = self.hamiltonian[omega]
        x = rand_betwee_plus_and_minus(np.sqrt(2 * H / omega**2))
        y = np.sqrt(2 * H - omega ** 2 * x ** 2)

        return x, y

    def get_next_nth_omega_zeta(self, n, X=None, X_next=None):
        # Upadate omega using fixed point iteration
        _, omega0, omegaplus, zeta0 = self.propogator[n]
        if X is None:
            X, _, _, _ = self.propogator[n]
        if X_next is None:
            X_next, _, _, _ = self.propogator[n + 1]
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

    def update_propogator(self, X_true_poly):
        # Used fixed point iteration to create a new propogator
        x0s = []
        y0s = []
        omegas = []
        zetas = []

        for n in range(1, self.propogator.N + 1):
            if n == self.propogator.N:
                omega, zeta = self.get_next_nth_omega_zeta(n, X=X_true_poly[1])
            elif n == self.propogator.N-1:
                omega, zeta = self.get_next_nth_omega_zeta(n, X=X_true_poly[0], X_next=X_true_poly[1])
            else:
                omega, zeta = self.get_next_nth_omega_zeta(n)

            if np.isnan(omega):
                omega = self.propogator.omegas[n-1]

            x0 = X_true_poly[n-1][0]
            y0 = X_true_poly[n-1][1]


            x0s.append(x0)
            y0s.append(y0)
            omegas.append(omega)
            zetas.append(zeta)

        return SystemPropogator(x0s, y0s, omegas, zetas, self.t0)

    def compute_system_err(self, x_list, time, step_size, test_len, propogator=None, start_i=0):
        err=0
        if propogator is None:
            propogator = self.propogator
        for j in range(0, self.propogator.N):
            x_guess, t_guess = propogator.evaluate_nth_polynomial(time, step_size, psm_order, n=j + 1, verbose=j == 0)
            x = x_list[j]
            current_err = N_sigma_err(x[start_i:start_i+len(x_guess)], x_guess) / test_len
            if not np.isinf(current_err):
                err += current_err
        err = err / self.propogator.N

        return err, x_guess, t_guess

    def random_walk_optimization(self, t, x_list, step_size, order, val_train_split=1, propogation_len=30):
        val_i = 0
        x = x_list[-1]
        last_err, x_guess, t_guess = self.compute_system_err(x_list, np.arange(0, propogation_len), step_size, propogation_len, start_i=val_i)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, x)
        line, = ax.plot(t_guess, x_guess)
        plt.title('Initial Guess')
        # plt.show()
        plt.draw()
        plt.pause(0.1)
        propogators = [self.propogator]
        errs = [last_err]


        for i in range(0, self.max_iter):
            val_i += propogation_len
            if (val_i + propogation_len) >= len(t):
                val_i = 0
            # Create a new propogator and compare to data
            poly = [np.flip(np.polyfit(np.arange(0, propogation_len), a[val_i:val_i+propogation_len], order), axis=0) for a in x_list]
            new_propogator = self.update_propogator(poly)
            err, x_guess, t_guess = self.compute_system_err(x_list, np.arange(0, propogation_len), step_size, propogation_len, start_i=val_i, propogator=new_propogator)
            line.set_ydata(x_guess)
            line.set_xdata(t_guess + val_i)
            plt.title('Iteration: ' + str(i + 1) + ', Error: ' + num2str(err, 4))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(5)
            # plt.close()

            # Create variables for Monte Carlo
            # TODO reject bad fits (that introduce higher error) with some probability
            if not np.isnan(err):
                propogators.append(new_propogator)
                errs.append(err)
                self.propogator = new_propogator
            else:
                self.propogator = propogators[np.argmin(np.array([errs]))]
                print('Final error: ' + num2str(np.min(np.array(errs)), digits=4))
                break

            if err < self.epsilon:
                break



if __name__ == "__main__":
    # -- Get raw data --
    use_saved_data = True
    sym_list = ['DNT', 'LINK', 'ZRX', 'XLM', 'ALGO', 'EOS', 'ETC', 'XRP', 'ZEC', 'GNT', 'CVC', 'XTZ', 'LOOM', 'DAI', 'BCH', 'BAT', 'DASH', 'MANA', 'ETH', 'REP', 'LTC', 'XLM', 'BTC']
    if not use_saved_data:
        cc = CryptoCompare(date_from='2019-10-18 15:30:00 EST', date_to='2019-10-19 08:30:00 EST')
        raw_data_list = []
        for sym in sym_list:
            data = cc.minute_price_historical(sym)[sym + '_close'].values
            raw_data_list.append(data)
            print(sym)

        data_len = np.min(np.array([len(x) for x in raw_data_list]))
        concat_data_list = [x[0:data_len] for x in raw_data_list]
        data_list = [(x - np.mean(x)) / (np.max(x) - np.min(x)) for x in concat_data_list]

        pickle.dump(data_list, open("psm_test.pickle", "wb"))
    else:
        data_list = pickle.load(open( "/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/psm_test.pickle", "rb" ))

    # -- Set Up Initial Guess --
    poly_list = [] # This list will contain polynomial approximations for finding the frequency
    t = np.arange(0, len(data_list[0])) # Time in minutes
    poly_len = 5000 # Length of the polynomial approximation (certain size needed for frequency resolution
    poly_t = np.linspace(0, len(data_list[0]), poly_len) # Time stamps for polynomials
    train_len = 250 # Length of data to be used for training
    test_len = 100
    poly_train_ind = (int(train_len*poly_len/len(t)))# Training length equivalent for the polynomial

    # create the polynomial approximation
    for data in data_list:
        coeff = construct_piecewise_polynomial_for_data(data, 15, t=t)
        poly_fit, start_stop = piece_wise_fit_eval(coeff, t=poly_t)
        # plt.plot(data[start_stop[0]:start_stop[1]])
        # plt.plot(poly_fit)
        # plt.show()
        poly_list.append(poly_fit)
    poly_t = poly_t[start_stop[0]:start_stop[1]]


    initial_xs = [x[0] for x in data_list]
    initial_ys = [np.mean(np.diff(x[0:10])) for x in data_list]
    omegas = [find_sin_freq(pfit[0:poly_train_ind], poly_t[0:poly_train_ind]) for pfit in poly_list] # TODO setup polynomial approximation to allow more points in omega
    zetas = [0 for x in data_list]
    psm_step_size = 0.02
    psm_order = 5
    t = np.arange(0, train_len)
    x_list = [x[0:train_len] for x in data_list]

    sys_iter = SystemIterator(initial_xs, initial_ys, omegas, zetas, None, 0.001)
    sys_iter.random_walk_optimization(np.linspace(0, np.max(t), len(t)), x_list, psm_step_size, psm_order)
    system_fit = sys_iter.propogator

    # --  Test Propogator --
    test_x0s = [x[train_len] for x in data_list]
    test_y0s = [np.mean(np.diff(x[train_len:train_len+10])) for x in data_list]
    t = np.arange(0, test_len)
    x_list = [x[train_len:train_len+test_len] for x in data_list]
    system_fit.reset(x0s=test_x0s, y0s=test_y0s)

    for i in range(0, len(sym_list)):
        x_fit, t_fit = system_fit.evaluate_nth_polynomial(t, 0.02, 5, n=i + 1)
        x_raw = data_list[i][train_len:train_len+100]
        plt.figure()
        plt.plot(np.linspace(0, np.max(t), len(x_raw)), x_raw)
        plt.plot(np.linspace(0, np.max(t), len(x_fit)), x_fit)
        plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
        plt.legend(('true', 'fit'))
        plt.title( sym_list[i] + ' Predicted Price Vs Actual')
        plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
    plt.show()



    # initial_xs = [0.5, -0.5]
    # initial_ys = [0, 0.7]
    # omegas = [np.sqrt(0.4), np.sqrt(0.8)]
    # zetas = [0, 0]
    # t = np.arange(0, 50, 0.05)
    # step_size = 0.04
    # order = 10
    # E_ref = (1/2) * ( initial_ys[1]**2 + omegas[1] ** 2 * initial_xs[1] **2 )
    #
    # system = SystemPropogator(initial_xs, initial_ys, omegas, zetas)
    # x2, t2 = system.evaluate_nth_polynomial(t, step_size, order, n=2)
    # x1, t1 = system.evaluate_nth_polynomial(t, step_size, order, n=1)
    #
    # # H = Hamiltonian(x1, np.linspace(0, np.max(t), len(x1)))
    # # H.normalize(E_ref, omegas[1])
    #
    # sys_iter = SystemIterator([0.5, -0.5], [0, 0.7], [np.sqrt(0.9), np.sqrt(0.2)], [0, 0], None, 0.0004)
    # sys_iter.random_walk_optimization(np.linspace(0, np.max(t), len(x1)), [x1, x2], step_size, order)
    #
    # system_fit = sys_iter.propogator
    # x2_fit, t2_fit = system_fit.evaluate_nth_polynomial(t, step_size, order, n=2)
    # x1_fit, t1_fit = system_fit.evaluate_nth_polynomial(t, step_size, order, n=1)
    #
    # plt.close()
    # plt.ioff()
    # # plt.plot(H.frequncies, H.energy_density * H.frequncies)
    # plt.figure()
    # plt.plot(np.linspace(0, np.max(t), len(x2)), x2)
    # plt.plot(np.linspace(0, np.max(t), len(x2_fit)), x2_fit)
    # plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
    # plt.legend(('true', 'fit'))
    # plt.title('2')
    # plt.figure()
    # plt.plot(np.linspace(0, np.max(t), len(x1)), x1)
    # plt.plot(np.linspace(0, np.max(t), len(x1_fit)), x1_fit)
    # plt.title('1')
    # plt.legend(('true', 'fit'))
    # plt.plot([np.min(t), np.max(t)], [0, 0], 'r--')
    # plt.show()