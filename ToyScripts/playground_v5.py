import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.fftpack import fft
from CryptoBot.CryptoBot_Shared_Functions import num2str
from CryptoPredict.CryptoPredict import CryptoCompare
from CryptoBot.CryptoBot_Shared_Functions import progress_printer


# -- Auxillary functions --

# construct_piecewise_polynomial_for_data, constructs a piecewise polynomial of desired order to approximate data
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

# sinusoidal_test_func and find_sin_freq together guess the frequency of the primary sinusoidal component in data
def sinusoidal_test_func(x, b):
    return np.sin(b * x)

def find_sin_freq(data, t):
    T = t[1] - t[0]
    N = len(t)
    dataf = fft(data)
    f = np.linspace(0, 2 * np.pi * 1 / (2 * T), int(N / 2))
    guess_freq = f[np.argmax(dataf[0:int(N / 2)])]

    omega, _ = optimize.curve_fit(sinusoidal_test_func, t, data, p0=[guess_freq])

    return omega[0]

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

# This class represents a system of ode's (all N masses). It holds the polynomials and evolves them over time
class SystemPropogator:

    def __init__(self, x0s, y0s, omegas, zetas, t0=0):
        self.polynomials = {t0:[]}
        self.N = len(x0s)
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
            x0 = nth_poly.evaluate_polynomial(step_size)
            y0, _ = nth_poly.evaluate_derivatives(step_size, derivative_order=1)
            self.polynomials[t_next].append(Polynomial(x0, y0, nth_poly.zeta, nth_poly.omega, nth_poly.omega_plus))

        self.generate_nth_order_polynomials(polynomial_order)

    def break_up_time(self, time, step_size, polynomial_order):
        # This method breaks up the time vectors so that the polynoials at each step can be used to evaluate
        t_poly = list(self.polynomials.keys())
        t_poly.sort()
        ind = 0
        i = 0
        max_i = len(t_poly)
        t_dict = {}

        while i < max_i:
            progress_printer(len(time), round(ind, -1), tsk='Propogation')
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
            t_current = time[time_mask]
            t_dict[t0] = t_current
            ind += len(t_current)
            i += 1
            if len(t_current) == 0:
                break

        return t_dict


    def evaluate_nth_polynomial(self, t, step_size, polynomial_order, n=None):
        if not n:
            n = self.N
        n -= 1
        x = np.array([])
        self.generate_nth_order_polynomials(polynomial_order)
        t_dict = self.break_up_time(t, step_size, polynomial_order)

        for t0 in self.polynomials.keys():
            if t0 not in t_dict.keys():
                break
            time = t_dict[t0]
            if len(time) == 0:
                break
            polynomial = self.polynomials[t0][n]
            current_x = polynomial.evaluate_polynomial(time - time[0])
            x = np.append(x, current_x)

        return x

# -- Fixed Point Iteration to Find All N Omega's and Zeta's --

# -class SystemIterator
# new_omegas = np.array([])
# new_zetas = np.array([])
# -initialize with a SystemPropegator obj as an attribute (or maybe with n omegas, zetas, etc...)
# -def update_nth_omega_zeta: a function to plug in the current omega, zeta, and initial conditions that feed the nth
# equation and use to a get a new omega and zeta, add to the new_omegas and new_zetas arrays
# -def update_nth_initial_conditions: find a random x_n0 and y_n0 that maintain the same Hamitonian as the last iteration
# -def update_propegator: loop through all n zetas, omegas, and initial conditions, then use to create a new SystemPropegator obj
# -def iterate(epsilon, max_iterations, patience): succesively run update_propegator then evaluate all n polynomials,
# compare with the last solutions. Stop when the rmse for each solution from one to next is less than epsilon for
# patience number of iterations, or when max_iterations has been reached





if __name__ == "__main__":
    initial_xs = [1, 2, 3]
    initial_ys = [0.5, 0.5, 0.5]
    omegas = [np.sqrt(0.4), np.sqrt(1.808), np.sqrt(2.712)]
    zetas = [0.025, 0.05, 0.1]

    system = SystemPropogator(initial_xs, initial_ys, omegas, zetas)
    x2 = system.evaluate_nth_polynomial(np.arange(0, 50, 0.01), 0.3, 10)
    x1 = system.evaluate_nth_polynomial(np.arange(0, 50, 0.01), 0.3, 10, n=1)
    plt.plot(np.linspace(0, 50, len(x2)), x2)
    plt.plot([0, 50], [0, 0], 'r--')
    plt.figure()
    plt.plot(np.linspace(0, 50, len(x1)), x1)
    plt.plot([0, 50], [0, 0], 'r--')
    plt.show()