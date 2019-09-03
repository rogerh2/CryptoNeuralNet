import requests
from requests.auth import AuthBase
import re
from datetime import datetime
from datetime import timedelta
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
import pytz
import pickle
import tzlocal
import base64
import hashlib
import hmac
import gdax
import sys
from textblob import TextBlob as txb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


def convert_time_to_uct(naive_date_from):
    current_tz = get_current_tz()
    sys_tz = pytz.timezone(current_tz)
    sys_tz_date_from = sys_tz.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = sys_tz_date_from.astimezone(utc)
    return utc_date

def get_current_tz():
    now = datetime.now(tzlocal.get_localzone())
    tz = now.strftime('%Z')
    if tz == 'EDT':
        tz = 'America/New_York'

    return tz

def find_trade_strategy_value(buy_bool, sell_bool, all_prices, return_value_over_time=False):
    #This finds how much money was gained from a starting value of $100 given a particular strategy
    usd_available = 100
    eth_available = 0

    all_buys = all_prices[buy_bool]
    all_sells = all_prices[sell_bool]
    b = 0
    s = 0
    trade_fee_correction = 1

    portfolio_value_over_time = np.array([])

    for i in range(0,len(sell_bool)):
        if buy_bool[i]:
            if (usd_available > 0):
                eth_available = trade_fee_correction*usd_available/(all_buys[b])
                usd_available = 0
            b += 1
        elif sell_bool[i]:
            if (eth_available > 0):
                usd_available = trade_fee_correction*all_sells[s]*eth_available
                eth_available = 0
            s += 1

        portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    if usd_available > 0:
        value = usd_available - 100
    else:
        value = all_prices[-1]*eth_available - 100

    portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    if return_value_over_time:
        return value, portfolio_value_over_time
    else:
        return value

def find_optimal_trade_strategy(offset_data, show_plots=False, min_price_jump = 1.0001):
    data = offset_data - np.min(offset_data)
    buy_array = np.zeros(len(data))
    sell_array = np.zeros(len(data))
    all_times = np.arange(0,len(data))
    data_len = len(data)
    price_is_rising = None
    for i in range(data_len):
        ind = data_len - i -1
        current_price = data[ind]
        prior_price = data[ind-1]
        bool_price_test = current_price > prior_price

        if price_is_rising is None:
            price_is_rising = bool_price_test
            last_inflection_price = current_price
        else:
            if bool_price_test != price_is_rising: # != acts as an xor gate
                if price_is_rising:
                    if current_price*min_price_jump < last_inflection_price:
                        buy_array[ind] = 1
                else:
                    if current_price/min_price_jump > last_inflection_price:
                        sell_array[ind] = 1
                last_inflection_price = current_price
                price_is_rising = bool_price_test

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]
    buy_bool[0] = False
    sell_bool[-1] = False
    if show_plots:
        gain = find_trade_strategy_value(buy_bool, sell_bool, data)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('$'+str(gain))
        plt.show()
    else:
        return sell_bool, buy_bool

def find_optimal_trade_strategy_stochastic(prediction, data, offset=40, prediction_length=30): #Cannot be copie pasted, this is a test
    buy_array = np.zeros(len(data))
    sell_array = np.zeros(len(data))
    data_len = len(data)
    price_is_rising = None

    #zerod_prediction = prediction - np.min(prediction)
    #scaled_prediction = zerod_prediction/np.max(zerod_prediction)
    #prediction = np.max(data)*scaled_prediction + np.mean(data)

    # data = data - np.min(data)
    # data = data/np.max(data)

    for i in range(0, data_len-offset):
        ind = data_len - i - 1
        data_ind = ind + prediction_length
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr #this array will contain the residual from the prior datum
        fuzzzy_counter = 0

        for N in range(10, offset):
            if i >= (prediction_length):
                past_predictions = prediction[(ind-N):(ind)]
                past_data = data[(data_ind-N):(data_ind)]
            else:
                past_predictions = prediction[(data_len - prediction_length - N + 1):(data_len - prediction_length + 1)]
                past_data = data[(-N)::]

            #Find error
            current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
            current_coeff = current_fit[0][0]
            current_off = current_fit[0][1]
            current_err = 2*np.sqrt(current_fit[1]/(N-1))
            err_arr = np.append(err_arr, current_err)
            off_arr = np.append(off_arr, current_off)
            coeff_arr = np.append(coeff_arr, current_coeff)

            if i >= (prediction_length):
                err_judgement_arr = np.append(err_judgement_arr, np.abs(prediction[ind-1] - current_off - current_coeff*data[data_ind-1]) +  current_err) # current_err/np.sqrt(N)) #
            else:
                err_judgement_arr = np.append(err_judgement_arr, np.abs(
                    prediction[ind - 1] - current_off - current_coeff * data[
                        N - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1/coeff_arr[err_ind]
        if fit_coeff < 0:
            #This screens for bad fits
            continue

        err = err_arr[err_ind]*fit_coeff
        fit_offset = -off_arr[err_ind]*fit_coeff
        const_diff = 2*err
        fuzziness = int((err_ind + 10)/2)

        if (fuzziness + ind) > data_len:
            #This ensures that the fuzziness is not too large
            continue


        current_price = np.mean(fit_coeff * prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        prior_price = np.mean(fit_coeff * prediction[(ind - fuzziness - 1):(ind + fuzziness - 1)] + fit_offset)
        bool_price_test = current_price > prior_price
        upper_price = current_price + err
        lower_price = current_price - err

        if price_is_rising is None:
            price_is_rising = bool_price_test
            next_inflection_ind = ind
        else:

            if (fuzziness + next_inflection_ind) > data_len:
                # This factors in the limited precitions available when live
                fuzziness = prediction_length - (next_inflection_ind  - ind)
                next_inflection_price = np.mean(fit_coeff * prediction[(next_inflection_ind - fuzziness):(next_inflection_ind + fuzziness)] + fit_offset)
                fuzzzy_counter += 1
            else:
                next_inflection_price = np.mean(fit_coeff * prediction[(next_inflection_ind - fuzziness):(next_inflection_ind + fuzziness)] + fit_offset)
            upper_inflec = next_inflection_price + err
            lower_inflec = next_inflection_price - err
            if bool_price_test != price_is_rising:  # != acts as an xor gate
                if price_is_rising:
                    ln_diff = (np.log(upper_price) - np.log(lower_price))/const_diff
                    sq_diff = ((upper_inflec)**2 - (lower_inflec)**2)/(2*const_diff)
                    check_val = sq_diff * ln_diff - 1
                    #The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    #for both the buy and sell prices based on past errors
                    #both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if check_val > 0:
                        buy_array[ind] = 1

                    next_inflection_ind = ind
                else:
                    ln_diff = (np.log(upper_inflec) - np.log(lower_inflec))/const_diff
                    sq_diff = ((upper_price)**2 - (lower_price)**2)/(2*const_diff)
                    check_val = sq_diff * ln_diff - 1

                    if check_val > 0:
                        sell_array[ind] = 1

                    next_inflection_ind = ind


                price_is_rising = bool_price_test

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]

    return sell_bool, buy_bool

def clean_data(data):
    new_data = data
    data_diff = np.diff(data)
    diff_mean = np.mean(np.abs(data_diff))
    diff_std = np.std(np.abs(data_diff))
    for i in range(1, len(data)-1):
        prior_diff = data_diff[i-1]
        next_diff = data_diff[i]
        prior_diff_sign = prior_diff > 0
        next_diff_sign = next_diff > 0
        if prior_diff_sign != next_diff_sign: #this implements xor
            if (np.abs(prior_diff) > (diff_mean + diff_std)) or (np.abs(next_diff) > (diff_mean + diff_std)):
                new_data[i] = (data[i-1] + data[i+1])/2

    return  new_data

def num2str(num, digits):
    fmt_str = "{:0." + str(digits) + "f}"
    num_str = fmt_str.format(num)

    return num_str

class OptimalTradeStrategy:

    offset = 40
    prediction_len = 30

    def __init__(self, prediction, data):
        self.data = data
        self.prediction = prediction
        self.buy_array = np.zeros(len(data)+1)
        self.sell_array = np.zeros(len(data)+1)
        self.data_len = len(data)

    def find_fit_info(self, ind):
        #This method searches the past data to determine what value should be used for the error
        prediction = self.prediction
        data = self.data
        offset = self.offset
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr  # this array will contain the residual from the prior datum

        for N in range(10, offset):
            past_predictions = prediction[(ind - N):(ind)]
            past_data = data[(ind - N):(ind)]

            # Find error
            current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
            current_coeff = current_fit[0][0]
            current_off = current_fit[0][1]
            current_err = 2 * np.sqrt(current_fit[1] / (N - 1))
            err_arr = np.append(err_arr, current_err)
            off_arr = np.append(off_arr, current_off)
            coeff_arr = np.append(coeff_arr, current_coeff)

            err_judgement_arr = np.append(err_judgement_arr, np.abs(
                prediction[ind - 1] - current_off - current_coeff * data[
                    ind - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1 / coeff_arr[err_ind]

        err = err_arr[err_ind] * fit_coeff
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind + 10) / 2)

        return err, fit_coeff, fit_offset, const_diff, fuzziness

    def find_next_inflection_ind(self, data, ind, fuzziness, is_high_peak):
        for i in range(ind+1, len(data)-1):
            last_datum = data[i-1]
            current_datum = data[i]
            next_datum = data[i+1]
            last_check = current_datum > last_datum
            next_check = current_datum > next_datum

            if (last_check == next_check) and (next_check == is_high_peak):
                if np.abs((current_datum - np.mean(data[(i-fuzziness):(i+fuzziness)]))) < 3*np.std(data[(i-fuzziness):(i+fuzziness)]):
                    return i

        return (len(data) - 1)

    def fuzzy_price(self, fit_coeff, ind, fuzziness, fit_offset):
        price = np.mean(fit_coeff * self.prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        return price

    def find_expected_value_over_single_trade(self, upper_buy, lower_buy, upper_sell, lower_sell, const_diff):
        ln_diff = (np.log(upper_buy) - np.log(lower_buy)) / const_diff
        sq_diff = ((upper_sell) ** 2 - (lower_sell) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff - 1
        return check_val

    def find_expected_value_over_many_trades(self, current_prediction, err, price_is_rising, const_diff, inflection_inds, fit_coeff, fuzziness, fit_offset):
        if price_is_rising:
            upper_buy = current_prediction + err
            lower_buy = current_prediction - err
            sell_now = False
            best_peak = np.argmax(inflection_inds)
        else:
            upper_sell = current_prediction + err
            lower_sell = current_prediction - err
            sell_now = True
            best_peak = np.argmin(inflection_inds)

        expected_return_arr = np.array([])

        for i in range(0,len(inflection_inds)):
            inflection_price = self.fuzzy_price(fit_coeff, int(inflection_inds[i]), fuzziness, fit_offset)
            if sell_now:
                upper_buy = inflection_price + err
                lower_buy = inflection_price - err
            else:
                upper_sell = inflection_price + err
                lower_sell = inflection_price - err

            current_expected_return = self.find_expected_value_over_single_trade(upper_buy, lower_buy, upper_sell, lower_sell, const_diff)
            expected_return_arr = np.append(expected_return_arr, current_expected_return)

        eval_arr = [x > 0 for x in expected_return_arr]

        if all(eval_arr):
            expected_return = 1
        else:
            expected_return = 0


        return expected_return

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        prediction = self.prediction
        data = self.data
        offset = self.offset
        price_is_rising = None
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        for i in range(offset, data_len):
            ind = i+1
            fuzzzy_counter = 0

            if ind == len(saved_inds):
                saved_inds = np.vstack((saved_inds, np.zeros((data_len + 1 - len(saved_inds), 5))))
                save_inds = True

            if save_inds:
                # TODO add the ability to increase saved length withut starting over
                if (ind%121 == 0) & (fin_table is not None):
                    # In theory this should retrain the model over predetermined intervals
                    to_date = fin_table.date[ind - 1].to_pydatetime()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    test_dates = pd.date_range(from_date, to_date, freq='1min')
                    from_ind = ind - len(test_dates)
                    fmt = '%Y-%m-%d %H:%M:%S'
                    training_fin_table = fin_table[from_ind:ind]
                    training_fin_table.index = np.arange(0, len(training_fin_table))
                    current_tz = get_current_tz()

                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' ' + current_tz,
                                            date_to=to_date.strftime(fmt) + ' ' + current_tz,
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = to_date
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_fin_table.index = np.arange(0, len(test_fin_table))
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' ' + current_tz,
                                        date_to=to_date.strftime(fmt) + ' ' + current_tz,
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    self.prediction[ind::] = prediction[(ind)::, 0]


                err, fit_coeff, fit_offset, const_diff, fuzziness = self.find_fit_info(ind)
                saved_inds[ind, 0] = err
                saved_inds[ind, 1] = fit_coeff
                saved_inds[ind, 2] = fit_offset
                saved_inds[ind, 3] = const_diff
                saved_inds[ind, 4] = fuzziness

            else:
                err = saved_inds[ind, 0]
                fit_coeff = saved_inds[ind, 1]
                fit_offset = saved_inds[ind, 2]
                const_diff = saved_inds[ind, 3]
                fuzziness = int(saved_inds[ind, 4])

            current_price = self.fuzzy_price(fit_coeff, ind, fuzziness, fit_offset)
            prior_price = self.fuzzy_price(fit_coeff, ind - 1, fuzziness, fit_offset)
            bool_price_test = current_price > prior_price
            upper_price = current_price + err
            lower_price = current_price - err

            if price_is_rising is None:
                price_is_rising = not bool_price_test

            inflection_inds = np.array([])

            current_price_is_rising = not price_is_rising
            inflection_ind = ind

            while (fuzziness + inflection_ind - ind) < 30:
                current_price_is_rising = not current_price_is_rising
                inflection_ind = self.find_next_inflection_ind(prediction, inflection_ind, fuzziness, current_price_is_rising)
                if (fuzziness + inflection_ind - ind) < 30:
                    inflection_inds = np.append(inflection_inds, inflection_ind)

            if len(inflection_inds) == 0:
                continue

            if bool_price_test != price_is_rising:  # != acts as an xor gate
                check_val = self.find_expected_value_over_many_trades(current_price, err, price_is_rising, const_diff,
                                                                      inflection_inds, fit_coeff, fuzziness, fit_offset)
                if price_is_rising:
                    # The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    # for both the buy and sell prices based on past errors
                    # both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if (check_val > 0) & (fit_coeff > 0):
                        buy_array[ind] = 1
                    elif (check_val > 0) & (fit_coeff < 0):
                        sell_array[ind] = 1

                else:
                    if (check_val > 0) & (fit_coeff > 0):
                        sell_array[ind] = 1
                    elif (check_val > 0) & (fit_coeff < 0):
                        buy_array[ind] = 1


            price_is_rising = bool_price_test

        self.buy_array = np.array([bool(x) for x in buy_array])
        self.sell_array = np.array([bool(x) for x in sell_array])
        if show_plots:
            all_times = np.arange(0, len(data))
            sell_bool = self.sell_array
            buy_bool = self.buy_array
            market_returns = 100 * (data[-1] - data[30]) / data[30]
            returns, value_over_time = find_trade_strategy_value(buy_bool[1:-1], sell_bool[1:-1], data[0:-1], return_value_over_time=True)
            plt.plot(all_times[sell_bool[0:-1]], data[sell_bool[0:-1]], 'rx')
            plt.plot(all_times[buy_bool[0:-1]], data[buy_bool[0:-1]], 'gx')
            plt.plot(data)
            plt.title( 'Return of ' + str(np.round(returns, 3)) + '% vs ' + str(np.round(market_returns, 3)) + '% Market' )

            plt.figure()
            plt.plot(value_over_time, label='Strategy')
            plt.plot(100 * data / (data[1]), label='Market')
            plt.title('Precentage Returns Strategy and Market')
            plt.ylabel('% Return')
            plt.legend()

            plt.show()

class CryptoCompare:

    comparison_symbols = ['USD']
    exchange = ''
    aggregate = 1

    def __init__(self, comparison_symbols=None, exchange=None, date_from=None, date_to=None):

        if comparison_symbols:
            self.comparison_symbols = comparison_symbols

        if exchange:
            self.exchange = exchange

        current_tz = get_current_tz()

        fmt = '%Y-%m-%d %H:%M:%S %Z'
        naive_date_from = datetime.strptime(date_from, fmt)
        sys_tz = pytz.timezone(current_tz)
        sys_tz_date_from = sys_tz.localize(naive_date_from)
        utc = pytz.UTC
        self.date_from = sys_tz_date_from.astimezone(utc)

        if date_to:
            naive_date_to = datetime.strptime(date_to, fmt)
            sys_tz_date_to = sys_tz.localize(naive_date_to)
            self.date_to = sys_tz_date_to.astimezone(utc)
        else:
            self.date_to = None

    def datedelta(self, units):
        d1_ts = time.mktime(self.date_from.timetuple())
        if self.date_to:
            d2_ts = time.mktime(self.date_to.timetuple())
        else:
            d2_ts = time.mktime(datetime.now().timetuple())

        if units == "days":
            del_date = int((d2_ts-d1_ts)/86400)
        elif units == "hours":
            del_date = int((d2_ts - d1_ts) / 3600)
        elif units == "minutes":
            del_date = int((d2_ts - d1_ts) / 60)

        return del_date

    def price(self, symbol='LTC'):
        comparison_symbols = self.comparison_symbols
        exchange = self.exchange

        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}' \
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        try:
            page = requests.get(url)
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
        data = page.json()
        return data


    def create_data_frame(self, url, symbol='LTC', return_time_stamp=False):
        try:
            page = requests.get(url)
            data = page.json()['Data']
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
            data = page.json()['Data']
        symbol = symbol.upper()
        df = pd.DataFrame(data)
        df = df.add_prefix(symbol + '_')
        df.insert(loc=0, column='date', value=[datetime.fromtimestamp(d) for d in df[symbol + '_time']])

        if return_time_stamp:
            time_stamps = df[symbol + '_time'].values
            time_stamp = time_stamps[0]
            df = df.drop(columns=[symbol + '_time'])
            return df, time_stamp

        df = df.drop(columns=[symbol + '_time']) #Drop this because the unix timestamp column is no longer needed
        return df

    def daily_price_historical(self, symbol='LTC', all_data=True, limit=1):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        if self.date_from:
            all_data=False
            limit = self.datedelta("days")

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate, int(self.date_to.timestamp()))
        else:
            url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        if all_data:
            url += '&allData=true'

        df = self.create_data_frame(url, symbol)
        return df

    def hourly_price_historical(self, symbol = 'LTC'):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("hours")

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate, int(self.date_to.timestamp()))
        else:
            url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), limit, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

        df = self.create_data_frame(url, symbol)
        return df

    def minute_price_historical(self, symbol='LTC'):
        comparison_symbol = self.comparison_symbols[0]
        exchange = self.exchange
        limit = self.datedelta("minutes")
        first_lim = limit
        minute_lim = 2000

        if limit > minute_lim:
            first_lim = minute_lim

        if self.date_to:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate, self.date_to.timestamp())
        else:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
                .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate)
        temp_url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}' \
            .format(symbol.upper(), comparison_symbol.upper(), first_lim, self.aggregate)
        if exchange:
            url += '&e={}'.format(exchange)

        loop_len = int(np.ceil(limit / minute_lim))
        if limit > minute_lim: # This if statement is to allow the gathering of historical minute data beyond 2000 points (the limit)
            df, time_stamp = self.create_data_frame(url, symbol, return_time_stamp=True)
            for num in range(1, loop_len):
                toTs = time_stamp - 60 # have to subtract a value of 60 had to be added to avoid repeated indices
                url_new = temp_url + '&toTs={}'.format(toTs)
                if num == (loop_len - 1):
                    url_new = 'https://min-api.CryptoCompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}' \
                        .format(symbol.upper(), comparison_symbol.upper(), limit - num * minute_lim, self.aggregate, toTs)
                df_to_append, time_stamp = self.create_data_frame(url_new, symbol, return_time_stamp=True)
                df = df_to_append.append(df, ignore_index=True) # The earliest data go on top
            return df

        df = self.create_data_frame(url, symbol)
        return df

    def coin_list(self):
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def coin_snapshot_full_by_id(self, symbol='LTC', symbol_id_dict={}):#TODO fix the id argument mutability

        if not symbol_id_dict:
            symbol_id_dict = {
                'BTC': 1182,
                'ETH': 7605,
                'LTC': 3808
            }
        symbol_id = symbol_id_dict[symbol.upper()]
        url = 'https://www.cryptocompare.com/api/data/coinsnapshotfullbyid/?id={}' \
            .format(symbol_id)
        try:
            page = requests.get(url)
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
        data = page.json()['Data']
        return data

    def live_social_status(self, symbol, symbol_id_dict=None):

        if not symbol_id_dict:
            symbol_id_dict = {
                'BTC': 1182,
                'ETH': 7605,
                'LTC': 3808
            }
        symbol_id = symbol_id_dict[symbol.upper()]
        url = 'https://www.cryptocompare.com/api/data/socialstats/?id={}' \
            .format(symbol_id)
        page = requests.get(url)
        data = page.json()['Data']
        return data

    def news(self, symbol, date_before=None):
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        current_tz = get_current_tz()

        naive_date_before = datetime.strptime(date_before, fmt)
        sys_tz = pytz.timezone(current_tz)
        sys_tz_date_before = sys_tz.localize(naive_date_before)
        utc = pytz.UTC
        date_before = sys_tz_date_before.astimezone(utc)
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={},BTC,Regulation,Altcoin,Blockchain,Mining,Trading,Market&lTs={}" \
        .format(symbol.upper(), int(date_before.timestamp()))
        try:
            page = requests.get(url)
            data = page.json()['Data']
        except:
            print('suspected timeout, taking a 1 minute break')
            time.sleep(120)
            page = requests.get(url)
            data = page.json()['Data']
        return data

class DataSet:
    prediction_length=1

    def __init__(self, date_from, date_to, prediction_length=None, bitinfo_list = None, prediction_ticker ='ltc', time_units='hours', fin_table=None, aggregate=1, news_hourly_offset=5):
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        self.bitinfo_list = bitinfo_list
        cryp_obj = CryptoCompare(date_from=date_from, date_to=date_to)
        cryp_obj.aggregate = aggregate
        self.cryp_obj = cryp_obj

        temp_fin_table = fin_table
        #This adds the price data from the bitinfo_list currencies to the DataFrame
        sym = bitinfo_list[0]

        if temp_fin_table is not None:
            self.fin_table = temp_fin_table
        else:
            if time_units == 'days':
                fin_table = cryp_obj.daily_price_historical(symbol=sym)
                price_func = lambda symbol: cryp_obj.daily_price_historical(symbol=symbol)
            elif time_units == 'hours':
                fin_table = cryp_obj.hourly_price_historical(symbol=sym)
                price_func = lambda symbol: cryp_obj.hourly_price_historical(symbol=symbol)
            else:
                fin_table = cryp_obj.minute_price_historical(symbol=sym)
                price_func = lambda symbol: cryp_obj.minute_price_historical(symbol=symbol)

            self.price_func = price_func

            for num in range(1,len(bitinfo_list)):
                #cryp_obj.symbol = sym
                sym = bitinfo_list[num]
                temp_table = price_func(symbol=sym)
                temp_table = temp_table.drop(columns='date')
                fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

            #This section adds the news data
            total_len = len(fin_table)
            news_sentiment = []
            news_count = []
            iterations_complete = 0

            rate_limit_url = 'https://min-api.cryptocompare.com/stats/rate/limit'
            # try:
            #     page = requests.get(rate_limit_url)
            # except:
            #     print('suspected timeout, taking a 1 minute break')
            #     time.sleep(120)
            #     page = requests.get(rate_limit_url)
            date_len = len(fin_table.date.values)
            last_news = None

            n = 4500
            for i in range(1, date_len + 1):
                ind = date_len - i
                current_dt = fin_table.date.values[ind]
                current_dt = pd.to_datetime(current_dt)
                utc_current_dt = convert_time_to_uct(current_dt)
                delta_ts = utc_current_dt.timestamp() - news_hourly_offset * 3600
                current_ts = utc_current_dt.timestamp()
                # if page.json() is not None:
                #     current_minute_limit = page.json()['Minute']['CallsLeft']['News']
                #     current_hour_limit = page.json()['Hour']['CallsLeft']['News']
                # else:
                #     current_minute_limit = 1
                # if current_minute_limit < 2:
                #     time.sleep(60)
                # if current_hour_limit < 2:
                #     while current_hour_limit < 2:
                #         time.sleep(60)
                #         current_hour_limit = page.json()['Hour']['CallsLeft']['News']

                if last_news is not None:
                    last_news_publication_times = [news['published_on'] < current_ts for news in last_news]
                    if all(last_news_publication_times):
                        current_news = last_news
                    else:
                        current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' UTC')
                else:
                    current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' UTC')

                last_news = current_news


                current_full_sentiment = [n*(txb(news['title']).sentiment.polarity)/(n + current_ts - news['published_on']) for news in current_news]

                weighted_news_count = np.sum([n*(news['published_on'] > delta_ts)/(n + current_ts - news['published_on']) for news in current_news])
                news_count.insert(0, weighted_news_count)

                current_news_count = np.sum([(news['published_on'] > delta_ts) for news in current_news])
                current_sentiment = current_full_sentiment[0:current_news_count]

                sentiment_sum = np.mean(current_sentiment)
                news_sentiment.insert(0, sentiment_sum)

                iterations_complete += 1
                if total_len > 30:
                    print('news scraping ' + str(round(100 * iterations_complete / total_len, 2)) + '% complete')

            temp_table = pd.DataFrame({'Sentiment': news_sentiment, 'News Frequency': news_count}, index=fin_table.index)
            fin_table = pd.concat([fin_table, temp_table], axis=1, join_axes=[temp_table.index])

            #This section adds the relevat data to the DataSet
            self.fin_table = fin_table
        self.prediction_ticker = prediction_ticker
        self.date_to = date_to
        self.date_from = date_from
        if prediction_length is not None:
            self.prediction_length = prediction_length
        self.time_units = time_units

    def create_price_prediction_columns(self):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker

        prediction_table = pd.DataFrame(data=self.fin_table[sym.upper()+'_high'].values, columns=[sym.upper()+'_high Prediction'])

        #prediction_table = temp_prediction_table.drop(columns=['date', sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_open', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])

        if np.count_nonzero(prediction_table.values) != len(prediction_table.values):
            raise ValueError('Prediction table should only contain non zero values. This table has ' + str(np.count_nonzero(prediction_table.values)) + ' zeros in it. Check the cryptocompare servers.')

        fin_table = pd.concat([self.fin_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        data_frame = fin_table.set_index('date')
        self.final_table = data_frame[(data_frame.index <= self.date_to)]

    def create_difference_prediction_columns(self):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        price_table = self.fin_table
        #price_table = price_table.drop(columns=['date', sym.upper() + '_low', sym.upper() + '_high', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])
        close_values = price_table[sym.upper() + '_close'].values
        open_values = price_table[sym.upper() + '_open'].values
        del_values = close_values - open_values
        prediction_table = pd.DataFrame(data=del_values, index=self.fin_table.index)
        fin_table = pd.concat([self.fin_table, prediction_table], axis=1, join_axes=[prediction_table.index])
        data_frame = fin_table.set_index('date')
        self.final_table = data_frame[(data_frame.index <= self.date_to)]

    def create_binary_classification_for_strategy(self, prices):
        buy_array = np.zeros((len(prices), 1))
        sell_array = np.zeros((len(prices), 1))
        for i in range(0, len(prices)-1):
            if prices[i+1] > prices[i]:
                buy_array[i] = 1
            elif prices[i+1] < prices[i]:
                sell_array[i] = 1

        return sell_array, buy_array

    def create_buy_sell_prediction_frame(self, m):
        cryp_obj = self.cryp_obj
        cryp_obj.symbol = self.prediction_ticker
        sym = self.prediction_ticker
        price_data_frame = pd.DataFrame(data=self.fin_table[sym.upper()+'_high'])
        #price_data_frame = price_data_frame.drop(columns=[sym.upper() + '_close', sym.upper() + '_low', sym.upper() + '_open', sym.upper() + '_volumefrom', sym.upper() + '_volumeto'])
        #price_data_frame = price_data_frame.set_index('date')

        sell_array, buy_array = self.create_binary_classification_for_strategy(price_data_frame.values)


        prediction_frame = pd.DataFrame(data=np.hstack((buy_array, sell_array)), index=self.final_table.index, columns=['Buy', 'Sell'])

        #TODO check on this
        self.create_price_prediction_columns()

        self.final_table = pd.concat([self.fin_table.set_index('date'), prediction_frame], axis=1, join_axes=[prediction_frame.index])

    def create_arrays(self, model_type='price', min_distance_between_trades=5):
        if model_type == 'price':
            self.create_price_prediction_columns()
            n = -1
            m = n
        elif model_type == 'difference':
            self.create_difference_prediction_columns()
            n = -1
            m = n
        elif model_type == 'buy&sell':
            self.create_buy_sell_prediction_frame(min_distance_between_trades)
            n = -1 #This should be -2 for two columns
            m = n #This should be -3 for two columns

        temp_input_table = self.final_table#.drop(columns='date')
        fullArr = temp_input_table.values
        temp_array = fullArr[np.logical_not(np.isnan(np.sum(fullArr, axis=1))), ::]
        self.output_array = np.array(temp_array[self.prediction_length:-1, n:])
        temp_input_array = np.array(temp_array[0:-(self.prediction_length + 1), 0:m])
        self.input_array_times = temp_input_table.index[0:-(self.prediction_length + 1)]
        scaler = StandardScaler()
        temp_input_array = scaler.fit_transform(temp_input_array)
        self.input_array = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)

    def create_prediction_arrays(self):
        data_frame = self.fin_table.set_index('date')
        temp_input_array = data_frame.values
        scaler = StandardScaler()
        temp_array = temp_input_array[np.logical_not(np.isnan(np.sum(temp_input_array, axis=1))), ::]
        temp_array = scaler.fit_transform(temp_array)
        self.input_array = temp_array.reshape(temp_array.shape[0], temp_array.shape[1], 1)
        self.output_array = data_frame[self.prediction_ticker.upper() + '_high'].values
        self.final_table=data_frame

    def add_data(self, date_to, retain_length=False):
        # TODO add ability to change prediction length
        time_del = timedelta(minutes=1)
        current_tz = get_current_tz()
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        fmt_sans_tz = '%Y-%m-%d %H:%M:%S '
        from_datetime = datetime.strptime(self.date_to, fmt) + time_del
        date_from = datetime.strftime(from_datetime, fmt_sans_tz) + current_tz
        if date_from == date_to:
            from_datetime = datetime.strptime(self.date_to, fmt)
            date_from = datetime.strftime(from_datetime, fmt_sans_tz) + current_tz
            old_fin_table = self.fin_table
            temp_data_obj = DataSet(date_from, date_to, prediction_length=self.prediction_length,
                                    bitinfo_list=self.bitinfo_list, prediction_ticker=self.prediction_ticker,
                                    time_units=self.time_units)
            fin_table_addition = temp_data_obj.fin_table
            fin_table_addition.index = fin_table_addition.index + np.max(old_fin_table.index.values) + 1
            new_fin_table = old_fin_table.append(fin_table_addition.iloc[-1])

        else:
            old_fin_table = self.fin_table
            temp_data_obj = DataSet(date_from, date_to, prediction_length=self.prediction_length, bitinfo_list=self.bitinfo_list, prediction_ticker=self.prediction_ticker, time_units=self.time_units)
            fin_table_addition = temp_data_obj.fin_table
            fin_table_addition.index = fin_table_addition.index + np.max(old_fin_table.index.values) + 1
            new_fin_table = old_fin_table.append(fin_table_addition)

        if retain_length:
            self.fin_table = new_fin_table.iloc[len(fin_table_addition)::]
        else:
            self.fin_table = new_fin_table

        self.date_to = date_to

class CoinPriceModel:

    model = None
    training_array_input = None
    training_array_output = None
    data_obj = None
    prediction_length = 1
    activation_func=None
    optimization_scheme="adam"
    loss_func="mean_absolute_percentage_error"
    epochs = None
    bitinfo_list = None
    google_list = None
    pred_data_obj = None

    def __init__(self, date_from, date_to, model_path=None, days=None, bitinfo_list=None, google_list=None,
                 prediction_ticker='ltc', epochs=500, activ_func='relu', time_units='hours', is_leakyrelu=True, need_data_obj=True, data_set_path=None, aggregate_val=1):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        if bitinfo_list is None:
            bitinfo_list = ['btc', 'eth']
        if google_list is None:
            google_list = ['Litecoin']
        if days is not None:
            self.prediction_length = days
        self.prediction_ticker = prediction_ticker

        if need_data_obj:
            if data_set_path:
                with open(data_set_path, 'rb') as ds_file:
                    saved_table = pickle.load(ds_file)

                fmt = '%Y-%m-%d %H:%M:%S %Z'
                date_from_object = datetime.strptime(date_from, fmt)
                date_to_object = datetime.strptime(date_to, fmt)
                dates_list = saved_table.date

                start_ind = (dates_list == date_from_object).argmax()
                stop_ind = (dates_list == date_to_object).argmax() + 1

                saved_table = saved_table[start_ind:stop_ind]
                saved_table.index = np.arange(len(saved_table))

            else:
                saved_table=None\

            self.data_obj = DataSet(date_from=date_from, date_to=date_to, prediction_length=self.prediction_length,
                                    bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, fin_table=saved_table, aggregate=aggregate_val)
            if saved_table is None:
                table_file_name = '_' + time_units + '_from_' + date_from + '_to_' + date_to + '.pickle'
                table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet' + table_file_name.replace(
                    ' ', '_')
                with open(table_file_name, 'wb') as cp_file_handle:
                    pickle.dump(self.data_obj.fin_table, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)


        self.epochs = epochs
        self.bitinfo_list = bitinfo_list
        self.google_list = google_list
        self.activation_func = activ_func
        self.is_leakyrelu=is_leakyrelu

    def build_model(self, inputs, neurons, output_size=1, dropout=0.25, layer_count=3):  #TODO make output_size someing editable outside the class
        is_leaky = self.is_leakyrelu
        activ_func = self.activation_func
        loss = self.loss_func
        optimizer = self.optimization_scheme
        self.model = Sequential()

        self.model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        self.model.add(Dropout(dropout))

        if is_leaky:
            for i in range(0, layer_count):
                self.model.add(Dense(units=neurons, activation="linear", kernel_initializer='normal'))
                self.model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, layer_count):
                self.model.add(Dense(units=neurons, activation=activ_func, kernel_initializer='normal'))

        self.model.add(Dense(units=output_size, activation="linear"))
        self.model.compile(loss=loss, optimizer=optimizer)

    def create_arrays(self, min_distance_between_trades=5, model_type='price'):
        self.data_obj.create_arrays(model_type, min_distance_between_trades)
        training_length = (int(2*len(self.data_obj.input_array)/3))

        self.input = self.data_obj.input_array
        self.output = self.data_obj.output_array
        self.times = self.data_obj.input_array_times

        self.training_array_input = self.data_obj.input_array[0:training_length]
        self.training_array_output = self.data_obj.output_array[0:training_length]
        self.training_times = self.data_obj.input_array_times[0:training_length]

        self.test_array_input = self.data_obj.input_array[training_length::]
        self.test_array_output = self.data_obj.output_array[training_length::]
        self.test_times = self.data_obj.input_array_times[training_length::]

    #TODO create method to save Data Frames without testing the models

    def train_model(self, neuron_count=200, min_distance_between_trades=5, model_type='price', save_model=False, train_saved_model=False, layers=3, batch_size=96 ):
        self.create_arrays(min_distance_between_trades, model_type=model_type)
        if train_saved_model:
            print('re-trianing model')
            self.model.reset_states()
        elif model_type == 'buy&sell':
            self.build_model(self.training_array_input, neurons=neuron_count, output_size=1, layer_count=layers)
        else:
            self.build_model(self.training_array_input, neurons=neuron_count, layer_count=layers)

        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        hist = self.model.fit(self.training_array_input, self.training_array_output, epochs=self.epochs,
                                  batch_size=batch_size, verbose=2,
                                  shuffle=False, validation_split=0.25, callbacks=[estop])

        if self.is_leakyrelu & save_model: #TODO add more detail to saves
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/' + str(layers) + '_Layers/' + self.prediction_ticker + 'model_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + 'leakyreluact_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_'+ str(neuron_count) + 'neurons_' + str(np.max(hist.epoch)) +'epochs' + str(datetime.now().timestamp()) + '.h5')

        elif save_model:
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/' + str(layers) + '_Layers/' + self.prediction_ticker + 'model_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss_' + str(neuron_count) + 'neurons_' + str(np.max(hist.epoch)) +'epochs_' + str(layers) + 'layers' + str(datetime.now().timestamp()) + '.h5')

        return hist

    def update_model_training(self):
        #This is for live model weight updates
        self.create_arrays(5, model_type='price')
        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        self.model.reset_states()
        self.model.fit(self.input, self.output, epochs=self.epochs, batch_size=96, verbose=2,
                              shuffle=False, validation_split=0.25, callbacks=[estop])

    def test_model(self, did_train=True, show_plots=True, min_distance_between_trades=0, model_type='price', save_model = False):
        if did_train:
            test_input = self.test_array_input
            test_output = self.test_array_output
            test_times = self.test_times
        else:
            self.create_arrays(min_distance_between_trades, model_type=model_type)
            test_input = self.input
            test_output = self.output
            test_times = self.times

        prediction = self.model.predict(test_input)

        absolute_output =test_output[::, 0]
        zerod_output = absolute_output - np.mean(test_output[::, 0])
        zerod_prediction =  clean_data(prediction[::, 0] - np.mean(prediction[::, 0]))

        if save_model:
            self.model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Updated_Models/' + self.prediction_ticker + 'model_' + str(
                self.prediction_length) + self.data_obj.time_units + '_' + self.activation_func + 'act_' + self.optimization_scheme + 'opt_' + self.loss_func + 'loss' + str(
                datetime.now().timestamp()) + '.h5')

        if show_plots:
            N = 3
            x = np.convolve(zerod_prediction, np.ones((N,)) / N)[N-1::]
            x = x.reshape(1, len(x))
            x = x.T

            sell_bool, buy_bool = find_optimal_trade_strategy(zerod_prediction, min_price_jump=1)
            buy_times = test_times[buy_bool]
            sell_times = test_times[sell_bool]
            buy_prices = absolute_output[buy_bool]
            sell_prices = absolute_output[sell_bool]

            #plot the prediction
            zerod_output = zerod_output.reshape(1, len(zerod_output))
            zerod_output = zerod_output.T
            zerod_prediction = zerod_prediction.reshape(1, len(zerod_prediction))
            zerod_prediction = zerod_prediction.T
            plot_df = pd.DataFrame(data=np.hstack((zerod_prediction/(np.max(zerod_prediction)), zerod_output/(np.max(zerod_output)))), index=test_times, columns=['Predicted', 'Measured'])
            ax = plot_df.plot(y='Measured', style='bo--')
            plot_df.plot(y='Predicted', style='rx--', ax=ax)

            plt.title('Prediction')
            plt.xlabel('Date/Time')
            plt.ylabel('Normalized Price')
            plt.figure()

            #plot the correlation
            plt.plot(zerod_output / (np.max(zerod_output)), zerod_prediction / (np.max(zerod_prediction)), 'b.')
            plt.xlabel('measured price')
            plt.ylabel('predicted price')

            #plot the returns
            sell_plot_df = pd.DataFrame(data=sell_prices, index=sell_times, columns=['Sell'])
            buy_plot_df = pd.DataFrame(data=buy_prices, index=buy_times, columns=['Buy'])
            price_plot_df = pd.DataFrame(data=test_output, index=test_times, columns=['Price'])
            ax = sell_plot_df.plot(style='rx')
            buy_plot_df.plot(style='gx', ax=ax)
            price_plot_df.plot(ax=ax)
            strategy_value = find_trade_strategy_value(buy_bool, sell_bool, absolute_output)
            strategy_returns = strategy_value
            market_returns = 100*(absolute_output[-1] - absolute_output[0])/absolute_output[0]

            plt.title('Strategy with ' + str(np.round(strategy_returns, 3)) + '% Returns vs ' + str(np.round(market_returns, 3)) + '% Market')
            plt.xlabel('Date/Time')
            plt.ylabel('Price ($)')

            plt.show()
        else:
            return prediction, test_output

    def create_standard_dates(self):
        to_date = datetime.now()
        return to_date

    def predict(self, time_units='hours', show_plots=True, old_prediction=np.array([]), is_first_prediction=True):
        fmt = '%Y-%m-%d %H:%M'
        current_tz = get_current_tz()
        date_str_add_on = ':00 ' + current_tz
        to_date = datetime.now()
        if time_units == 'minutes':
            delta = timedelta(minutes=self.prediction_length)
            from_delta = timedelta(hours=2)
        elif time_units == 'hours':
            delta = timedelta(hours=self.prediction_length)
            from_delta = timedelta(days=3)
        elif time_units == 'days':
            delta = timedelta(days=self.prediction_length)
            from_delta = timedelta(days=30)

        date_to_str = to_date.strftime(fmt) + date_str_add_on
        if self.pred_data_obj is None:
            from_date = to_date - from_delta
            date_from_str = from_date.strftime(fmt) + date_str_add_on
            test_data = DataSet(date_from=date_from_str, date_to=date_to_str, prediction_length=self.prediction_length, bitinfo_list=self.bitinfo_list,
                                prediction_ticker=self.prediction_ticker, time_units=time_units)
        else:
            # TODO add ability to change prediction length in case its not one minute
            test_data = self.pred_data_obj
            test_data.add_data(date_to_str)

        self.pred_data_obj = test_data

        test_data.create_prediction_arrays()
        prediction_input = test_data.input_array #do not use the create array methods here because the output is not needed
        if is_first_prediction:
            prediction = self.model.predict(prediction_input)
        else:
            prediction = self.model.predict(prediction_input[len(old_prediction)::, ::, ::])

        price_array = test_data.output_array
        self.data_obj = test_data

        columstr = 'Predicted ' + time_units
        prediction_table = pd.DataFrame({columstr: np.append(old_prediction, prediction[::,0])}, index=test_data.final_table.index + delta)
        price_table = pd.DataFrame({'Current': price_array},index=test_data.final_table.index)

        if show_plots:
            ax1 = prediction_table.plot(style='rx--')
            price_table.plot(style='bo--', ax=ax1)
            plt.title(self.prediction_ticker.upper() + ' ' + str(self.prediction_length) + ' ' + time_units + ' Prediction')
            plt.show()
        else:
            return prediction_table, price_table

#TODO make optimizer for trade strategy
#TODO play with price models to see which gives best strategy
class CryptoTradeStrategyModel(CoinPriceModel):

    strategy_loss_fun = 'binary_crossentropy'

    strategy_is_leakyrelu = True

    #TODO move the create methods to the DataSet class

    def create_test_price_columns(self, should_train=False, min_distance_between_trades=5, n=30):
        #TODO delete extraneous code
        #This creates a table with 2*n columns that contains n columns of the price for the past n units of time and prediction for the next n units. This is meant to train the strategy model

        if should_train:
            val_loss = 30
            iter = 0
            #np.random.seed(7)  # for reproducibility
            while (val_loss > 10) & (iter < 20):
                hist = self.train_model(model_type='price', neuron_count=100, batch_size=32)
                val_loss = hist.history['val_loss'][-1]
                iter += 1
                if val_loss > 10:
                    print('val_loss is too great, retraining')
                else:
                    print('valid model detected')

            prediction, price = self.test_model(show_plots=False, did_train=False)
        else:
            self.create_arrays(min_distance_between_trades, model_type='price')
            prediction, price = self.test_model(did_train=False, show_plots=False, min_distance_between_trades=min_distance_between_trades, model_type='price')

        all_times = self.times

        return prediction[::, 0], price[::, 0], all_times

    def find_fit_info(self, prediction, data, ind, offset=40):
        #This method searches the past data to determine what value should be used for the error
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr  # this array will contain the residual from the prior datum

        for N in range(10, offset):
            past_predictions = prediction[(ind - N):(ind)]
            past_data = data[(ind - N):(ind)]

            # Find error
            current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
            current_coeff = current_fit[0][0]
            current_off = current_fit[0][1]
            current_err = 2 * np.sqrt(current_fit[1] / (N - 1))
            err_arr = np.append(err_arr, current_err)
            off_arr = np.append(off_arr, current_off)
            coeff_arr = np.append(coeff_arr, current_coeff)

            err_judgement_arr = np.append(err_judgement_arr, np.abs(
                prediction[ind - 1] - current_off - current_coeff * data[
                    ind - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1 / coeff_arr[err_ind]

        err = np.abs(err_arr[err_ind] * fit_coeff) # The error must always be positive
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind + 10) / 2)  # TODO make more logical fuzziness

        return err, fit_coeff, fit_offset, const_diff, fuzziness

    def create_strategy_input_frame(self, price, predicted_price, full_times):
        input_frame = None
        times = full_times[40:(len(price)-31)]
        self.price = price[40:(len(price)-31)]
        all_i = len(range(40, len(price)-31))
        last_num = 0
        for i in range(40, len(price)-31):
            num = 100*(i-40)/((all_i-40))
            if (num - last_num) > 0.01:
                last_num = num
                print(num2str(100*(i-40)/((all_i-40)), 2)+'%')
            err, fit_coeff, fit_offset, const_diff, fuzziness = self.find_fit_info(predicted_price, price[0:-30], i)
            this_row = np.array([err, fit_coeff, fit_offset, const_diff, fuzziness])
            if input_frame is not None:
                input_frame = np.vstack((input_frame, this_row))
            else:
                input_frame = this_row

        input_frame = np.hstack((input_frame, self.price.reshape(len(self.price), 1), predicted_price[71::].reshape(len(self.price), 1)))

        return input_frame, times

    def create_strategy_prediction_frame(self, n, min_distance_between_trades=5, show_plots=False): #Set show_plots to True for debug only
        predicted_price, price, full_time = self.create_test_price_columns(n=n)
        self.data_obj.create_price_prediction_columns()
        self.data_obj.create_buy_sell_prediction_frame(min_distance_between_trades)
        input_for_model, price_times = self.create_strategy_input_frame(price, predicted_price, full_time)
        strategy_input_frame = pd.DataFrame(data=input_for_model, index=price_times)

        sym = self.data_obj.prediction_ticker.upper()
        buy_frame = pd.DataFrame(self.data_obj.final_table['Buy'])#.drop(columns=['Sell'])
        sell_frame = pd.DataFrame(self.data_obj.final_table['Sell'])#.drop(columns=['Buy'])

        #TODO delete extraneous code
        buy_strategy_frame = pd.merge(strategy_input_frame, buy_frame[40:(len(price)-31)], left_index=True, right_index=True)
        sell_strategy_frame = pd.merge( strategy_input_frame, sell_frame[40:(len(price)-31)], left_index=True, right_index=True)

        self.strategy_frame = pd.concat([strategy_input_frame, buy_frame[40:(len(price)-31)], sell_frame[40:(len(price)-31)]], axis=1, join_axes=[strategy_input_frame.index])

        if show_plots:
            ax1 = sell_strategy_frame[self.prediction_ticker.upper() + '_high'].plot(style='b--')
            buy_strategy_frame[self.prediction_ticker.upper() + '_high'][buy_strategy_frame['Buy'].values == 1].plot(style='gx', ax=ax1)
            sell_strategy_frame[self.prediction_ticker.upper() + '_high'][sell_strategy_frame['Sell'].values == 1].plot(style='rx', ax=ax1)
            plt.show()
        return buy_strategy_frame, sell_strategy_frame

    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        loss_val = true_positives/(possible_positives + K.epsilon())
        return loss_val

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.clip((1 - y_true) * (1 - y_pred), 0, 1))
        possible_negatives = K.sum(K.clip(1 - y_true, 0, 1))
        loss_val = true_negatives / (possible_negatives + K.epsilon())

        return loss_val

    def custom_sell_loss_func(self, y_true, y_pred):
        price_data = self.data_obj.fin_table[self.prediction_ticker.upper()+'_high'].values
        predicted_trades = [x > K.mean(y_pred) for x in y_pred]
        true_trades = [bool(x) for x in y_true]

        sell_bool, buy_bool = find_optimal_trade_strategy(price_data)
        true_val = find_trade_strategy_value(buy_bool, sell_bool, price_data)
        predicted_val = find_trade_strategy_value(buy_bool, predicted_trades, price_data)

        loss_val = predicted_val/true_val
        return loss_val

    def custom_buy_loss_func(self, y_true, y_pred):
        price_data = self.data_obj.fin_table[self.prediction_ticker.upper()+'_high'].values
        predicted_trades = [x > K.mean(y_pred) for x in y_pred]
        true_trades = [bool(x) for x in y_true]

        sell_bool, buy_bool = find_optimal_trade_strategy(price_data)
        true_val = find_trade_strategy_value(buy_bool, sell_bool, price_data)
        predicted_val = find_trade_strategy_value(predicted_trades, predicted_trades, price_data)

        loss_val = predicted_val/true_val
        return loss_val

    #TODO make model methods more modular for simpler integration of new models (like the strategy model)
    def build_strategy_model(self, inputs, neurons, strategy_type, strategy_activ_func = 'tanh', output_size=1, dropout=0, layers=3, final_activation = 'sigmoid'):
        is_leaky = self.strategy_is_leakyrelu
        activ_func = strategy_activ_func
        strategy_model = Sequential()

        strategy_model.add(LSTM(1, input_shape=(inputs.shape[1], inputs.shape[2])))
        strategy_model.add(Dropout(dropout))

        if is_leaky:
            for i in range(0, layers):
                strategy_model.add(Dense(units=neurons, activation="linear", kernel_initializer='normal'))
                strategy_model.add(LeakyReLU(alpha=0.1))
        else:
            for i in range(0, layers):
                strategy_model.add(Dense(units=neurons, activation=activ_func, kernel_initializer='normal'))

        strategy_model.add(Dense(units=output_size, activation=final_activation))

        strategy_model.compile(loss=self.strategy_loss_fun, optimizer=keras.optimizers.adam(lr=0.001))

        return strategy_model

    def prep_arrays_for_model(self, arr, training_len, should_reshape):

        if should_reshape:
            scaler = StandardScaler()
            temp_input_array = scaler.fit_transform(arr)
            output_arr = temp_input_array.reshape(temp_input_array.shape[0], temp_input_array.shape[1], 1)
        else:
            output_arr = arr

        training_output = output_arr[0:training_len]
        test_output = output_arr[training_len::]

        return training_output, test_output

    def train_strategy_model(self, sell_neuron_count=20, buy_neuron_count=20, min_distance_between_trades=5, save_model=False, t = 0.5):
        buy_frame, sell_frame = self.create_strategy_prediction_frame(min_distance_between_trades=min_distance_between_trades, n=5)

        buy_values = buy_frame['Buy'].values
        sell_values = sell_frame['Sell'].values

        n_trades = np.sum(buy_values)
        n_holds = len(buy_values) - n_trades
        class_weight = {
            1: (n_holds/n_trades)*t,
            0: (n_holds/n_trades)*(1 - t)
        }#This weights the trades because they are few relative to the holds

        input_values = buy_frame.drop(columns=['Buy'])

        training_length = (int(2 * len(buy_values) / 3))

        #np.random.seed(7)  # for reproducibility
        training_input, test_input = self.prep_arrays_for_model(input_values, training_length, should_reshape=True)
        training_buy_output, test_buy_output = self.prep_arrays_for_model(sell_values, training_length, should_reshape=False)
        training_sell_output, test_sell_output = self.prep_arrays_for_model(buy_values, training_length,should_reshape=False)



        self.buy_model = self.build_strategy_model(training_input, neurons=buy_neuron_count, strategy_type='buy', layers=1, strategy_activ_func='soft_max', final_activation='sigmoid')
        self.sell_model = self.build_strategy_model(training_input, neurons=sell_neuron_count, strategy_type='sell', layers=1, strategy_activ_func='soft_max', final_activation='sigmoid')

        estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=0, verbose=0, mode='auto')

        self.buy_model.fit(training_input, training_buy_output, epochs=500, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])#, class_weight=class_weight)

        self.sell_model.fit(training_input, training_sell_output, epochs=500, batch_size=32, verbose=2,
                                    shuffle=False, validation_split=0.25, callbacks=[estop])#, class_weight=class_weight)

        self.buy_model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/buy_sell/buy_model_' + str(buy_neuron_count) + '_neurons')
        self.sell_model.save('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/buy_sell/sell_model_' + str(buy_neuron_count) + '_neurons')
        test_prices = self.price[training_length::]

        return test_buy_output, test_input, test_sell_output, test_prices

    def test_strategy_model(self, did_train=True, show_plots=True, min_distance_between_trades=5, sell_neurons=10, buy_neurons=10):
        # if did_train:
        #     test_input = self.test_array_input
        #     test_output = self.test_array_output
        # else:
        #     self.create_arrays(min_distance_between_trades, model_type=model_type)
        #     test_input = self.input
        #     test_output = self.output

        test_buy_output, test_input, test_sell_output, price = self.train_strategy_model(sell_neuron_count=sell_neurons, buy_neuron_count=buy_neurons)


        if show_plots:

            buy_prediction = self.buy_model.predict(test_input)
            sell_prediction = self.sell_model.predict(test_input)

            buy_frame = pd.DataFrame(self.strategy_frame['Buy'][(-len(buy_prediction))::])
            sell_frame = pd.DataFrame(self.strategy_frame['Sell'][(-len(buy_prediction))::])
            zerod_price = price - np.min(price)
            scaled_price = zerod_price / np.max(zerod_price)
            buy_frame['Price'] = pd.Series(scaled_price[(-len(buy_prediction))::], index=buy_frame.index[(-len(buy_prediction))::])
            buy_frame['Prediction'] = pd.Series(buy_prediction[::, 0], index=buy_frame.index[(-len(buy_prediction))::])
            sell_frame['Price'] = buy_frame['Price']
            sell_frame['Prediction'] = pd.Series(sell_prediction[::, 0],
                                                 index=buy_frame.index[(-len(sell_prediction))::])

            sell_frame.plot()
            buy_frame.plot()

            plt.figure()
            plt.title('Sell')
            plt.plot(sell_prediction, 'rx--')
            plt.plot(test_sell_output, 'bx--')

            plt.figure()
            plt.title('Buy')
            plt.plot(buy_prediction, 'rx--')
            plt.plot(test_buy_output, 'bx--')

            plt.show()

            test_buys = buy_prediction > sell_prediction
            test_sells = sell_prediction > buy_prediction

            market_returns = 100*(price[-1] - price[0])/price[0]
            algorithm_returns = 100*find_trade_strategy_value(test_buys[::, 0], test_sells[::, 0], price)

            print('Algorithm returns ' + num2str(algorithm_returns, 3) + ' and market returns are ' + num2str(market_returns, 3))
        else:
            buy_loss = self.buy_model.evaluate(test_input, test_buy_output)
            sell_loss = self.sell_model.evaluate(test_input, test_sell_output)

            return buy_loss, sell_loss

class BaseTradingBot:

    image_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Images//'
    last_true_countdown = 2

    def __init__(self, hourly_model, minute_model, hourly_len=6, minute_len=15, prediction_ticker='ETH', bitinfo_list = ['eth'], is_hourly_prediction_needed=True):
        temp = "2018-05-05 00:00:00 EST"

        if is_hourly_prediction_needed:
            self.hourly_cp = CoinPriceModel(temp, temp, days=hourly_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='hours', model_path=hourly_model, need_data_obj=False)

        self.minute_cp = CoinPriceModel(temp, temp, days=minute_len, prediction_ticker=prediction_ticker,
                                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=minute_model, need_data_obj=False)

        self.hour_length = hourly_len
        self.minute_length = minute_len
        self.prediction_ticker = prediction_ticker.upper()
        self.minute_prediction = None
        self.minute_price = None

    def find_data(self, is_hourly_prediction_needed = True):

        if self.minute_price is None:
            full_minute_prediction, full_minute_price = self.minute_cp.predict(time_units='minutes', show_plots=False)
        else:
            full_minute_prediction, full_minute_price = self.minute_cp.predict(time_units='minutes', show_plots=False, old_prediction=self.minute_prediction.values[::, 0], is_first_prediction=False)

        self.minute_prediction = full_minute_prediction
        self.minute_price = full_minute_price

        if is_hourly_prediction_needed:
            full_hourly_prediction, full_hourly_price = self.hourly_cp.predict(time_units='hours', show_plots=False)
            self.hourly_prediction = full_hourly_prediction[-(self.hour_length + 4)::]
            self.hourly_price = full_hourly_price[-4::]

    def prepare_images(self):
        #This method creates, saves, and closes the figures
        #Create minute by minute image
        minute_ax = self.minute_prediction.plot(style='rx--')
        self.minute_price.plot(style='bo--', ax=minute_ax)
        minute_fig_title = self.prediction_ticker.upper() + ' ' + str(self.minute_length) + 'min' + ' Prediction'
        plt.title(minute_fig_title)

        self.current_minute_fig_filename = self.image_path + str(datetime.now().timestamp()) + minute_fig_title.replace(' ', '') + '.png'
        plt.savefig(self.current_minute_fig_filename)

        #Create hourly
        hourly_ax = self.hourly_prediction.plot(style='rx--')
        self.hourly_price.plot(style='bo--', ax=hourly_ax)
        hourly_fig_title = self.prediction_ticker.upper() + ' ' + str(self.hour_length) + 'hrs' + ' Prediction'
        plt.title(hourly_fig_title)

        self.current_hourly_fig_filename = self.image_path + str(datetime.now().timestamp()) + hourly_fig_title.replace(' ', '') + '.png'
        plt.savefig(self.current_hourly_fig_filename)

        plt.close('all')

    def send_data(self):
        #Create the message

        altMsgText = MIMEText('Graphs not displaying properly!' + '\n' + str(round(self.minute_prediction, 3)) + '\n' + str(round(self.hourly_prediction, 3)))

        # Create the root message and fill in the from, to, and subject headers
        msg_root = MIMEMultipart('related')
        msg_root['From'] = 'redacted'
        msg_root['To'] = 'redacted'
        msg_root['Subject'] = 'Ethereum Prediction From Your Digital Broker'

        # Encapsulate the plain and HTML versions of the message body in an
        # 'alternative' part, so message agents can decide which they want to display.
        msg_alternative = MIMEMultipart('alternative')
        msg_root.attach(msg_alternative)
        msg_alternative.attach(altMsgText)

        # We reference the image in the IMG SRC attribute by the ID we give it below
        msg_text = MIMEText('Crypto currency prediction plots below.<br><img src="cid:imagem"><br><br><img src="cid:imageh"><br>', 'html')
        msg_alternative.attach(msg_text)

        self.prepare_images()

        minute_fp = open(self.current_minute_fig_filename, 'rb')
        minute_image = MIMEImage(minute_fp.read())
        minute_fp.close()

        hourly_fp = open(self.current_hourly_fig_filename, 'rb')
        hourly_image = MIMEImage(hourly_fp.read())
        hourly_fp.close()


        # Define the image's ID as referenced above
        minute_image.add_header('Content-ID', '<imagem>')
        msg_root.attach(minute_image)
        hourly_image.add_header('Content-ID', '<imageh>')
        msg_root.attach(hourly_image)

        # Setup the SMTP server
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('redacted', 'redacted')
        s.sendmail('redacted', ['redacted'], msg_root.as_string())

    def send_err(self):
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('reacted', 'reacted')
        msg = MIMEText('Error detected, abort')
        msg['Subject'] = 'Ethereum Prediction Error From Your Digital Broker'
        msg['From'] = 'reacted'
        s.sendmail('reacted', ['reacted'], msg.as_string())

    def trade_logic(self, last_bool):
        current_price = self.hourly_prediction.values[-(self.hour_length + 1)]
        next_price = self.hourly_prediction.values[-self.hour_length]
        check_price = self.hourly_prediction.values[-(self.hour_length - 1)]
        val_price = self.hourly_prediction.values[-(self.hour_length - 2)]

        #This tells the bot to send an email if the next predicted price is an inflection point for the next two hours
        if (next_price > current_price) & (next_price > check_price) & (next_price > (val_price + 0.01)):
            self.last_true_countdown = 6
            return True
        elif (next_price < current_price) & (next_price < check_price) & (next_price < (val_price - 0.01)):
            self.last_true_countdown = 6
            return True
        else:
            #If the next price is no longer an inflection point but the last one was keep sending until this hour passes
            if (self.last_true_countdown > 0) & last_bool:
                self.last_true_countdown -= 1
                return True

            return False

    def continuous_monitoring(self):
        current_time = datetime.now().timestamp()
        last_check = 0
        cutoff_time = current_time + 14*3600
        should_send_email = False
        while current_time < cutoff_time:
            if current_time > (last_check + 15*60):
                try:
                    self.find_data()
                    should_send_email = self.trade_logic(should_send_email)
                except:
                    self.send_err()
                    should_send_email = False
                last_check = current_time
                if should_send_email:
                    self.send_data()
            else:
                time.sleep(1)
            current_time = datetime.now().timestamp()

class NaiveTradingBot(BaseTradingBot):
    usd_id = None
    crypto_id = None
    current_state = 'hold'
    buy_history = []
    sell_history = []
    min_usd_balance = 107.90 #Make sure the bot does not trade away all my money, will remove limiter once it has proven itself
    price_ax = None
    pred_ax = None
    buy_ax = None
    sell_ax = None
    starting_price = None
    #TODO remove dependence on hourly prediction

    def __init__(self, hourly_model, minute_model, api_key, secret_key, passphrase, hourly_len=6, minute_len=15, prediction_ticker='ETH', bitinfo_list=None, is_sandbox_api=True):

        #plt.ion()
        #self.fig = plt.figure("buy")
        if bitinfo_list is None:
            bitinfo_list = ['eth']

        super(NaiveTradingBot, self).__init__(hourly_model, minute_model, hourly_len=hourly_len, minute_len=minute_len, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list, is_hourly_prediction_needed=False)

        self.product_id = prediction_ticker.upper() + '-USD'

        if is_sandbox_api:
            self.api_base = 'https://api-public.sandbox.gdax.com'
            self.auth_client = gdax.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)
        else:
            self.api_base = 'https://api.gdax.com'
            self.auth_client = gdax.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)

    def prepare_data_for_plotting(self, corrected_full_minute_prediction, full_minute_prices, trade_inds):

        future_trade_inds = np.nonzero(trade_inds[-self.minute_length::])[0] - 30
        all_trade_inds = np.nonzero(trade_inds)[0]

        if len(trade_inds) == 0:
            values = np.array([0])
            inds = values
        else:
            inds = all_trade_inds
            future_trade_values = corrected_full_minute_prediction[future_trade_inds]
            if len(future_trade_values) == 0:
                trade_price_values = full_minute_prices[all_trade_inds]
            else:
                trade_price_values = full_minute_prices[all_trade_inds[0:-len(future_trade_values)]]
            values = np.concatenate((trade_price_values, future_trade_values))

        return inds, values

    def plot_prediction(self, full_minute_prediction, full_minute_prices, original_buy_inds, original_sell_inds):
        #TODO completely rewrite this to utilize the prepare_data_for_plotting method
        #The two lines below create x axis labels
        full_prediction_times = [dt.strftime('%H:%M') for dt in self.minute_prediction.index]
        full_price_times = [dt.strftime('%H:%M') for dt in self.minute_price.index]

        #This section sets offsets for the verticle and horizontal positions of the prediction
        off_set = self.minute_length
        reverse_offset = len(full_minute_prices) - off_set
        new_prediction_mean = np.mean(full_minute_prices[off_set::]) - np.mean(full_minute_prediction[0:reverse_offset])

        #This section creates the y limits for the graph
        min_price = np.min(full_minute_prices)
        min_pred = np.min(full_minute_prediction + new_prediction_mean)
        max_price = np.max(full_minute_prices)
        max_pred = np.max(full_minute_prediction + new_prediction_mean)

        ymin = 0.99*np.min(np.array([min_price, min_pred]))
        ymax = 1.01*np.max(np.array([max_price, max_pred]))

        #This creates the x axis and ticks
        x_pred = range(off_set, len(full_minute_prediction) + off_set)
        x_price = range(0, len(full_minute_prices))
        n = int((len(full_minute_prediction) + self.minute_length)/5)
        x = range(0, len(full_minute_prediction) + self.minute_length + n, n)
        all_ticks = full_price_times + full_prediction_times[-(self.minute_length+1)::]

        buy_inds, buy_values = self.prepare_data_for_plotting(full_minute_prediction, full_minute_prices, original_buy_inds)
        sell_inds, sell_values = self.prepare_data_for_plotting(full_minute_prediction, full_minute_prices, original_sell_inds)



        if self.price_ax is None:
            self.price_ax = plt.plot(x_price, full_minute_prices, 'b-')
            self.pred_ax = plt.plot(x_pred, full_minute_prediction + new_prediction_mean, 'r--')
            self.sell_ax = plt.plot(sell_inds + off_set, sell_values, 'rx')
            self.buy_ax = plt.plot(buy_inds + off_set, buy_values, 'gx')
        else:
            self.price_ax[0].set_ydata(full_minute_prices)
            self.pred_ax[0].set_ydata(full_minute_prediction + new_prediction_mean)
            self.pred_ax[0].set_xdata(x_pred)
            self.buy_ax[0].set_ydata(buy_values)
            self.buy_ax[0].set_xdata(buy_inds + off_set)
            self.sell_ax[0].set_ydata(sell_values)
            self.sell_ax[0].set_xdata(sell_inds + off_set)

        current_price = full_minute_prices[-1]
        starting_price = self.starting_price
        market_returns = str(np.round((current_price - starting_price)/starting_price, 3))

        plt.ylim((ymin, ymax))
        plt.xlim((x[0], x[-1]))
        plt.xticks(x, all_ticks[::n])

        plt.legend(('Current', 'Predicted'))
        plt.title('Normalized Price and Prediction ( ' + str(market_returns) + '% Market )')

        plt.draw()
        plt.pause(0.1)

    def is_peak_in_minute_price_prediction(self, jump_sign, show_plots=False):
        #jump_sign should be +-1, +1 for valleys (which will rise) and -1 for mountains (which will fall)
        if jump_sign == 1:
            move_type = 'buy'
            peak_type = 'low peak'
        elif jump_sign == -1:
            move_type = 'sell'
            peak_type = 'high peak'

        prediction = self.minute_prediction.values[::, 0]
        prices = self.minute_price.values[::, 0]

        off_ind = 0
        strategy_obj = OptimalTradeStrategy(prediction[-90::], prices[-60::])
        strategy_obj.find_optimal_trade_strategy()
        sell_column = strategy_obj.sell_array
        buy_column = strategy_obj.buy_array

        full_sell_inds = np.nonzero(sell_column)[0]
        full_buy_inds = np.nonzero(buy_column)[0]

        #TODO Set the current decision to be the same as the last decision would have been if the last decision was missed

        sell_inds = np.nonzero(sell_column)[0]
        buy_inds = np.nonzero(buy_column)[0]
        if ((len(buy_inds) == 0) & (len(sell_inds) == 0)):
            if show_plots:
                self.plot_prediction(prediction, prices, full_buy_inds, full_sell_inds)
            print('no trades')
            return False, None

        if show_plots & (jump_sign == 1): #Saves the graph from needing to plot twice
            self.plot_prediction(prediction, prices, full_buy_inds, full_sell_inds)

        if jump_sign == -1:
            trade_now = sell_column[-1]
            prior_inds = np.nonzero(sell_column[0:-1])[0]
        elif jump_sign == 1:
            trade_now = buy_column[-1]
            prior_inds = np.nonzero(buy_column[0:-1])[0]
        else:
            print('no ' + move_type + 's')
            return False, None

        if trade_now:
            print(peak_type)
            print(str(datetime.now().timestamp()))
            return True, trade_now
        elif (len(prior_inds) > 0):
            print('missed ' + move_type)
            return_ind = len(sell_column) - prior_inds[-1]
            return True, return_ind

        print('no peak')
        return False, None

    def time_out_check(self, order_dict):
        first_order_dict_entry = list(order_dict.keys())[0]
        if first_order_dict_entry != 'sequence':
            print('time out')
            time.sleep(6)
            order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        return order_dict

    def get_wallet_contents(self):
        #TODO get rid of cringeworthy repitition

        data = self.auth_client.get_accounts()
        USD_ind = [acc["currency"] == 'USD' for acc in data]
        usd_wallet = data[USD_ind.index(True)]

        crypto_ind = [acc["currency"] == self.prediction_ticker for acc in data]
        crypto_wallet = data[crypto_ind.index(True)]

        if (self.usd_id is None) or (self.crypto_id is None):
            self.usd_id = usd_wallet['id']
            self.crypto_id = crypto_wallet['id']

        return  usd_wallet, crypto_wallet

    def trade_logic(self, order_dict, last_order_dict):
        buy_bool, buy_ind = self.is_peak_in_minute_price_prediction(1)

        sell_bool, sell_ind = self.is_peak_in_minute_price_prediction(-1)

        if buy_bool & sell_bool: #If an up and down jump is approaching this lets you know which to perform first
            buy_bool = buy_ind < sell_ind
            sell_bool = not buy_bool

        if buy_bool:
            current_price = round(float(order_dict['bids'][0][0]), 2)
            last_price = round(float(last_order_dict['bids'][0][0]), 2)
            current_price_delta = (current_price - last_price) / last_price
            if current_price_delta < -0.003:
                buy_bool = False

        elif sell_bool:
            current_price = round(float(order_dict['asks'][0][0]), 2)
            last_price = round(float(last_order_dict['bids'][0][0]), 2)
            current_price_delta = (current_price - last_price) / last_price
            if current_price_delta > 0.003:
                sell_bool = False

        return  buy_bool, sell_bool

    def detect_whale_position(self, order_book):
        #This looks for whales making the same trade as myself. If they are doing so, it trades behind them (in case they leave)
        prices = [round(float(price[1]), 2) > 130 for price in order_book]
        if np.sum(prices) > 0:
            return np.argmax(prices)
        else:
            return None

    def detect_opposing_whale(self, order_book):
        # This looks for whales making the trade opposite mine. It is binary because the algorithm is to take be inactive until the whale is gone or farther up the book
        n = 10
        first_n_sizes = [round(float(price[1]), 2) for price in order_book[0:n]]
        if np.max(first_n_sizes) > 75:
            return True
        else:
            return False

    def determine_trade_price(self, side):
        #side must be 'asks' or 'bids'
        order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        order_dict = self.time_out_check(order_dict)

        if side == 'buy':
            order_type = 'bids'
            opposing_order_type = 'asks'
            sign = -1
        elif side == 'sell':
            order_type = 'asks'
            opposing_order_type = 'bids'
            sign = 1


        #the below chooses the best price that will still be at the top of the order book
        trade_price_opp_type = round(float(order_dict[opposing_order_type][0][0]), 2) + 0.01*sign
        trade_price_type = round(float(order_dict[order_type][0][0]), 2) - 0.01*sign
        trade_price = np.abs(np.max(sign*np.array([trade_price_opp_type, trade_price_type])))


        return trade_price

    def trade_limit(self, side, final_sell=False, trade_price=None, cancel_time='min'):
        # based on function from https://cryptostag.com/basic-gdax-api-trading-with-python/
        if trade_price is None:
            trade_price = self.determine_trade_price(side=side)
            ignore_best_trade = False
        else:
            ignore_best_trade = True

        if (trade_price is None) & (not final_sell):
            return False

        usd_wallet, crypto_wallet = self.get_wallet_contents()

        if side == 'buy':
            usd_available = np.round(float(usd_wallet['available']) - self.min_usd_balance, 2)
            trade_size = np.round(usd_available / (trade_price), 8)
            current_orders = self.auth_client.get_orders()[0] #This and the following if statements with current price checks to make sure a good order is not already out
            current_price = 0
            if len(current_orders) > 0:
                current_price = current_orders[0]['price']
                current_price = float(current_price)
                if (trade_price > (current_price)):
                    self.auth_client.cancel_all(product=self.product_id)

            if (usd_available > 10) & ((trade_price > (current_price)) or ignore_best_trade):
                #for off_set in [0, 1, 3, 5, 7]: #The for loop here and in sell spread out the asking price for a better chance of part of the order being taken
                trade_str = "{:0.2f}".format(trade_price)
                size_str = "{:0.2f}".format(trade_size)
                self.auth_client.buy(price=trade_str, size=size_str, product_id=self.product_id, time_in_force='GTT', cancel_after=cancel_time, post_only=True, trade_size=trade_size)
                print('buy at $' + trade_str)
                print(str(datetime.now().timestamp()))
            elif len(current_orders) == 0:
                return True


        elif side == 'sell':
            crypto_available = np.round(float(crypto_wallet['available']), 8)
            trade_size = crypto_available
            current_orders = self.auth_client.get_orders()[0]
            current_price = 10000
            if len(current_orders) > 0:
                current_price = current_orders[0]['price']
                current_price = float(current_price)
                if (trade_price < (current_price)):
                    self.auth_client.cancel_all(product=self.product_id)

            if crypto_available > 0.05:
                #for off_set in [0, 1, 3, 5, 7]:
                if final_sell:
                    self.auth_client.sell(price=str(trade_price), size=str(trade_size), product_id=self.product_id,
                                          time_in_force='GTT', cancel_after='day', post_only=True, trade_size=trade_size)
                elif (trade_price < (current_price)) or ignore_best_trade:
                    trade_str = "{:0.2f}".format(trade_price)
                    self.auth_client.sell(price=trade_str, size=str(trade_size), product_id=self.product_id, time_in_force='GTT', cancel_after=cancel_time, post_only=True, trade_size=trade_size)
                    print('sell at $' + trade_str)
                    print(str(datetime.now().timestamp()))
            elif len(current_orders) == 0:
                return True

        return False

    def continuous_monitoring(self):
        err_count = 0
        current_time = datetime.now().timestamp()
        last_check = 0
        last_training_time = 0
        cutoff_time = current_time + 15*24*3600
        last_order_dict = self.auth_client.get_product_order_book(self.product_id, level=1)
        hold_eth = False
        whale_watch = False
        #last_price = 0
        self.starting_price = round(float(last_order_dict['asks'][0][0]), 2)
        print('Begin trading at ' + datetime.strftime(datetime.now(), '%m-%d-%Y %H:%M')
+ ' with current price of $' + str(self.starting_price) + ' per ' + self.prediction_ticker + ' and a minnimum required balance of $' + str(self.min_usd_balance))
        downtime = 0
        while current_time < cutoff_time:
            #TODO break up tasks in this loop into different methods
            if (current_time > (last_check + 60)):# & (current_time < (last_training_time + 1*3600)):
                try:
                    downtime = 0
                    last_check = current_time
                    order_dict = self.auth_client.get_product_order_book(self.product_id, level=1)
                    order_dict = self.time_out_check(order_dict)
                    #order_book = order_dict['bids']
                    #last_price = round(float(order_book[0][0]), 2)
                    time.sleep(1) #ratelimit
                    self.find_data(is_hourly_prediction_needed=False)
                    should_buy, should_sell = self.trade_logic(order_dict, last_order_dict)

                    last_order_dict = order_dict
                    last_trade_check = current_time
                    current_trade_time = current_time
                    should_stop_loop = False
                    if should_buy or should_sell:
                        while (current_trade_time < (last_trade_check + 1.2*60)) & (not should_stop_loop):
                            if should_buy:
                                should_stop_loop = self.trade_limit('buy')
                            elif should_sell:
                                should_stop_loop = self.trade_limit('sell')
                            time.sleep(1)
                            current_trade_time = datetime.now().timestamp()
                except:
                    downtime += 1
                    time.sleep(60)
                    print('downtime of ' + str(downtime) + 'min')

            elif current_time > (last_training_time + 2*3600):
                try:
                    #In theory this should retrain the model every hour
                    last_training_time = current_time
                    to_date = self.minute_cp.create_standard_dates()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    fmt = '%Y-%m-%d %H:%M:%S %Z'
                    training_data = DataSet(date_from=from_date.strftime(fmt), date_to=to_date.strftime(fmt),
                                            prediction_length=self.minute_cp.prediction_length, bitinfo_list=self.minute_cp.bitinfo_list,
                                            prediction_ticker=self.prediction_ticker, time_units='minutes')
                    self.minute_cp.data_obj = training_data

                    self.minute_cp.update_model_training()
                    self.minute_cp.model.save(str(current_time) + 'minutes_' + str(self.minute_cp.prediction_length) + 'currency_' + self.prediction_ticker)
                except:
                    downtime += 1
                    time.sleep(60)
                    print('downtime of ' + str(downtime) + 'min')



            current_time = datetime.now().timestamp()

        for i in range(0, 6):
            self.auth_client.cancel_all(product=self.product_id)
            self.trade_limit('sell')
            print('sell')
            time.sleep(12)
        self.trade_limit('sell', final_sell=True)
        print('fin')


def increase_saved_dataset_length(original_ds_path, date_to, time_units='minutes'):
    date_from_search = re.search(r'^.*from_(.*)_to_.*$', original_ds_path).group(1)
    date_from = date_from_search.replace('_', ' ')
    date_to_search = re.search('^.*to_(.*).pickle.*$', original_ds_path).group(1)
    og_to_date = date_to_search.replace('_', ' ')

    with open(original_ds_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)


    data_obj = DataSet(date_from=date_from, date_to=og_to_date, prediction_length=30, bitinfo_list=['eth'], prediction_ticker='ETH', time_units='min', fin_table=saved_table, aggregate=1)
    data_obj.add_data(date_to)

    table_file_name = '_' + time_units + '_from_' + date_from + '_to_' + date_to + '.pickle'
    table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet' + table_file_name.replace(' ', '_')

    with open(table_file_name, 'wb') as cp_file_handle:
        pickle.dump(data_obj.fin_table, cp_file_handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO eliminate unnecessary legacy variables from run_neural_net and CryptoPredict
def run_neural_net(date_from, date_to, prediction_length, epochs, prediction_ticker, bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, min_distance_between_trades, model_path, model_type='price', use_type='test', data_set_path=None, save_test_model=True, test_saved_model=False, layer_count=3, batch_size=32, neuron_grid=None):

    #This creates a CoinPriceModel and saves the data
    if (data_set_path is not None) & (use_type != 'predict') & (not test_saved_model):

        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs,
                            prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                        time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu, data_set_path=data_set_path)

    elif (use_type != 'predict')  & (not test_saved_model):
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, epochs=epochs, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list,
                            time_units=time_unit, activ_func=activ_func, is_leakyrelu=isleakyrelu)


    # 'test' will train a model with given conditions then test it, 'optimize' optimizes neuron count by evaluation data loss, 'predict' predicts data
    if use_type == 'test':
        if test_saved_model:
            cp = CoinPriceModel(date_from, date_to, days=prediction_length, prediction_ticker=prediction_ticker, bitinfo_list=bitinfo_list, time_units=time_unit, model_path=model_path, need_data_obj=True, data_set_path=data_set_path)
            #cp.update_model_training()
            cp.test_model(did_train=False, save_model=True)
        else:
            cp.train_model(neuron_count=neuron_count, min_distance_between_trades=min_distance_between_trades, model_type=model_type, save_model=save_test_model, layers=layer_count, batch_size=batch_size)
            cp.test_model()

    elif use_type == 'optimize':
        hist = []
        if neuron_grid is None:
            neuron_grid = [65, 70, 75, 80, 85, 90, 95, 100]
        for neuron_count in neuron_grid:
            current_hist = cp.train_model(neuron_count=neuron_count, min_distance_between_trades=min_distance_between_trades, model_type='price', save_model=True, layers=layer_count, batch_size=batch_size)

            hist.append(current_hist.history['val_loss'][-2])

        plt.plot(neuron_grid, hist, 'bo--')
        plt.show()

    elif use_type == 'predict':
        cp = CoinPriceModel(date_from, date_to, days=prediction_length, prediction_ticker=prediction_ticker,
                            bitinfo_list=bitinfo_list, time_units=time_unit,
                            model_path=model_path,
                            need_data_obj=False)
        cp.predict(time_units=time_unit, show_plots=True)


#TODO replace cryptocompare with gdax
if __name__ == '__main__':

    code_block = 2
    # 1 for test recent code
    # 2 run_neural_net
    # 3 BaseTradingBot

    if code_block == 1:
        date_from = '2018-10-11 00:00:00 EST'
        date_to = '2018-10-11 06:00:00 EST'
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'minutes'
        pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-10-07_20:42:00_EST.pickle'
        neuron_grid = [100, 200, 300, 400]
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Current_Best_Model/most_recent30currency_ETH.h5'

        strategy_model = CryptoTradeStrategyModel(date_from, date_to, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units, data_set_path=pickle_path, days=30, model_path=model_path)
        strategy_model.is_leakyrelu = False
        buy_loss_grid = []
        sell_loss_grid = []
        should_optimize = False

        if should_optimize:
            for i in neuron_grid:
                buy_loss, sell_loss = strategy_model.test_strategy_model(buy_neurons=i, sell_neurons=i, show_plots=False)
                buy_loss_grid.append(buy_loss)
                sell_loss_grid.append(sell_loss)

            plt.plot(neuron_grid, buy_loss_grid)
            plt.title('Buy Loss')
            plt.xlabel('Neurons')
            plt.ylabel('Loss')

            plt.figure()
            plt.plot(neuron_grid, sell_loss_grid)
            plt.title('Sell Loss')
            plt.xlabel('Neurons')
            plt.ylabel('Loss')

            plt.show()
        else:
            strategy_model.test_strategy_model(buy_neurons=100, sell_neurons=100, show_plots=True)

    elif code_block == 2:
        day = '24'

        #date_from = '2018-10-01 20:00:00 EST'
        #date_to = '2018-10-02 16:00:00 EST'
        #date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00 EST'
        date_from = '2018-11-11 21:00:00 EST'
        date_to = '2018-11-18 20:00:00 EST'
        prediction_length = 30
        epochs = 5000
        prediction_ticker = 'LTC'
        bitinfo_list = ['ltc']
        time_unit = 'minutes'
        activ_func = 'relu'
        isleakyrelu = True
        neuron_count = 37
        layer_count = 3
        batch_size = 96
        old_neuron_grid = [37, 70, 90, 300]#[100, 200, 300, 400, 1000]#
        neuron_grid = old_neuron_grid#[x*2 for x in old_neuron_grid]
        time_block_length = 60
        min_distance_between_trades = 5
        model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Current_Best_Model/most_recent30currency_ETH.h5'
        model_type = 'price' #Don't change this
        use_type = 'test' #valid options are 'test', 'optimize', 'predict'. See run_neural_net for description
        #pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-08-11_08:46:00_EST.pickle'
        pickle_path = None#'/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-11-15_21:06:00_EST.pickle'
        test_model_save_bool = False
        test_model_from_model_path = False
        run_neural_net(date_from, date_to, prediction_length, epochs, prediction_ticker, bitinfo_list, time_unit, activ_func, isleakyrelu, neuron_count, min_distance_between_trades, model_path, model_type, use_type, data_set_path=pickle_path, save_test_model=test_model_save_bool, test_saved_model=test_model_from_model_path, batch_size=batch_size, layer_count=layer_count, neuron_grid=neuron_grid)

    elif code_block == 3:

        if len(sys.argv) > 2:
            minute_path = sys.argv[1]
            api_input = sys.argv[2]
            secret_input = sys.argv[3]
            passphrase_input = sys.argv[4]
            sandbox_bool = bool(sys.argv[5])
            hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_62epochs_30neuron1527097308.228338.h5'

        else:

            hour_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_6hours_leakyreluact_adamopt_mean_absolute_percentage_errorloss_62epochs_30neuron1527097308.228338.h5'
            minute_path = input('What is the model path?')
            api_input = input('What is the api key?')
            secret_input = input('What is the secret key?')
            passphrase_input = input('What is the passphrase?')
            sandbox_bool = input('Is this for a sandbox?')


        naive_bot = NaiveTradingBot(hourly_model=hour_path, minute_model=minute_path,
                                    api_key=api_input,
                                    secret_key=secret_input,
                                    passphrase=passphrase_input, is_sandbox_api=sandbox_bool, minute_len=30)

        naive_bot.continuous_monitoring()

        #Another great unit test
        # minute_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_6epochs_200neuron1527096914.695041.h5'
        #
        # date_from = "2018-05-23 14:00:00 EST"
        # date_to = "2018-05-23 14:45:00 EST"
        # prediction_length = 15
        # epochs = 5000
        # prediction_ticker = 'ETH'
        # bitinfo_list = ['eth']
        # time_unit = 'minutes'
        # activ_func = 'relu'
        # isleakyrelu = True
        # neuron_count = 200
        # time_block_length = 60
        # min_distance_between_trades = 5
        # model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/Legacy/ETHmodel_15minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_6epochs_200neuron1527096914.695041.h5'
        # model_type = 'price'  # Don't change this
        # use_type = 'test'  # valid options are 'test', 'optimize', 'predict'. See run_neural_net for description
        # pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-05-23_14:00:00_EST_to_2018-05-23_15:00:00_EST.pickle'
        # test_model_save_bool = False
        #
        # cp = CoinPriceModel(date_from, date_to, days=prediction_length, prediction_ticker=prediction_ticker,
        #                     bitinfo_list=bitinfo_list, time_units=time_unit, model_path=model_path, need_data_obj=True,
        #                     data_set_path=pickle_path)
        # prediction, test_output = cp.test_model(did_train=False, show_plots=False)
        #
        #TODO make a non VC folder to hide keys (also delete keys before commit or just run them from command line)
        # naive_bot = NaiveTradingBot(hourly_model=hour_path, minute_model=minute_path, api_key='Secret', secret_key='Secret', passphrase='Secret')
        # naive_bot.minute_price = test_output
        # naive_bot.minute_prediction = prediction
        # buy_bool, sell_bool = naive_bot.trade_logic()
        # print(str(buy_bool))
        # print(str(sell_bool))
        # #
        # plt.plot(prediction[-(15+1)::] , 'r--x')
        # plt.plot(test_output[-(15+1)::] , 'b--o')
        # plt.show()

        #The below code would make a great unit test
        # fake_prediction = pd.DataFrame({'Test':np.array([0.1, 0.2, 0.1, 0, 0.1, 0.1, 0.1])})
        # naive_bot.minute_prediction = fake_prediction
        # ans = naive_bot.is_jump_in_minute_price_prediction(-1)
        # print(str(ans))
        # fake_prediction = pd.DataFrame({'Test': np.array([0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1])})
        # naive_bot.minute_prediction = fake_prediction
        # ans = naive_bot.is_jump_in_minute_price_prediction(1)
        # print(str(ans))