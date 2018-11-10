import numpy as np
import matplotlib.pyplot as plt
import pickle
from CryptoPredict.CryptoPredict import find_trade_strategy_value
from  CryptoPredict.CryptoPredict import CoinPriceModel
from  CryptoPredict.CryptoPredict import DataSet
from datetime import datetime
from datetime import timedelta
from time import sleep
import pandas as pd

def find_spread_trade_strategy_value(buy_bool, sell_bool, all_prices, return_value_over_time=False):
    # This finds how much money was gained from a starting value of $100 given a particular strategy
    usd_available = 100
    eth_available = 0

    all_buys = all_prices[buy_bool]
    all_sells = all_prices[sell_bool]
    b = 0
    s = 0
    trade_fee_correction = 1

    portfolio_value_over_time = np.array([])

    for i in range(0, len(sell_bool)):

        if buy_bool[i]:
            if (usd_available > 0):
                total_val = all_buys[b] * eth_available + usd_available
                trade_amt = 0.25 * total_val
                if trade_amt > usd_available:
                    trade_amt = usd_available

                eth_available = trade_fee_correction * trade_amt / (all_buys[b]) + eth_available
                usd_available = usd_available - trade_fee_correction * trade_amt
            b += 1
        elif sell_bool[i]:
            if (eth_available > 0):
                total_val =  eth_available + usd_available/all_sells[s]
                trade_amt = 0.25 * total_val
                if trade_amt > eth_available:
                    trade_amt = eth_available

                usd_available = trade_fee_correction * all_sells[s] * trade_amt + usd_available
                eth_available = eth_available - trade_fee_correction * trade_amt
            s += 1

        portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    value = all_prices[-1] * eth_available + usd_available - 100

    portfolio_value_over_time = np.append(portfolio_value_over_time, usd_available + eth_available * all_prices[i])

    if return_value_over_time:
        return value, portfolio_value_over_time
    else:
        return value

def findoptimaltradestrategy(data, show_plots=False, min_price_jump = 1):
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
    if show_plots:
        market_returns = 100 * (data[-1] - data[30]) / data[30]
        returns, value_over_time = find_trade_strategy_value(buy_bool, sell_bool, data,
                                                             return_value_over_time=True)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('Return of ' + str(np.round(returns, 3)) + '% vs ' + str(np.round(market_returns, 3)) + '% Market')

        plt.figure()
        plt.plot(value_over_time, label='Strategy')
        plt.plot(100 * data / (data[1]), label='Market')
        plt.title('Precentage Returns Strategy and Market')
        plt.ylabel('% Return')
        plt.legend()

        plt.show()
    else:
        return sell_bool, buy_bool

def findoptimaltradestrategystochastic(prediction, data, offset, show_plots=True): #Cannot be copie pasted, this is a test
    #offset refers to how many minutes back in time can be checked for creating a fit
    #TODO add shift size to prediction to determine offset for trade
    buy_array = np.zeros(len(data))
    sell_array = np.zeros(len(data))
    all_times = np.arange(0, len(data))
    data_len = len(data)
    price_is_rising = None

    #zerod_prediction = prediction - np.min(prediction)
    #scaled_prediction = zerod_prediction/np.max(zerod_prediction)
    #prediction = np.max(data)*scaled_prediction + np.mean(data)

    # data = data - np.min(data)
    # data = data/np.max(data)

    for i in range(0, data_len-offset):
        print(str(round(100*i/(data_len-offset), 2)) + '% done')
        ind = data_len - i - 1
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr #this array will contain the residual from the prior datum
        fuzzzy_counter = 0

        for N in range(10, offset):
            past_predictions = prediction[(ind-N):(ind)]
            past_data = data[(ind-N):(ind)]

            #Find error
            current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
            current_coeff = current_fit[0][0]
            current_off = current_fit[0][1]
            current_err = 2*np.sqrt(current_fit[1]/(N-1))
            err_arr = np.append(err_arr, current_err)
            off_arr = np.append(off_arr, current_off)
            coeff_arr = np.append(coeff_arr, current_coeff)

            err_judgement_arr = np.append(err_judgement_arr, np.abs(prediction[ind-1] - current_off - current_coeff*data[ind-1]) +  current_err) # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1/coeff_arr[err_ind]

        err = err_arr[err_ind]*fit_coeff
        fit_offset = -off_arr[err_ind]*fit_coeff
        const_diff = 2*err
        fuzziness = int((err_ind + 10)/2) #TODO make more logical fuzziness

        current_price = np.mean(fit_coeff * prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        prior_price = np.mean(fit_coeff * prediction[(ind - fuzziness - 1):(ind + fuzziness - 1)] + fit_offset)
        bool_price_test = current_price > prior_price
        upper_price = current_price + err
        lower_price = current_price - err

        if price_is_rising is None:
            price_is_rising = bool_price_test
            next_inflection_ind = ind
        else:
            if (fuzziness + next_inflection_ind  - ind) > 30: #This factors in the limited precitions available live
                fuzziness = 30 - (next_inflection_ind  - ind)
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
                    if (check_val > 0) and (fit_coeff > 0): #(const_diff/(current_price)):
                        buy_array[ind] = 1

                    next_inflection_ind = ind
                else:
                    ln_diff = (np.log(upper_inflec) - np.log(lower_inflec))/const_diff
                    sq_diff = ((upper_price)**2 - (lower_price)**2)/(2*const_diff)
                    check_val = sq_diff * ln_diff - 1

                    print(str(check_val))

                    if (check_val > 0) and (fit_coeff > 0): #(const_diff/(next_inflection_price)):
                        sell_array[ind] = 1

                    next_inflection_ind = ind


                price_is_rising = bool_price_test

    print(str(fuzzzy_counter))

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]
    if show_plots:
        market_returns = 100 * (data[-1] - data[30]) / data[30]
        returns, value_over_time = find_trade_strategy_value(buy_bool, sell_bool, data, return_value_over_time=True)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('Return of ' + str(np.round(returns, 3)) + '% vs ' + str(np.round(market_returns, 3)) + '% Market')

        plt.figure()
        plt.plot(value_over_time, label='Strategy')
        plt.plot(100 * data / (data[1]), label='Market')
        plt.title('Precentage Returns Strategy and Market')
        plt.ylabel('% Return')
        plt.legend()

        plt.show()

    else:
        return sell_bool, buy_bool

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
        fuzziness = int((err_ind + 10) / 2)  # TODO make more logical fuzziness

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
        # TODO add shift size to prediction to determine offset for trade
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
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
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
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = to_date
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_fin_table.index = np.arange(0, len(test_fin_table))
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
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


            # if ((fuzziness + next_inflection_ind - ind) > 30) or ((fuzziness + far_inflection_ind - ind) > 30):  # This factors in the limited precitions available live
            #     fuzziness = 30 - (next_inflection_ind - ind)
            #     far_fuzziness = 30 - (far_inflection_ind - ind)
            #     next_inflection_price = self.fuzzy_price(fit_coeff, next_inflection_ind, fuzziness, fit_offset)
            #     fuzzzy_counter += 1
            # else:
            #     next_inflection_price = self.fuzzy_price(fit_coeff, next_inflection_ind, fuzziness, fit_offset)

            if bool_price_test != price_is_rising:  # != acts as an xor gate
                check_val = self.find_expected_value_over_many_trades(current_price, err, price_is_rising, const_diff,
                                                                      inflection_inds, fit_coeff, fuzziness, fit_offset)
                if price_is_rising:
                    # TODO run big test without allowing negative coefficients
                    # The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    # for both the buy and sell prices based on past errors
                    # both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if (check_val > 0) & (fit_coeff > 0):
                        buy_array[ind] = 1
                    elif (check_val > 0) & (fit_coeff < 0):
                        sell_array[ind] = 1

                else:
                    print(str(check_val))
                    if (check_val > 0) & (fit_coeff > 0):
                        sell_array[ind] = 1
                    elif (check_val > 0) & (fit_coeff < 0):
                        buy_array[ind] = 1


            price_is_rising = bool_price_test

        if save_inds:
            table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
            with open(table_file_name, 'wb') as file_handle:
                pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

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

class OptimalTradeStrategyV2:

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
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr  # this array will contain the residual from the prior datum
        N = 15

        for i in range(-15, 15):
            past_predictions = prediction[(ind - N - i):(ind-i)]
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

        return err, fit_coeff, fit_offset, const_diff, err_ind

    def offset_price(self, fit_coeff, ind, offset, fit_offset):
        price = np.mean(fit_coeff * self.prediction[ind-offset] + fit_offset)
        return price

    def find_expected_value_over_single_trade(self, buy, sell, err, const_diff):
        ln_diff = (np.log(buy+err) - np.log(buy-err)) / const_diff
        sq_diff = ((sell+err) ** 2 - (sell-err) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff - 1
        return check_val

    def find_best_trade(self, ind, offset, price_is_rising):
        if price_is_rising:
            best_trade = np.argmax(self.prediction[(ind-offset):(ind+self.prediction_len-offset)]) + ind-offset
        else:
            best_trade = np.argmin(self.prediction[(ind-offset):(ind+self.prediction_len-offset)]) + ind-offset

        return best_trade

    def find_expected_value_over_many_trades(self, current_prediction, err, price_is_rising, const_diff, fit_coeff, offset, fit_offset, ind):
        inflection_price = self.offset_price(fit_coeff, ind, offset, fit_offset)
        if price_is_rising:
            buy = inflection_price
            sell_now = False
            best_peak_ind = self.find_best_trade(ind, offset, price_is_rising)
            sell = self.offset_price(fit_coeff, best_peak_ind, offset, fit_offset)
            check_price = sell
        else:
            sell = inflection_price
            sell_now = True
            best_peak_ind = self.find_best_trade(ind, offset, price_is_rising)
            buy = self.offset_price(fit_coeff, best_peak_ind, offset, fit_offset)
            check_price = buy

        expected_return_arr = np.array([])

        current_expected_return = self.find_expected_value_over_single_trade(buy, sell, err, const_diff)
        expected_return_arr = np.append(expected_return_arr, current_expected_return)

        for i in range(1, best_peak_ind - ind):
            current_ind = i + ind
            current_inflection = self.offset_price(fit_coeff, current_ind, offset, fit_offset)
            next_inflection = self.offset_price(fit_coeff, current_ind+1, offset, fit_offset)

            inflection_price = self.offset_price(fit_coeff, current_ind, offset, fit_offset)
            if sell_now:
                if (current_inflection > inflection_price) == (current_inflection > next_inflection):
                    inflection_price = current_inflection
                sell = inflection_price
            else:
                if (current_inflection < inflection_price) == (current_inflection < next_inflection):
                    inflection_price = current_inflection
                buy = inflection_price

            current_expected_return = self.find_expected_value_over_single_trade(buy, sell, err, const_diff)
            expected_return_arr = np.append(expected_return_arr, current_expected_return)

        eval_arr = [x > 0 for x in expected_return_arr]

        if (np.argmax(expected_return_arr) == 0):
            expected_return = 1
        else:
            expected_return = 0


        return expected_return

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        data = self.data
        offset = self.offset
        price_is_rising = None
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        buy_now = False
        sell_now = False
        buy_check = False
        sell_check = False

        for i in range(offset, data_len):
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
            ind = i+1
            fuzzzy_counter = 0

            if ind == len(saved_inds):
                saved_inds = np.vstack((saved_inds, np.zeros((data_len + 1 - len(saved_inds), 5))))
                save_inds = True

            if save_inds:
                # TODO add the ability to increase saved length withut starting over
                if (ind%121 == 0) & (fin_table is not None):
                    #In theory this should retrain the model over predetermined intervals
                    to_date = fin_table.date[ind-1].to_pydatetime()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    test_dates = pd.date_range(from_date, to_date, freq='1min')
                    from_ind = ind - len(test_dates)
                    fmt = '%Y-%m-%d %H:%M:%S'

                    training_fin_table = fin_table[from_ind:ind]
                    training_fin_table.index = np.arange(0, len(training_fin_table))
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST', date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length, bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = to_date
                    to_date = fin_table.date[len(fin_table.date.values)-1].to_pydatetime()
                    test_fin_table = fin_table
                    test_fin_table.index = np.arange(0, len(test_fin_table))
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    #TODO Check to make sure no access to future data!
                    self.prediction[ind::] = prediction[(ind)::, 0]


                err, fit_coeff, fit_offset, const_diff, offset = self.find_fit_info(ind)
                saved_inds[ind, 0] = err
                saved_inds[ind, 1] = fit_coeff
                saved_inds[ind, 2] = fit_offset
                saved_inds[ind, 3] = const_diff
                saved_inds[ind, 4] = offset

            else:
                err = saved_inds[ind, 0]
                fit_coeff = saved_inds[ind, 1]
                fit_offset = saved_inds[ind, 2]
                const_diff = saved_inds[ind, 3]
                offset = int(saved_inds[ind, 4])

            current_price = self.offset_price(fit_coeff, ind, offset, fit_offset)
            prior_price = self.offset_price(fit_coeff, ind - 1, offset, fit_offset)
            bool_price_test = current_price > prior_price

            if price_is_rising is None:
                price_is_rising = not bool_price_test

            if bool_price_test != price_is_rising:  # != acts as an xor gate
                check_val = self.find_expected_value_over_many_trades(current_price, err, price_is_rising, const_diff, fit_coeff, offset, fit_offset, ind)
                if price_is_rising:
                    # TODO run big test without allowing negative coefficients
                    # The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    # for both the buy and sell prices based on past errors
                    # both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if (check_val > 0) & (fit_coeff > 0):
                        buy_now = True
                    elif (check_val > 0) & (fit_coeff < 0):
                        sell_now = True

                else:
                    print(str(check_val))
                    if (check_val > 0) & (fit_coeff > 0):
                        sell_now = True
                    elif (check_val > 0) & (fit_coeff < 0):
                        buy_now = True

            if buy_now:
                buy_array[ind] = 1
                buy_now = False
            elif sell_now:
                sell_array[ind] = 1
                sell_now = False


            price_is_rising = bool_price_test

        # if save_inds:
        #     table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
        #     with open(table_file_name, 'wb') as file_handle:
        #         pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

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

class OptimalTradeStrategyV3:
    #TODO investigate odd fits

    offset = 14
    prediction_len = 30

    def __init__(self, prediction, data):
        self.data = data
        self.prediction = prediction
        self.buy_array = np.zeros(len(data)+1)
        self.sell_array = np.zeros(len(data)+1)
        self.data_len = len(data)

    def find_avg_window_length(self, ind, offset):
        #This method searches the past data to determine what value should be used for the error
        data = self.data
        prediction = self.prediction
        max_fuzz = self.offset
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr  # this array will contain the residual from the prior datum

        for N in range(3, max_fuzz):
            past_predictions = prediction[(ind - N - offset):(ind - offset)]
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
                prediction[ind - 1 - offset] - current_off - current_coeff * data[
                    ind - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1 / coeff_arr[err_ind]

        err = err_arr[err_ind] * fit_coeff
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind + 2) / 2)  # TODO make more logical fuzziness

        return err, fit_coeff, fit_offset, const_diff, fuzziness

    def find_best_offset(self, ind, avg_window):
        # This method searches the past data to determine what value should be used for the error
        data = self.data
        base_prediction = self.prediction
        prediction = np.convolve(base_prediction, np.ones((avg_window,)) / avg_window)[avg_window - 1::]
        err_arr = np.array([])
        off_arr = err_arr
        coeff_arr = err_arr
        err_judgement_arr = err_arr  # this array will contain the residual from the prior datum
        N = 20

        for i in range(-15, 15):
            past_predictions = prediction[(ind - i - N):(ind - i)]
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
                prediction[ind - 1 - i] - current_off - current_coeff * data[
                    ind - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))

        return err_ind

    def find_fit_info(self, ind):

        stop_check = 1
        inf_loop_protection = 100
        ref_price = self.data[ind-1]
        err = ref_price
        i = 0
        window_len = 1
        offset = 0

        while (i<inf_loop_protection) & (stop_check > 0.0005):
            #This loop iteratively finds the best transformations for the fit
            i += 1
            offset_new = self.find_best_offset(ind, window_len)
            err_new, fit_coeff_new, fit_offset_new, const_diff_new, window_len_new = self.find_avg_window_length(ind, offset)

            if (err > np.abs(err_new)) or (i==1):
                offset = offset_new
                err = np.abs(err_new)
                fit_coeff = fit_coeff_new
                fit_offset = fit_offset_new
                const_diff = const_diff_new
                window_len = window_len_new
            else:
                break

            stop_check = err/ref_price

        return err, fit_coeff, fit_offset, const_diff, window_len, offset

    def offset_price(self, fit_coeff, ind, offset, window_len, fit_offset):
        price = np.mean(fit_coeff * self.prediction[(ind-offset-window_len):(ind-offset+window_len)] + fit_offset)
        return price

    def find_transformed_prices(self, fit_coeff, ind, offset, window_len, fit_offset, price_range_start, price_range_stop):
        price = np.array([])

        for i in range(price_range_start, price_range_stop):
            next_price = self.offset_price(fit_coeff, i + ind - offset, offset, window_len, fit_offset)
            price = np.append(price, next_price)

        return price

    def find_expected_value_over_single_trade(self, buy, sell, err, const_diff):
        ln_diff = (np.log(buy+err) - np.log(buy-err)) / const_diff
        sq_diff = ((sell+err) ** 2 - (sell-err) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff - 1
        return check_val

    def find_best_trade(self, fit_coeff, ind, offset, window_len, fit_offset, price_is_rising):

        price = self.find_transformed_prices(fit_coeff, ind, offset, window_len, fit_offset, offset, self.prediction_len)

        if price_is_rising:
            best_trade = np.argmax(price) + ind - offset
        else:
            best_trade = np.argmin(price) + ind - offset

        return best_trade

    def find_expected_value_over_many_trades(self, err, price_is_rising, const_diff, fit_coeff, offset, fit_offset, ind, window_len):

        prices = self.find_transformed_prices(fit_coeff, ind, offset, window_len, fit_offset, offset, self.prediction_len)
        inflection_price = prices[0]

        if price_is_rising:
            buy = inflection_price
            sell_now = True
            best_peak_ind = np.argmax(prices)
            sell = np.max(prices)
            ref_prices = prices[0:best_peak_ind+1]
            if np.argmin(ref_prices) != 0:
                return 0

            check_price = sell
        else:
            sell = inflection_price
            sell_now = False
            best_peak_ind = np.argmin(prices)
            buy = np.min(prices)
            ref_prices = prices[0:best_peak_ind+1]
            if np.argmax(ref_prices) != 0:
                return 0

            check_price = buy

        expected_return_arr = np.array([])

        current_expected_return = self.find_expected_value_over_single_trade(buy, sell, err, const_diff)
        expected_return_arr = np.append(expected_return_arr, current_expected_return)

        for i in range(best_peak_ind, len(prices)):
            current_inflection = prices[i]
            #next_inflection = prices[i+1]

            if sell_now:
                # if (current_inflection > inflection_price) == (current_inflection > next_inflection):
                #     inflection_price = current_inflection
                # else:
                #     continue
                inflection_price = current_inflection
                sell = inflection_price
            else:
                # if (current_inflection < inflection_price) == (current_inflection < next_inflection):
                #     inflection_price = current_inflection
                # else:
                #     continue
                inflection_price = current_inflection
                buy = inflection_price

            current_expected_return = self.find_expected_value_over_single_trade(buy, sell, err, const_diff)
            expected_return_arr = np.append(expected_return_arr, current_expected_return)

        eval_arr = [x > 0 for x in expected_return_arr]

        if np.sum(expected_return_arr) > 0:#(np.argmax(expected_return_arr) == 0) & (np.max(expected_return_arr) > 0):
            expected_return = 1
        else:
            expected_return = 0


        return expected_return

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        data = self.data
        loop_offset = 40
        price_is_rising = None
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        buy_now = False
        sell_now = False
        buy_check = False
        sell_check = False

        for i in range(loop_offset, data_len):
            print(str(round(100 * i / (data_len - loop_offset), 2)) + '% done')
            ind = i+1
            fuzzzy_counter = 0

            if ind == len(saved_inds):
                saved_inds = np.vstack((saved_inds, np.zeros((data_len + 1 - len(saved_inds), 5))))
                save_inds = True

            if save_inds:
                # TODO add the ability to increase saved length withut starting over
                if (ind%121 == 0) & (fin_table is not None):
                    #In theory this should retrain the model over predetermined intervals
                    to_date = fin_table.date[ind-1].to_pydatetime()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    test_dates = pd.date_range(from_date, to_date, freq='1min')
                    from_ind = ind - len(test_dates)
                    fmt = '%Y-%m-%d %H:%M:%S'

                    training_fin_table = fin_table[from_ind:ind]
                    training_fin_table.index = np.arange(0, len(training_fin_table))
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST', date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length, bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = to_date
                    to_date = fin_table.date[len(fin_table.date.values)-1].to_pydatetime()
                    test_fin_table = fin_table
                    test_fin_table.index = np.arange(0, len(test_fin_table))
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    #TODO Check to make sure no access to future data!
                    self.prediction[ind::] = prediction[(ind)::, 0]

                err, fit_coeff, fit_offset, const_diff, window_len, offset = self.find_fit_info(ind)
                if fit_coeff < 0:
                    continue
                saved_inds[ind, 0] = err
                saved_inds[ind, 1] = fit_coeff
                saved_inds[ind, 2] = fit_offset
                saved_inds[ind, 3] = const_diff
                saved_inds[ind, 4] = offset

            else:
                err = saved_inds[ind, 0]
                fit_coeff = saved_inds[ind, 1]
                fit_offset = saved_inds[ind, 2]
                const_diff = saved_inds[ind, 3]
                offset = int(saved_inds[ind, 4])

            current_price = self.offset_price(fit_coeff, ind, offset, window_len, fit_offset)
            prior_price = self.offset_price(fit_coeff, ind-1, offset, window_len, fit_offset)
            bool_price_test = current_price > prior_price

            if price_is_rising is None:
                price_is_rising = not bool_price_test

            if bool_price_test != price_is_rising:  # != acts as an xor gate
                #TODO check that inputs
                check_val = self.find_expected_value_over_many_trades(err, price_is_rising, const_diff, fit_coeff, offset, fit_offset, ind, window_len)
                if price_is_rising:
                    # TODO run big test without allowing negative coefficients
                    # The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    # for both the buy and sell prices based on past errors
                    # both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if (check_val > 0) & (fit_coeff > 0):
                        buy_now = True
                    elif (check_val > 0) & (fit_coeff < 0):
                        sell_now = True

                else:
                    print(str(check_val))
                    if (check_val > 0) & (fit_coeff > 0):
                        sell_now = True
                    elif (check_val > 0) & (fit_coeff < 0):
                        buy_now = True

            if buy_now:
                buy_array[ind] = 1
                buy_now = False
            elif sell_now:
                sell_array[ind] = 1
                sell_now = False


            price_is_rising = bool_price_test

        # if save_inds:
        #     table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
        #     with open(table_file_name, 'wb') as file_handle:
        #         pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

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

class OptimalTradeStrategyV4:

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
        fuzziness = int((err_ind + 10) / 2)  # TODO make more logical fuzziness

        return err, fit_coeff, fit_offset, const_diff, fuzziness

    def fuzzy_price(self, fit_coeff, ind, fuzziness, fit_offset):
        price = np.mean(fit_coeff * self.prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        return price

    def point_value_func(self, upper_buy, lower_buy, upper_sell, lower_sell, const_diff):
        ln_diff = (np.log(upper_buy) - np.log(lower_buy)) / const_diff
        sq_diff = ((upper_sell) ** 2 - (lower_sell) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff - 1
        return check_val

    #def value_func(self, current_prediction, err, price_is_rising, const_diff, fit_coeff, fuzziness, fit_offset):

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        prediction = self.prediction
        data = self.data
        offset = self.offset
        hodl = False

        for i in range(offset, data_len):
            print(str(round(100 * (i - offset) / (data_len - offset), 2)) + '% done')
            ind = i+1


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

class OptimalTradeStrategyV5:

    offset = 40
    prediction_len = 30

    def __init__(self, prediction, data):
        self.data = data
        self.prediction = prediction
        self.buy_array = np.zeros(len(data)+1)
        self.sell_array = np.zeros(len(data)+1)
        self.data_len = len(data)
        self.last_return = 100
        self.last_type = 0

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
        fit_coeff = np.abs(1 / coeff_arr[err_ind])

        err = err_arr[err_ind] * fit_coeff
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind + 10) / 2)  # TODO make more logical fuzziness

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

    def find_expected_value_over_many_trades(self, ind, err, price_is_rising, const_diff, fit_coeff, fuzziness, fit_offset):
        current_prediction = self.fuzzy_price(fit_coeff, ind, fuzziness, fit_offset)

        if price_is_rising:
            upper_buy = current_prediction + err
            lower_buy = current_prediction - err
            sign = -1
        else:
            upper_sell = current_prediction + err
            lower_sell = current_prediction - err
            sign = 1

        value_arr = np.array([])

        for i in range(ind + 1, ind + self.prediction_len - fuzziness):
            price = self.fuzzy_price(fit_coeff, i, fuzziness, fit_offset)
            if price_is_rising:
                upper_sell = price + err
                lower_sell = price - err
                sign = -1

            else:
                upper_buy = price + err
                lower_buy = price - err
                sign = 1

            expected_point_value = self.find_expected_value_over_single_trade(upper_buy, lower_buy, upper_sell, lower_sell,
                                                                  const_diff) + 1

            value_arr = np.append(value_arr, expected_point_value)

        # Below 3 times the standard deviation is used to determine the to aim for but 2 times the standard deviation is used to assess risk

        ref_value_err = 2 * np.std(value_arr)/np.sqrt((fuzziness - 1))
        ref_return = np.mean(value_arr) + ref_value_err
        is_greater = sign*self.prediction[ind] > sign*self.prediction[ind-1]
        is_lesser = sign*self.prediction[ind] > sign*self.prediction[ind+1]
        is_not_inflection = (is_greater != is_lesser)

        if (ref_return < 1) or (is_not_inflection) or (not is_greater) or np.isnan(ref_return):
            return 0

        return 1

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
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
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
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
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = fin_table.date[0].to_pydatetime()
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
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


            buy_check_val = self.find_expected_value_over_many_trades(ind, err, True, const_diff, fit_coeff,
                                                                  fuzziness, fit_offset)
            sell_check_val = self.find_expected_value_over_many_trades(ind, err, False, const_diff, fit_coeff,
                                                                       fuzziness, fit_offset)
            if (buy_check_val is None) or (sell_check_val is None):
                continue

            if sell_check_val != buy_check_val:
                if (buy_check_val > 0):# & (fit_coeff > 0):
                    buy_array[ind] = 1
                # elif (buy_check_val > 0) & (fit_coeff < 0):
                #     sell_array[ind] = 1

                print(str(sell_check_val))
                if (sell_check_val > 0):# & (fit_coeff > 0):
                    sell_array[ind] = 1
                # elif (sell_check_val > 0) & (fit_coeff < 0):
                #     buy_array[ind] = 1

        if save_inds:
            table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
            with open(table_file_name, 'wb') as file_handle:
                pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

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

class OptimalTradeStrategyFuzzyPriceVisualizer:

    offset = 40
    prediction_len = 30

    def __init__(self, prediction, data, low_data):
        self.data = data
        self.prediction = prediction
        self.buy_array = np.zeros(len(data)+1)
        self.sell_array = np.zeros(len(data)+1)
        self.data_len = len(data)
        self.last_return = 100
        self.last_type = 0
        self.data_low = low_data
        self.fig1 = None
        self.fig2 = None

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

            err_judgement_arr = np.append(err_judgement_arr, current_err)#np.abs( prediction[ind - 1] - current_off - current_coeff * data[ind - 1]) + current_err)  # current_err/np.sqrt(N)) #

        err_ind = np.argmin(np.abs(err_judgement_arr))
        err_ind_o = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = np.abs(1 / coeff_arr[err_ind])

        err = err_arr[err_ind] * fit_coeff
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind_o + 10) / 2)  # TODO make more logical fuzziness

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
        #price = fit_coeff * self.prediction[ind] + fit_offset
        return price

    def find_expected_value_over_single_trade(self, upper_buy, lower_buy, upper_sell, lower_sell, const_diff):
        ln_diff = (np.log(upper_buy) - np.log(lower_buy)) / const_diff
        sq_diff = ((upper_sell) ** 2 - (lower_sell) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff - 1
        return check_val

    def create_price_arr(self, ind, fuzziness, fit_offset, fit_coeff):

        max_ind = ind + self.prediction_len - fuzziness
        old_data = self.data[(ind):(max_ind)]
        price_arr = np.array([])

        for i in range(ind, max_ind):
            price = self.fuzzy_price(fit_coeff, i, fuzziness, fit_offset)
            price_arr = np.append(price_arr, price)

        #data = (data - np.mean(data))/np.std(data)
        std_const = np.std(old_data)/np.std(price_arr)
        price_arr = std_const*(price_arr - price_arr[0]) + np.mean(old_data)

        return price_arr, old_data

    def plot_fuzzy_price(self, ind, err, price_is_rising, const_diff, fit_coeff, fuzziness, fit_offset):
        max_ind = ind + self.prediction_len - fuzziness
        data = self.data[(ind+self.prediction_len):(max_ind+self.prediction_len)]
        low_data = self.data_low[(ind+self.prediction_len):(max_ind+self.prediction_len)]
        price_arr, old_data = self.create_price_arr(ind, fuzziness, fit_offset, fit_coeff)
        std_const = np.std(old_data) / np.std(price_arr)

        #plt.plot(np.min(price_arr)*np.ones(np.shape(data)), 'k--')
        #plt.plot(np.max(price_arr)*np.ones(np.shape(data)), 'k--')
        t = np.arange(0, len(price_arr))
        del_correction = self.fuzzy_price(fit_coeff, ind-1, fuzziness, fit_offset) - old_data[-1] - self.fuzzy_price(fit_coeff, ind, fuzziness, fit_offset) + data[0]
        t_coeff = (std_const*err)
        #plt.plot(del_correction*price_arr + data[0]+ del_correction*(-price_arr[0]), 'k--')
        self.fig1 = plt.figure()
        plt.plot(price_arr, 'k--')
        plt.plot(price_arr - t_coeff*(t/30) - np.std(old_data), 'k--')
        plt.plot(price_arr + t_coeff*(t/30) + np.std(old_data), 'k--')

        s = np.argmax(price_arr)
        b = np.argmin(price_arr)

        plt.plot(data, 'g--')
        plt.plot(low_data, 'r--')
        plt.plot(b, data[b], 'gx')
        plt.plot(s, data[s], 'rx')
        plt.show(block=False)

    def plot_expected_value(self, ind, err, price_is_rising, const_diff, fit_coeff, fuzziness, fit_offset):
        max_ind = ind + self.prediction_len - fuzziness
        data = self.data[(ind+self.prediction_len):(max_ind+self.prediction_len)]
        low_data = self.data_low[(ind+self.prediction_len):(max_ind+self.prediction_len)]
        price_arr, old_data = self.create_price_arr(ind, fuzziness, fit_offset, fit_coeff)

        val_arr = np.array([])
        current_prediction = price_arr[0]

        if price_is_rising:
            upper_buy = current_prediction + err
            lower_buy = current_prediction - err
            sign = -1
        else:
            upper_sell = current_prediction + err
            lower_sell = current_prediction - err
            sign = 1

        for i in range(1, len(price_arr)):
            price = price_arr[i]
            if price_is_rising:
                upper_sell = price + err
                lower_sell = price - err
                sign = -1

            else:
                upper_buy = price + err
                lower_buy = price - err
                sign = 1

            expected_point_value = self.find_expected_value_over_single_trade(upper_buy, lower_buy, upper_sell, lower_sell,
                                                                  const_diff) + 1

            if np.isnan(expected_point_value):
                expected_point_value = 0

            val_arr = np.append(val_arr, expected_point_value)

        #plt.plot(np.min(val_arr)*np.ones(np.shape(data)), 'k--')
        #plt.plot(np.max(val_arr)*np.ones(np.shape(data)), 'k--')
        self.fig1 = plt.figure()
        plt.plot((val_arr- np.mean(val_arr))*np.std(data)/np.std(val_arr) + np.mean(data), 'k--')
        #plt.plot(val_arr, 'k--')

        s = np.argmin(price_arr)
        b = np.argmax(price_arr)
        plt.plot(b, data[b], 'gx')
        plt.plot(s, data[s], 'rx')

        plt.plot(data, 'b--')
        plt.plot(low_data, 'r--')
        plt.show(block=False)

    def characterize_shape(self, data):
        #This function encodes the rough shape of the data as a 4 bit number
        min_loc = np.argmin(data)
        max_loc = np.argmax(data)
        min_first = str(int(min_loc == 0))
        max_first = str(int(max_loc == 0))
        min_last = str(int(min_loc == (len(data)-1)))
        max_last = str(int(max_loc == (len(data)-1)))

        shape = eval(str('0b' + min_first + min_last + max_first + max_last))
        # 0000 = 0 /\/, 0001 = 1 increasing \/, 0010 = 2 decreasing \/, 0100 = 4 decreasing /\,  0110 = 6 \, 1000 = 8
        # increasing /\, 1001 = 9 /
        return shape

    def plot_predictions(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        prediction = self.prediction
        data = self.data
        offset = self.offset
        price_is_rising = None
        num_good_pts = 0
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        for i in range(offset, data_len):
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
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
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = fin_table.date[0].to_pydatetime()
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
                    self.prediction[(ind-120)::] = prediction[(ind-120)::, 0]


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

            if (np.std(data[(ind-self.prediction_len):ind]) < err) & (ind > 200):
                if self.fig1 is not None:
                    self.fig2 = self.fig1
                self.plot_expected_value(ind, err, True, const_diff, fit_coeff,
                                  fuzziness, fit_offset)
                sleep(1)

                if (self.fig2 is not None):
                    plt.close(self.fig2.number)

    def plot_shapes(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):
        # offset refers to how many minutes back in time can be checked for creating a fit
        data_len = self.data_len
        offset = self.offset
        shape_array = []

        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        for i in range(offset, data_len):
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
            ind = i + 1


            if ind == len(saved_inds):
                saved_inds = np.vstack((saved_inds, np.zeros((data_len + 1 - len(saved_inds), 5))))
                save_inds = True

            if save_inds:
                # TODO add the ability to increase saved length withut starting over
                if (ind % 121 == 0) & (fin_table is not None):
                    # In theory this should retrain the model over predetermined intervals
                    to_date = fin_table.date[ind - 1].to_pydatetime()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    test_dates = pd.date_range(from_date, to_date, freq='1min')
                    from_ind = ind - len(test_dates)
                    fmt = '%Y-%m-%d %H:%M:%S'

                    training_fin_table = fin_table[from_ind:ind]
                    training_fin_table.index = np.arange(0, len(training_fin_table))
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = fin_table.date[0].to_pydatetime()
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
                    self.prediction[(ind - 120)::] = prediction[(ind - 120)::, 0]

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

            prediction_data, temp = self.create_price_arr(ind, fuzziness, fit_offset, fit_coeff)
            shape = self.characterize_shape(prediction_data)
            shape_array


class OptimalTradeStrategyTestBedForVisualizer(OptimalTradeStrategyFuzzyPriceVisualizer):

    offset = 40
    prediction_len = 30
    order_limits = {'buy': 0, 'sell':0}

    def __init__(self, prediction, data, data_low):
        super(OptimalTradeStrategyTestBedForVisualizer, self).__init__(prediction, data, data_low)

    def compare_shapes(self, prediction, actual):
        pred_shape = self.characterize_shape(prediction)
        actual_shape = self.characterize_shape(actual)

        return pred_shape == actual_shape

    def find_expected_value_over_many_trades(self, ind, err, is_buy, const_diff, fit_coeff, fuzziness, fit_offset):

        data = self.data[(ind-10):ind]

        if is_buy:
            trade_group = [4, 8, 9]
        else:
            trade_group = [0, 1, 2, 6]

        max_ind = ind + self.prediction_len - fuzziness

        price_arr = np.array([])
        old_price_arr = np.array([])

        for i in range(ind-10, max_ind):
            price = self.fuzzy_price(fit_coeff, i, fuzziness, fit_offset)
            if i >= ind:
                price_arr = np.append(price_arr, price)
            else:
                old_price_arr = np.append(old_price_arr, price)

        price_shape = self.characterize_shape(price_arr)
        past_shape = self.characterize_shape(data)
        should_trade = 0

        if (price_shape in trade_group):
            should_trade = 1

        return should_trade

    def place_trade(self, sell_check_val, buy_check_val, sell_array, buy_array, sell_price, buy_price, i, ind):
        data = self.data

        if sell_check_val != buy_check_val:
            # if buy_check_val & (((data[i] + spread*data[i]) > self.order_limits['sell']) or (self.order_limits['buy'] == 0)):
            #     buy_array[ind] = 1
            #     self.order_limits['sell'] = data[i] + spread*data[i]
            #     self.order_limits['buy'] = 0
            #
            # elif sell_check_val & (((low_data[i] - spread*low_data[i]) < self.order_limits['buy']) or (self.order_limits['sell'] == 0)):
            #     self.order_limits['sell'] = 0
            #     sell_array[ind] = 1
            #     self.order_limits['buy'] = data[i] - spread * data[i]
            #
            # elif (self.order_limits['buy'] > low_data[i]) & (self.order_limits['buy'] > 0):
            #     buy_array[ind] = 1
            #     sell_data[i] = self.order_limits['buy']
            #     self.order_limits['buy'] = 0
            #
            # elif (self.order_limits['sell'] < data[i]) & (self.order_limits['sell'] > 0):
            #     sell_array[ind] = 1
            #     sell_data[i] = self.order_limits['sell']
            #     self.order_limits['sell'] = 0

            price = data[i]
            data_diff = np.abs(data[i] - data[i - 1])
            older_data_diff = np.abs(data[i - 1] - data[i - 3])
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
            jump_criteria = (older_data_diff > 2 * np.std(data[(i - 30):i])) and (
            data_diff < 0.5 * np.std(data[(i - 30):i]))
            if buy_check_val:
                buy_array[ind] = 1
                if ((price < sell_price) or (sign < 0)) and jump_criteria:

                    buy_price = price + 0.001 * price
            elif sell_check_val:
                sell_array[ind] = 1
                if ((price > buy_price) or (sign > 0)) and jump_criteria:

                    sell_price = price - 0.001 * (price)

        return sell_price, sell_array, buy_price, buy_array

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO add shift size to prediction to determine offset for trade
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len
        prediction = self.prediction
        data = self.data
        sell_data = data
        low_data = self.data_low
        offset = self.offset
        price_is_rising = None
        last_sell = 0
        sell_price = 1000
        buy_price = 0
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        for i in range(offset, data_len):
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
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
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = fin_table.date[0].to_pydatetime()
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
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


            buy_check_val = self.find_expected_value_over_many_trades(ind, err, True, const_diff, fit_coeff,
                                                                  fuzziness, fit_offset)
            sell_check_val = self.find_expected_value_over_many_trades(ind, err, False, const_diff, fit_coeff, fuzziness, fit_offset)
            if (buy_check_val is None) or (sell_check_val is None):
                continue

            spread = 0.001

            sell_price, sell_array, buy_price, buy_array = self.place_trade(sell_check_val, buy_check_val, sell_array, buy_array, sell_price, buy_price, i, ind)

        if save_inds:
            table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
            with open(table_file_name, 'wb') as file_handle:
                pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

        self.buy_array = np.array([bool(x) for x in buy_array])
        self.sell_array = np.array([bool(x) for x in sell_array])
        if show_plots:
            all_times = np.arange(0, len(data))
            sell_bool = self.sell_array
            buy_bool = self.buy_array
            market_returns = 100 * (data[-1] - data[30]) / data[30]
            returns, value_over_time = find_trade_strategy_value(buy_bool[1:-1], sell_bool[1:-1], sell_data[0:-1], return_value_over_time=True)
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

class OptimalTradeStrategyV5Realistic(OptimalTradeStrategyV5):

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False, fin_table=None, minute_cp=None):  # Cannot be copie pasted, this is a test
        # offset refers to how many minutes back in time can be checked for creating a fit
        # TODO set back to old find_optimal_trade_strategy for unittesting
        buy_array = self.buy_array
        sell_array = self.sell_array
        data_len = self.data_len-1
        prediction = self.prediction
        data = self.data
        offset = self.offset
        price_is_rising = None
        if saved_inds is None:
            saved_inds = np.zeros((data_len + 1, 5))
            save_inds = True
        elif len(saved_inds):
            save_inds = False

        buy_check_val = False
        sell_check_val = False

        for i in range(offset, data_len):
            print(str(round(100 * (i-offset) / (data_len - offset), 2)) + '% done')
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
                    training_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                            date_to=to_date.strftime(fmt) + ' EST',
                                            prediction_length=minute_cp.prediction_length,
                                            bitinfo_list=minute_cp.bitinfo_list,
                                            prediction_ticker='ETH', time_units='minutes', fin_table=training_fin_table)
                    minute_cp.data_obj = training_data

                    minute_cp.update_model_training()

                    from_date = fin_table.date[0].to_pydatetime()
                    to_date = fin_table.date[len(fin_table.date.values) - 1].to_pydatetime()
                    test_fin_table = fin_table
                    test_data = DataSet(date_from=from_date.strftime(fmt) + ' EST',
                                        date_to=to_date.strftime(fmt) + ' EST',
                                        prediction_length=minute_cp.prediction_length,
                                        bitinfo_list=minute_cp.bitinfo_list,
                                        prediction_ticker='ETH', time_units='minutes', fin_table=test_fin_table)
                    minute_cp.data_obj = test_data

                    prediction, test_output = minute_cp.test_model(did_train=False, show_plots=False)
                    # TODO Check to make sure no access to future data!
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


            buy_decision = self.find_expected_value_over_many_trades(ind, err, True, const_diff, fit_coeff,
                                                                  fuzziness, fit_offset)
            sell_decision = self.find_expected_value_over_many_trades(ind, err, False, const_diff, fit_coeff,
                                                                       fuzziness, fit_offset)
            if (buy_check_val is None) or (sell_check_val is None):
                continue

            if buy_decision != sell_decision:
                if buy_decision:
                    buy_check_val = True
                    sell_check_val = False
                elif sell_decision:
                    buy_check_val = False
                    sell_check_val = True

            if (buy_check_val > 0) & (data[ind] < data[ind-1]):# & (fit_coeff > 0):
                buy_array[ind] = 1
            # elif (buy_check_val > 0) & (fit_coeff < 0):
            #     sell_array[ind] = 1

            if (sell_check_val > 0) & (data[ind] > data[ind-1]):# & (fit_coeff > 0):
                sell_array[ind] = 1
            # elif (sell_check_val > 0) & (fit_coeff < 0):
            #     buy_array[ind] = 1

        if save_inds:
            table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/703ModelSavedTestIndsto8042018.pickle'
            with open(table_file_name, 'wb') as file_handle:
                pickle.dump(saved_inds, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(str(fuzzzy_counter))

        self.buy_array = np.array([bool(x) for x in buy_array])
        self.sell_array = np.array([bool(x) for x in sell_array])
        if show_plots:
            all_times = np.arange(0, len(data))
            sell_bool = self.sell_array
            buy_bool = self.buy_array
            market_returns = 100 * (data[-1] - data[30]) / data[30]
            returns, value_over_time = find_spread_trade_strategy_value(buy_bool[0:-2], sell_bool[0:-2], data[0:-1], return_value_over_time=True)
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



if __name__ == '__main__':
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-11-10_09:33:00_EST.pickle'
    inds_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/802ModelSavedTestIndsto8042018.pickle'

    with open(pickle_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)

    with open(inds_path, 'rb') as ind_file:
        saved_inds = pickle.load(ind_file)

    #model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_4epochs1530856066.874304.h5'
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_90neurons_2epochs1540346883.579073.h5'

    #date_from = '2018-09-18 22:23:00 EST'
    #date_to = '2018-09-21 18:58:00 EST'
    date_from = '2018-10-23 00:00:00 EST'
    date_to = '2018-11-10 09:30:00 EST'
    bitinfo_list = ['eth']
    cp = CoinPriceModel(date_from, date_to, days=30, prediction_ticker='ETH',
                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=model_path, need_data_obj=True,
                        data_set_path=pickle_path)
    #cp.test_model(did_train=False)
    prediction, test_output = cp.test_model(did_train=False, show_plots=False)
    data_high = test_output[0:-30, 0]
    data_low = cp.data_obj.final_table[cp.data_obj.prediction_ticker + '_low'][cp.prediction_length:-31].values

    #temp_rand_arr = np.random.rand(data.shape[0], 1)[::, 0] #this can be used to test strategy with perfect data (the class needs randomness)

    #findoptimaltradestrategystochastic(prediction[::, 0], test_output[::, 0], 40, show_plots=True)
    fin_table = cp.data_obj.fin_table
    fin_table.index = np.arange(len(fin_table))
    strategy_obj = OptimalTradeStrategyTestBedForVisualizer(prediction[0::, 0], data_high, data_low)
    #strategy_obj = OptimalTradeStrategyV3(prediction[::, 0], data_high)
    strategy_obj.find_optimal_trade_strategy(saved_inds=None, show_plots=True, fin_table=fin_table, minute_cp=cp )

    #price = saved_table.ETH_high.values
    #findoptimaltradestrategy(data_high, show_plots=True)

