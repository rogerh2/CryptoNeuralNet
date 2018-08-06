import numpy as np
import matplotlib.pyplot as plt
import pickle
from CryptoPredict.CryptoPredict import find_trade_strategy_value
from  CryptoPredict.CryptoPredict import CoinPriceModel



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

    def find_optimal_trade_strategy(self, saved_inds=None, show_plots=False):  # Cannot be copie pasted, this is a test
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
        else:
            save_inds = False

        # zerod_prediction = prediction - np.min(prediction)
        # scaled_prediction = zerod_prediction/np.max(zerod_prediction)
        # prediction = np.max(data)*scaled_prediction + np.mean(data)

        # data = data - np.min(data)
        # data = data/np.max(data)

        for i in range(offset, data_len):
            print(str(round(100 * i / (data_len - offset), 2)) + '% done')
            ind = i+1
            fuzzzy_counter = 0

            if save_inds:
                # TODO add the ability to increase saved length withut starting over
                # if ind%1440 == 0:
                #     #In theory this should retrain the model every hour
                #     to_date = self.minute_cp.create_standard_dates()
                #     from_delta = timedelta(hours=2)
                #     from_date = to_date - from_delta
                #     fmt = '%Y-%m-%d %H:%M:%S %Z'
                #     training_data = DataSet(date_from=from_date.strftime(fmt), date_to=to_date.strftime(fmt),
                #                             prediction_length=self.minute_cp.prediction_length, bitinfo_list=self.minute_cp.bitinfo_list,
                #                             prediction_ticker=self.prediction_ticker, time_units='minutes')
                #     self.minute_cp.data_obj = training_data
                #
                #     self.minute_cp.update_model_training()


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

            #if fit_coeff < 0:
            #    continue

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
                    # TODO find expected value from discrete integral over resiuals
                    # The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    # for both the buy and sell prices based on past errors
                    # both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    if (check_val > 0) & (fit_coeff > 0):
                        buy_array[ind] = 1

                else:
                    print(str(check_val))
                    if (check_val > 0) & (fit_coeff > 0):
                        sell_array[ind] = 1

            price_is_rising = bool_price_test

        if save_inds:
            table_file_name = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/savedTestInds2.pickle'
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




if __name__ == '__main__':
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-08-04_13:46:00_EST.pickle'
    #pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-06-15_10:20:00_EST_to_2018-07-30_20:34:00_EST.pickle'
    inds_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ToyScripts/SavedInds/savedTestInds1.pickle'

    with open(pickle_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)

    with open(inds_path, 'rb') as ind_file:
        saved_inds = pickle.load(ind_file)

    #model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_4epochs1530856066.874304.h5'
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/Current_Best_Model/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_80neurons_3epochs1532511217.103676.h5'

    date_from = '2018-06-15 10:20:00 EST'
    date_to = '2018-08-04 13:46:00 EST'
    start_ind = 0
    #date_from = '2018-06-15 10:20:00 EST'
    #date_to = '2018-07-30 20:34:00 EST'
    bitinfo_list = ['eth']
    cp = CoinPriceModel(date_from, date_to, days=30, prediction_ticker='ETH',
                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=model_path, need_data_obj=True,
                        data_set_path=pickle_path)
    #cp.test_model(did_train=False)
    prediction, test_output = cp.test_model(did_train=False, show_plots=False)
    data = test_output[::, 0]

    #findoptimaltradestrategystochastic(prediction[::, 0], test_output[::, 0], 40, show_plots=True)
    strategy_obj = OptimalTradeStrategy(prediction[start_ind::, 0], test_output[start_ind:-30, 0])
    #strategy_obj = OptimalTradeStrategy(prediction[200:629, 0], test_output[200:599 , 0])
    strategy_obj.find_optimal_trade_strategy(saved_inds=None, show_plots=True )

    #price = saved_table.ETH_high.values
    #findoptimaltradestrategy(price, show_plots=True)
