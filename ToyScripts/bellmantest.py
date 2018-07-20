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
        returns = find_trade_strategy_value(buy_bool, sell_bool, data)
        plt.plot(all_times[sell_bool], data[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], data[buy_bool], 'gx')
        plt.plot(data)
        plt.title('Return of '+str(returns)+'%')
        plt.show()
    else:
        return sell_bool, buy_bool

def findoptimaltradestrategystochastic(prediction, data, offset, absolute_output, show_plots=True): #Cannot be copie pasted, this is a test
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

        for N in range(10, offset):
            past_predictions = prediction[(ind-N):(ind)]
            past_data = data[(ind-N):(ind)]

            #Find error
            current_fit = np.polyfit(past_data, past_predictions, 1, full=True)
            curr_coeff = current_fit[0][0]
            curr_off = current_fit[0][1]
            current_err = np.sqrt(current_fit[1]/(N-1))
            err_arr = np.append(err_arr, current_err)
            off_arr = np.append(off_arr, curr_off)
            coeff_arr = np.append(coeff_arr, curr_coeff)
            err_judgement_arr = np.append(err_judgement_arr, current_err/np.sqrt(N))#np.abs(prediction[ind-1] - curr_off - curr_coeff*data[ind-1]) - current_err)

        err_ind = np.argmin(np.abs(err_judgement_arr))
        fit_coeff = 1/coeff_arr[err_ind]
        err = err_arr[err_ind]*fit_coeff
        fit_offset = -off_arr[err_ind]*fit_coeff
        const_diff = 2*err
        fuzziness = err_ind + 10


        current_price = np.mean(fit_coeff * prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        prior_price = np.mean(fit_coeff * prediction[(ind - fuzziness - 1):(ind + fuzziness - 1)] + fit_offset)
        bool_price_test = current_price > prior_price
        upper_price = current_price + err
        lower_price = current_price - err

        #TODO check integrals, they appear to be WRONG
        if price_is_rising is None:
            price_is_rising = bool_price_test
            last_inflection_price = current_price #TODO have this depend on current info only, not on future data
        else:
            upper_inflec = last_inflection_price + err
            lower_inflec = last_inflection_price - err
            if bool_price_test != price_is_rising:  # != acts as an xor gate
                if price_is_rising:
                    ln_diff = np.log(upper_price) - np.log(lower_price)
                    sq_diff = ((upper_inflec)**2 - (lower_inflec)**2)/2
                    check_val = sq_diff * ln_diff / const_diff - const_diff
                    #The formula for check val comes from integrating sell_price/buyprice - 1 over the predicted errors
                    #for both the buy and sell prices based on past errors
                    #both the sq and ln differences are needed for symmetry (else you get unbalanced buy or sells)
                    print(str(check_val))
                    if check_val > 0:
                        buy_array[ind] = 1
                        last_inflection_price = current_price
                    else:
                        last_inflection_price = current_price
                else:
                    ln_diff = np.log(upper_inflec) - np.log(lower_inflec)
                    sq_diff = ((upper_price)**2 - (lower_price)**2)/2
                    check_val = sq_diff * ln_diff / const_diff - const_diff

                    if check_val > 0:
                        sell_array[ind] = 1
                        last_inflection_price = current_price
                    else:
                        last_inflection_price = current_price

                price_is_rising = bool_price_test

    buy_bool = [bool(x) for x in buy_array]
    sell_bool = [bool(x) for x in sell_array]
    if show_plots:
        market_returns = 100 * (absolute_output[-1] - absolute_output[30]) / absolute_output[30]
        returns = find_trade_strategy_value(buy_bool, sell_bool, absolute_output)
        plt.plot(all_times[sell_bool], absolute_output[sell_bool], 'rx')
        plt.plot(all_times[buy_bool], absolute_output[buy_bool], 'gx')
        plt.plot(absolute_output)
        plt.title('Return of ' + str(np.round(returns, 3)) + '% vs ' + str(np.round(market_returns, 3)) + '% Market')
        plt.show()

if __name__ == '__main__':
    #pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-07-08_00:00:00_UTC_to_2018-07-09_19:52:00_EST.pickle'
    pickle_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/DataSets/CryptoPredictDataSet_minutes_from_2018-07-15_00:00:00_UTC_to_2018-07-16_00:00:00_UTC.pickle'
    with open(pickle_path, 'rb') as ds_file:
        saved_table = pickle.load(ds_file)

    #model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_4epochs1530856066.874304.h5'
    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/Models/Models/3_Layers/ETHmodel_30minutes_leakyreluact_adamopt_mean_absolute_percentage_errorloss_40neurons_4epochs1530856066.874304.h5'

    date_from = '2018-07-15 00:00:00 UTC'
    date_to = '2018-07-16 00:11:00 UTC'
    #date_from = '2018-06-15 10:20:00 EST'
    #date_to = '2018-07-05 20:29:00 EST'
    bitinfo_list = ['eth']
    cp = CoinPriceModel(date_from, date_to, days=30, prediction_ticker='ETH',
                        bitinfo_list=bitinfo_list, time_units='minutes', model_path=model_path, need_data_obj=True,
                        data_set_path=pickle_path)
    cp.test_model(did_train=False)
    prediction, test_output = cp.test_model(did_train=False, show_plots=False)
    absolute_output = test_output[::, 0]
    zerod_output = absolute_output - np.mean(test_output[::, 0])
    zerod_output = zerod_output.reshape(1, len(zerod_output))
    zerod_output = zerod_output.T
    findoptimaltradestrategystochastic(prediction[::, 0], test_output[::, 0], 40, absolute_output, show_plots=True)

    price = saved_table.ETH_high.values
    findoptimaltradestrategy(price, show_plots=True)

