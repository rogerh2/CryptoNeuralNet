import matplotlib
matplotlib.use('Agg')
import sys
# sys.path.append("home/rjhii/CryptoNeuralNet/CryptoPredict")
# use the below for AWS
sys.path.append("home/ubuntu/CryptoNeuralNet/CryptoPredict")
from CryptoPredict import CoinPriceModel
from CryptoPredict import DataSet
import cbpro
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
from time import sleep
import pytz
import os
import traceback
import random

def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def num2str(num, digits):
    fmt_str = "{:0." + str(digits) + "f}"
    num_str = fmt_str.format(num)

    return num_str

def current_est_time():
    naive_date_from = datetime.now()
    utc = pytz.timezone('UTC')
    est_date_from = utc.localize(naive_date_from)
    est = pytz.timezone('America/New_York')
    est_date = est_date_from.astimezone(est)
    return est_date

class SpreadTradeBot:
    min_usd_balance = 100.04  # Make sure the bot does not trade away all my money
    offset = 40
    usd_id = None
    crypto_id = None
    initial_price = 1
    initial_value = 1
    trade_ids = {'buy':'', 'sell':''}
    trade_prices = {'buy':0, 'sell':1000000000}
    trade_logic = {'buy': True, 'sell': True}
    order_status = 'active'
    timer = {'buy':1, 'sell':1}
    should_reset_timer = {'buy': False, 'sell': False}

    def __init__(self, minute_model, api_key, secret_key, passphrase, minute_len=30,
                 prediction_ticker='ETH', bitinfo_list=None, is_sandbox_api=True):

        temp = "2018-05-05 00:00:00 EST"

        if bitinfo_list is None:
            self.bitinfo_list = ['eth']
        else:
            self.bitinfo_list = bitinfo_list

        self.cp = CoinPriceModel(temp, temp, days=minute_len, prediction_ticker=prediction_ticker,
                                 bitinfo_list=self.bitinfo_list, time_units='minutes', model_path=minute_model, need_data_obj=False)

        self.minute_length = minute_len
        self.prediction_ticker = prediction_ticker.upper()
        self.prediction = None
        self.price = None

        self.product_id = prediction_ticker.upper() + '-USD'
        self.save_str = 'most_recent' + str(self.cp.prediction_length) + 'currency_' + self.prediction_ticker + '.h5'

        if is_sandbox_api:
            self.api_base = 'https://api-public.sandbox.pro.coinbase.com'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)
        else:
            self.api_base = 'https://api.pro.coinbase.com'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)

        data = {'Market': 100, 'Algorithm': 100}
        current_datetime = current_est_time()
        self.returns = pd.DataFrame(data=data, index=[current_datetime])


    def spread_bot_predict(self):

        if self.price is None:
            full_minute_prediction, full_minute_price = self.cp.predict(time_units='minutes', show_plots=False)
        else:
            full_minute_prediction, full_minute_price = self.cp.predict(time_units='minutes', show_plots=False, old_prediction=self.prediction, is_first_prediction=False)

        self.prediction = full_minute_prediction.values[::, 0]
        self.price = full_minute_price.values[30::, 0]

    def find_fit_info(self):
        #This method searches the past data to determine what value should be used for the error
        prediction = self.prediction
        data = self.price
        ind = len(data)# I only care about the info for the current time
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

        err = np.abs(err_arr[err_ind] * fit_coeff) # The error must always be positive
        fit_offset = -off_arr[err_ind] * fit_coeff
        const_diff = 2 * err
        fuzziness = int((err_ind + 10) / 2)  # TODO make more logical fuzziness

        return err, fit_coeff, fit_offset, const_diff, fuzziness

    def fuzzy_price(self, fit_coeff, ind, fuzziness, fit_offset):
        price = np.mean(fit_coeff * self.prediction[(ind - fuzziness):(ind + fuzziness)] + fit_offset)
        return price

    def find_point_expected_value(self, upper_buy, lower_buy, upper_sell, lower_sell, const_diff):
        ln_diff = (np.log(upper_buy) - np.log(lower_buy)) / const_diff
        sq_diff = ((upper_sell) ** 2 - (lower_sell) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff
        return check_val

    def find_expected_value(self, err, is_buy, const_diff, fit_coeff, fuzziness, fit_offset):
        current_prediction = self.fuzzy_price(fit_coeff, len(self.prediction)-self.minute_length, fuzziness, fit_offset)
        ind = -self.minute_length

        if is_buy:
            sign = -1
            upper_buy = current_prediction + err
            lower_buy = current_prediction - err
        else:
            sign = 1
            upper_sell = current_prediction + err
            lower_sell = current_prediction - err


        value_arr = np.array([])

        for i in range(-self.minute_length+1, -fuzziness):
            price = self.fuzzy_price(fit_coeff, i, fuzziness, fit_offset)
            if is_buy:
                upper_sell = price + err
                lower_sell = price - err

            else:
                upper_buy = price + err
                lower_buy = price - err

            expected_point_value = self.find_point_expected_value(upper_buy, lower_buy, upper_sell, lower_sell, const_diff)

            value_arr = np.append(value_arr, expected_point_value)

        # Below 3 times the standard deviation is used to determine the to aim for but 2 times the standard deviation is used to assess risk
        expected_value_err = 3*np.std(value_arr)
        expected_return = np.mean(value_arr) + expected_value_err

        ref_value_err = 2 * np.std(value_arr) / np.sqrt((fuzziness - 1))
        ref_return = np.mean(value_arr) + ref_value_err
        is_greater = sign * self.prediction[ind] > sign * self.prediction[ind - 1]
        is_lesser = sign * self.prediction[ind] > sign * self.prediction[ind + 1]
        is_not_inflection = (is_greater != is_lesser)

        if (ref_return < 1) or (is_not_inflection) or (not is_greater) or np.isnan(ref_return):
            return -1, 1

        return expected_return, err

    def find_spread_bounds(self, err, const_diff, fit_coeff, fuzziness, fit_offset, order_type, order_dict):



        if order_type == 'buy':
            dict_type = 'asks'
            opposite_dict_type = 'bids'
            is_buy = True
            trade_sign = 1
        else:
            dict_type = 'bids'
            opposite_dict_type = 'asks'
            is_buy = False
            trade_sign = -1

        prices = self.find_dict_info(order_dict[opposite_dict_type])

        min_allowable_price = np.min(prices)
        max_allowable_price = np.max(prices)

        #Finds the expected return
        expected_return, err = self.find_expected_value(err, is_buy, const_diff, fit_coeff, fuzziness, fit_offset)
        if (expected_return < 1) or (np.isnan(expected_return)):
            return None, None, None

        current_price = float(order_dict[dict_type][0][0])

        expected_future_price = current_price*(expected_return**trade_sign)
        min_future_price = expected_future_price - err
        max_future_price = expected_future_price + err

        if order_type == 'buy':
            min_future_price = min_allowable_price
        elif order_type == 'sell':
            max_future_price = max_allowable_price

        #print('expected return from ' + order_type + 'ing is ' + num2str(100*(expected_return-1), 3) + '% at $' + num2str(expected_future_price, 2) + ' per')

        return min_future_price, max_future_price, current_price

    #TODO write function to allocate funds for buys and sells in get_available_wallet_contents based on current orders and desired orders

    def get_available_wallet_contents(self, data):
        sleep(0.4)
        USD_ind = [acc["currency"] == 'USD' for acc in data]
        usd_wallet = data[USD_ind.index(True)]

        crypto_ind = [acc["currency"] == self.prediction_ticker for acc in data]
        crypto_wallet = data[crypto_ind.index(True)]

        if (self.usd_id is None) or (self.crypto_id is None):
            self.usd_id = usd_wallet['id']
            self.crypto_id = crypto_wallet['id']

        usd_available = np.round(float(usd_wallet['available']) - self.min_usd_balance, 2)
        crypto_available = np.round(float(crypto_wallet['available']), 8)

        return usd_available, crypto_available

    def get_full_wallet_contents(self, type, data):
        sleep(0.4)
        USD_ind = [acc["currency"] == 'USD' for acc in data]
        usd_wallet = data[USD_ind.index(True)]

        crypto_ind = [acc["currency"] == self.prediction_ticker for acc in data]
        crypto_wallet = data[crypto_ind.index(True)]

        if (self.usd_id is None) or (self.crypto_id is None):
            self.usd_id = usd_wallet['id']
            self.crypto_id = crypto_wallet['id']

        usd_available = np.round(float(usd_wallet['balance']) - self.min_usd_balance, 2)
        crypto_available = np.round(float(crypto_wallet['balance']), 8)

        if type == 'buy':
            return usd_available
        else:
            return crypto_available

    def get_portfolio_value(self):
        data = self.auth_client.get_accounts()
        sleep(0.4)
        order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        price = float(order_dict['asks'][0][0])
        sleep(0.4)

        USD_ind = [acc["currency"] == 'USD' for acc in data]
        usd_wallet = data[USD_ind.index(True)]
        crypto_ind = [acc["currency"] == self.prediction_ticker for acc in data]
        crypto_wallet = data[crypto_ind.index(True)]

        usd_value = np.round(float(usd_wallet['balance']) - self.min_usd_balance, 2)
        crypto_value = np.round(price*float(crypto_wallet['balance']), 8)
        portfolio_value = usd_value + crypto_value

        return price, portfolio_value

    def format_price_info(self, returns, price):
        if returns >= 0:
            value_sym = '+'
        else:
            value_sym = '-'


        return_str = value_sym + num2str(returns, 3)
        formated_string = '$' + num2str(price, 2) + ' (' + return_str + ')'

        return formated_string

    def update_returns_data(self, price, portfolio_value):
        current_datetime = current_est_time()
        market_returns = 100 * price / self.initial_price - 100
        portfolio_returns = 100 * portfolio_value / self.initial_value - 100
        data = {'Market': market_returns + 100, 'Algorithm': portfolio_returns + 100}
        new_row = pd.DataFrame(data=data, index=[current_datetime])
        self.returns = self.returns.append(new_row)
        diff_from_max_len = len(self.returns.index) - 604800
        if diff_from_max_len > 0:
            self.returns = self.returns.iloc[diff_from_max_len::]

        return portfolio_returns, market_returns

    def plot_returns(self, portfolio_returns, portfolio_value, market_returns, price):
        # Get data
        portfolio_str = self.format_price_info(portfolio_returns, portfolio_value)
        market_str = self.format_price_info(market_returns, price)

        self.returns.plot()
        plt.title('Portfolio: ' + portfolio_str + '\n' + self.prediction_ticker + ': ' + market_str)
        plt.xlabel('Date/Time')
        plt.ylabel('% Initial Value')

        plt.savefig('returns.png')
        plt.close()

    def find_dict_info(self, dict):
        # dict is the order dict for a single side (either asks or bids)

        prices = np.array([round(float(x[0]), 2) for x in dict])
        return prices

    def price_loop(self, dict, max_price, min_price, num_trades, order_type):
        # trade_sign is -1 for buy and +1 for sell
        # Price loop finds trade prices for spreads based on predicted value
        # dict is the order dict for the side in question (either asks or bids)

        # spread_prices = diff_arr = np.array([])
        prices = self.find_dict_info(dict)

        #This allows the bot to take advantage of the full spread
        min_allowable_price = np.min(prices)
        max_allowable_price = np.max(prices)

        if (order_type == 'sell'):
            min_spread = np.round((max_price - min_allowable_price) / (2*num_trades), 2)
            min_price = min_allowable_price + min_spread
            if num_trades == 1:
                return np.array([min_price])

        elif (order_type == 'buy'):
            min_spread = np.round((max_allowable_price - min_price) / (2*num_trades), 2)
            max_price = max_allowable_price - min_spread
            if num_trades == 1:
                return np.array([max_price])

        price_step = np.round((max_price-min_price)/num_trades, 2)

        if price_step < 0.01:
            return np.array([])

        spread_prices = np.arange(min_price, max_price, price_step)

        return  spread_prices

    def cancel_out_of_bounds_orders(self, upper_limit_price, lower_limit_price, order_type):
        order_generator = self.auth_client.get_orders(self.product_id)
        sleep(0.4)
        order_list = list(order_generator)

        if upper_limit_price is None:
            upper_limit_price = -1.1
            lower_limit_price = 100000.1

        if len(order_list) > 0:
            for i in range(0, len(order_list)):
                order = order_list[i]
                if (order["side"] == order_type) & ((float(order["price"]) > upper_limit_price) or (float(order["price"]) < lower_limit_price)):
                    order_id = order["id"]
                    if order_id == self.trade_ids[order_type]:
                        continue
                    self.auth_client.cancel_order(order_id)
                    sleep(0.4)

    def find_trade_size_and_number(self, err, available, current_price, side):
        # This method finds the limit size to choose as well as the number of orders, it tries to keep orders to under
        # $5000  (hopefully this will be a problem soon!) and at least 4 cent spacing between orders

        past_limit_size = 0
        past_num_orders = 0
        if side == 'buy':
            trade_size = 10
            max_size = 5000
            round_off = 2

        elif side == 'sell':
            trade_size = 0.01
            max_size = 5000/current_price #While the limit prices will change this, it only needs to be approximate
            round_off = 8

        num_orders = int(available / trade_size)

        size_incriment = 0.5*trade_size

        while (abs(trade_size-past_limit_size) > 0.0000001) & (abs(num_orders-past_num_orders) > 0):
            past_limit_size = trade_size
            past_num_orders = num_orders
            num_orders = int(available/trade_size)
            if num_orders == 0:
                print('not enough funds available')
                return 0, 0

            trade_size = available/num_orders

            if num_orders > int(0.5*err*100):
                trade_size = trade_size + size_incriment

            if trade_size > max_size:
                trade_size = max_size

            if num_orders * trade_size > available:
                num_orders = num_orders - 1

        trade_size = round(trade_size, round_off) - 1*10**(-round_off)


        return trade_size, num_orders

    def cancel_old_hodl_order(self, order_type, price, force_limit_order = False, force_stop_limit_order = False, price_lim=0.0):
        if self.trade_ids[order_type] != '':
            hodl_id = self.trade_ids[order_type]
            hodl_order = self.auth_client.get_order(hodl_id)

            if not ('status' in hodl_order.keys()):
                msg = 'Error caught: status not found in order dict'
                return msg

            current_status = hodl_order['status']

            if force_limit_order:
                # This cancels a stop limit order so it can become a limit order
                if (current_status != 'open') & (current_status != 'done'):
                    msg = 'changing stop limit order to limit order'
                    print(msg)
                    price = 0

            elif force_stop_limit_order:
                # This cancels a limit order so it can become a stop limit order
                if (current_status != 'active') & (current_status != 'done'):
                    msg = 'changing limit order to stop limit order'
                    print(msg)
                    price = 0

            if (current_status == 'done') & self.should_reset_timer[order_type]:
                self.timer[order_type] = 1
                self.should_reset_timer[order_type] = False


            if (hodl_order['status'] != 'done') & (np.abs(self.trade_prices[order_type] - price) > price_lim):
                self.auth_client.cancel_order(hodl_id)
                self.trade_ids[order_type] = ''
                self.order_status = hodl_order['status']
                msg = 'just cancelled an ' + hodl_order['status'] + ' order'
                print(msg)
                return msg

        msg = 'waiting on outstanding orders'

        return msg

    def determine_trade_price(self, side, order_dict, is_stop=False):
        #side must be 'asks' or 'bids'

        if side == 'buy':
            order_type = 'bids'
            opposing_order_type = 'asks'
            sign = -1
        elif side == 'sell':
            order_type = 'asks'
            opposing_order_type = 'bids'
            sign = 1


        #the below chooses the best price that will still be at the top of the order book
        if is_stop:
            trade_price = round(float(order_dict[opposing_order_type][1][0]), 2) + 0.01*sign
        else:
            trade_price_opp_type = round(float(order_dict[opposing_order_type][0][0]), 2) + 0.01*sign
            trade_price_type = round(float(order_dict[order_type][0][0]), 2) - 0.01*sign
            trade_price = np.abs(np.max(sign*np.array([trade_price_opp_type, trade_price_type])))


        return trade_price

    def find_order_size_sums(self, order_dict, n):
        ref_price = float(order_dict[0][0])
        order_sum = float(order_dict[0][1])
        for i in range(1, n):
            price = float(order_dict[i][0])
            if price < (ref_price + n/100):
                order_sum += float(order_dict[i][1])
            else:
                break

        return order_sum

    def detect_trade_pressure(self, order_book, opposing_type, order_type):
        # This looks at the buying and selling pressure to detemrine wether to place a stop limit order vs a regular
        # limit order
        n = 4
        order_dict = order_book[order_type]
        opposing_dict = order_book[opposing_type]

        order_sum = self.find_order_size_sums(order_dict, n)
        opposing_sum = self.find_order_size_sums(opposing_dict, n)

        if order_sum > 1.25*opposing_sum:
            return True
        else:
            return False

    def place_limit_orders(self, err, const_diff, fit_coeff, fuzziness, fit_offset, order_type):
        #get min, max, and current price and order_book
        order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        sleep(0.4)
        accnt_data = self.auth_client.get_accounts()
        sleep(0.4)
        min_future_price, max_future_price, current_price = self.find_spread_bounds(err, const_diff, fit_coeff, fuzziness, fit_offset, order_type, order_dict)
        if order_type == 'buy':
            cancel_type = 'sell'
            min_cancel_balance = 0.01
        else:
            cancel_type = 'buy'
            min_cancel_balance = 10

        is_predicted_return = min_future_price is not None

        if (not is_predicted_return) and (self.trade_logic[cancel_type]):
            self.trade_logic[order_type] = False
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, cancel_type)
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            msg = 'Currently no value is expected'
            if self.trade_ids[order_type] != '':
                unused_msg = self.cancel_old_hodl_order(order_type, 0)

            cancel_type_balance = self.get_full_wallet_contents(cancel_type, accnt_data)

            if cancel_type_balance <= min_cancel_balance:
                self.trade_logic[order_type] = True

            return msg

        hodl = False

        trade_reason = 'predicted return'

        self.trade_logic[order_type] = True #this ensures that the last trade type predicted to be profitable is the one currently used

        #This determines whether to buy or sell
        if order_type == 'buy':
            dict_type = 'asks'
            opposite_dict_type = 'bids'
            order_type = 'sell'
            opposing_order_type = 'buy'
            stop_type = 'entry'
            sign = 1
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            usd_available, available = self.get_available_wallet_contents(accnt_data)
            price = self.determine_trade_price(order_type, order_dict, is_stop=True) #using is_stop for the most conservative answer
            opposite_available = usd_available/price
            trade_size_lim = 10/price

        else:
            dict_type = 'bids'
            opposite_dict_type = 'asks'
            order_type = 'buy'
            opposing_order_type ='sell'
            stop_type = 'loss'
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            available, crypto_available = self.get_available_wallet_contents(accnt_data)
            opposite_available = crypto_available
            sign = -1
            price = self.determine_trade_price(order_type, order_dict, is_stop=True) #using is_stop for the most conservative answer
            trade_size_lim = 0.01

        if True:
            last_trade_price = self.trade_prices[cancel_type]
            trade_probability = 0.5 * np.log(self.timer[cancel_type]) / np.log(120)
            rand_num = random.random()
            # elif sign*price > (sign*last_trade_price + 0.001*last_trade_price):
            #     #Trade if the price has moved so much that a new stable are has probably been reached
            #     hodl = True
            if (sign*price < sign*(last_trade_price - sign*0.001*last_trade_price)) or (self.timer[cancel_type] > 2*60):
                #Trade if the price is moving favorably since the last trade
                hodl = True
                trade_reason = 'guess'
            if min_future_price is not None:
                # Always do as the algorithm says
                # hodl = False
                trade_reason = 'predicted_return'

            order_type = opposing_order_type
            price = self.determine_trade_price(order_type, order_dict)
            available = opposite_available

        if hodl:
            size_str = num2str(available, 8)
            is_favorable_pressure = self.detect_trade_pressure(order_dict, dict_type, opposite_dict_type)

            if (self.order_status == 'open' and is_predicted_return) or (is_favorable_pressure):
                unused_msg = self.cancel_old_hodl_order(order_type, price, force_limit_order=True)
                price_str = num2str(price, 2)
                if available < trade_size_lim:
                    msg = 'insufficient funds'
                    return msg
                order = self.auth_client.place_limit_order(self.product_id, order_type, price_str, size_str,
                                                   time_in_force='GTT', cancel_after='hour', post_only=True)
                order_kind = 'limit'
            else:
                price = self.determine_trade_price(order_type, order_dict, is_stop=True)
                stop_price_str = num2str(price + sign * 0.01, 2)
                price_str = num2str(price, 2)
                unused_msg = self.cancel_old_hodl_order(order_type, price, force_stop_limit_order=True)
                if available < trade_size_lim:
                    msg = 'insufficient funds'
                    return msg
                order = self.auth_client.place_order(product_id=self.product_id, side=order_type, price=price_str, size=size_str, stop=stop_type, stop_price=stop_price_str, order_type='limit')
                order_kind = 'stop limit'
            if not ('id' in order.keys()):
                msg = str(order.values())
                return msg
            self.trade_ids[order_type] = order['id']
            self.trade_prices[order_type] = price
            self.should_reset_timer[order_type] = True
            msg = 'placing ' + order_kind + ' order at $' + price_str + ' due to ' + trade_reason
            return msg

        if True:
            unused_msg = self.cancel_old_hodl_order(order_type, 0)
            msg = 'Currently no value is expected.'
            return msg

        trade_size, num_orders = self.find_trade_size_and_number(err, available, current_price, order_type)

        if (num_orders == 0):
            msg = 'No satisfactory limit ' + order_type + 's' + ' found'
            return msg

        limit_prices = self.price_loop(order_dict[dict_type], max_future_price, min_future_price, num_orders, order_type)

        if len(limit_prices) == 0:
            msg = 'No satisfactory limit ' + order_type + 's' + ' found'
            return msg

        #This places the limit orders
        for price in limit_prices:
            if order_type == 'buy':
                size_str = num2str(trade_size/price, 8)
            else:
                size_str = num2str(trade_size, 8)
            price_str = num2str(price, 2)

            self.auth_client.place_limit_order(self.product_id, order_type, price_str, size_str, time_in_force='GTT', cancel_after='hour', post_only=True)
            sleep(0.4)
            print('Placed limit ' + order_type + ' for ' + size_str + ' ' + self.prediction_ticker + ' at $' + price_str + ' per')

        msg = 'Done placing ' + order_type + 's'
        return msg

    def print_err_msg(self, section_text, e, err_counter, current_time):
        last_check = current_time + 5 * 60
        err_counter += 1
        print('failed to' + section_text + ' due to error: ' + str(e))
        print('number of consecutive errors: ' + str(err_counter))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())

        return last_check, err_counter

    def trade_loop(self):
        # This method keeps the bot running continuously
        current_time = datetime.now().timestamp()
        last_check = 0
        last_scrape = 0
        last_training_time = current_time
        last_order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        starting_price = round(float(last_order_dict['asks'][0][0]), 2)
        price, portfolio_value = self.get_portfolio_value()
        self.initial_price = price
        self.initial_value = portfolio_value

        # This message is shown at the beginning for juding the bot's performance down the line
        print('Begin trading at ' + datetime.strftime(datetime.now(), '%m-%d-%Y %H:%M')
              + ' with current price of $' + str(
            starting_price) + ' per ' + self.prediction_ticker + 'and a portfolio worth $' + num2str(portfolio_value, 2))
        sleep(1)
        err_counter = 0
        check_period = 60
        last_plot = 0
        last_buy_msg = ''
        last_sell_msg = ''
        fmt = '%Y-%m-%d %H:%M:'
        portfolio_returns = 0
        market_returns = 0

        while 10.5 < portfolio_value:
            if (current_time > (last_check + check_period)) & (current_time < (last_training_time + 2 * 3600)):
                # Scrape price from cryptocompae
                try:
                    err_counter = 0
                    last_check = current_time
                    if (current_time > (last_scrape + 65)):
                        price, portfolio_value = self.get_portfolio_value()
                        self.spread_bot_predict()
                        last_scrape = current_time
                        #self.order_status = 'active' #This forces the order to be reset as a stop order after 1 minute passes
                        portfolio_returns, market_returns =  self.update_returns_data(price, portfolio_value)
                        self.timer['buy'] += 1
                        self.timer['sell'] += 1


                except Exception as e:
                    last_check, err_counter = self.print_err_msg('find new data', e, err_counter, current_time)

                # Plot returns
                try:
                    err_counter = 0
                    if (current_time > (last_plot + 5*60)):
                        self.plot_returns(portfolio_returns, portfolio_value, market_returns, price)
                        last_plot = current_time
                except Exception as e:
                    last_check, err_counter = self.print_err_msg('plot', e, err_counter, current_time)

                # Make trades
                try:
                    err_counter = 0
                    err, fit_coeff, fit_offset, const_diff, fuzziness = self.find_fit_info()
                    buy_msg = self.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'buy')
                    sell_msg = self.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'sell')
                    current_datetime = current_est_time()
                    if buy_msg != last_buy_msg:
                        print('Current time is ' + current_datetime.strftime(fmt) + ' EST')
                        print('Buy message: ' + buy_msg)
                        last_buy_msg = buy_msg

                    if sell_msg != last_sell_msg:
                        print('Current time is ' + current_datetime.strftime(fmt) + ' EST')
                        print('Sell message: ' + sell_msg)
                        last_sell_msg = sell_msg

                    if True: # self.trade_logic['buy'] != self.trade_logic['sell']:
                        check_period = 1
                    else:
                        check_period = 60
                        if self.trade_ids['buy'] != '':
                            unused_msg = self.cancel_old_hodl_order('buy', 0)

                        if self.trade_ids['sell'] != '':
                            unused_msg = self.cancel_old_hodl_order('sell', 0)
                except Exception as e:
                    last_check, err_counter = self.print_err_msg('trade', e, err_counter, current_time)


            # Update model training
            elif current_time > (last_training_time + 2*3600):
                try:
                    last_scrape = 0
                    err_counter = 0
                    last_training_time = current_time
                    to_date = datetime.now()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    date_to = to_date.strftime(fmt) + '00 UTC'
                    date_from = from_date.strftime(fmt) + '00 UTC'
                    training_data = DataSet(date_from=date_from, date_to=date_to,
                                            prediction_length=self.cp.prediction_length, bitinfo_list=self.bitinfo_list,
                                            prediction_ticker=self.prediction_ticker, time_units='minutes')
                    self.cp.data_obj = training_data

                    self.cp.update_model_training()
                    self.cp.model.save(self.save_str)

                    # Reinitialize CoinPriceModel
                    temp = "2018-05-05 00:00:00 EST"
                    self.cp = CoinPriceModel(temp, temp, days=self.minute_length, prediction_ticker=self.prediction_ticker,
                                 bitinfo_list=self.bitinfo_list, time_units='minutes', model_path=self.save_str, need_data_obj=False)
                    self.prediction = None
                    self.price = None
                    self.cp.pred_data_obj = None
                except Exception as e:
                    last_training_time = current_time + 5*60
                    err_counter += 1
                    print('failed to update training due to error: ' + str(e))
                    print('number of consecutive errors: ' + str(err_counter))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)



            current_time = datetime.now().timestamp()
            if err_counter > 12:
                print('Process aborted due to too many exceptions')
                break

        print('Algorithm failed either due to underperformance or due to too many exceptions. Now converting all crypto to USD')
        self.auth_client.cancel_all(self.product_id)
        accnt_data = self.auth_client.get_accounts()
        sleep(0.4)
        usd_available, crypto_available = self.get_available_wallet_contents(accnt_data)
        self.auth_client.place_market_order(self.product_id, side='sell', size=num2str(crypto_available, 8))





if __name__ == '__main__':
    if len(sys.argv) > 2:
        minute_path = sys.argv[1]
        api_input = sys.argv[2]
        secret_input = sys.argv[3]
        passphrase_input = sys.argv[4]
        sandbox_bool = bool(int(sys.argv[5]))

    else:
        minute_path = input('What is the model path?')
        api_input = input('What is the api key?')
        secret_input = input('What is the secret key?')
        passphrase_input = input('What is the passphrase?')
        sandbox_bool = bool(int(input('Is this for a sandbox?')))

    print(minute_path)
    print(api_input)
    print(secret_input)
    print(passphrase_input)
    print(sandbox_bool)

    naive_bot = SpreadTradeBot(minute_model=minute_path,
                                api_key=api_input,
                                secret_key=secret_input,
                                passphrase=passphrase_input, is_sandbox_api=sandbox_bool, minute_len=30)

    naive_bot.trade_loop()



















