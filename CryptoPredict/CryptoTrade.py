import matplotlib
matplotlib.use('Agg')
import sys
# sys.path.append("home/rjhii/CryptoNeuralNet/CryptoPredict")
# use the below for AWS
#sys.path.append("home/ubuntu/CryptoNeuralNet/CryptoPredict")
from CryptoPredict.CryptoPredict import CoinPriceModel
from CryptoPredict.CryptoPredict import DataSet
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
    min_usd_balance = 109  # Make sure the bot does not trade away all my money
    offset = 40
    usd_id = None
    crypto_id = None
    initial_price = 1
    initial_value = 1
    trade_ids = {'buy':'', 'sell':''}
    trade_prices = {'buy':'', 'sell':''}
    trade_logic = {'buy': True, 'sell': True}

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

        print('expected return from ' + order_type + 'ing is ' + num2str(100*(expected_return-1), 3) + '% at $' + num2str(expected_future_price, 2) + ' per')

        return min_future_price, max_future_price, current_price

    #TODO write function to allocate funds for buys and sells in get_wallet_contents based on current orders and desired orders

    def get_wallet_contents(self):
        # TODO get rid of cringeworthy repitition

        data = self.auth_client.get_accounts()
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

    def plot_returns(self):
        # Get data
        current_datetime = current_est_time()
        price, portfolio_value = self.get_portfolio_value()
        market_returns = 100 * price / self.initial_price - 100
        portfolio_returns = 100 * portfolio_value / self.initial_value - 100
        data = {'Market': market_returns + 100, 'Algorithm': portfolio_returns + 100}
        new_row = pd.DataFrame(data=data, index=[current_datetime])
        self.returns = self.returns.append(new_row)

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

        # prices = self.find_dict_info(dict)
        # j = 0
        #
        # for i in range(1, len(prices)):
        #     if (price < min_price):
        #         if trade_sign == -1:
        #             break
        #         else:
        #             continue
        #     elif(price > max_price):
        #         if trade_sign == 1:
        #             break
        #         else:
        #             continue
        #
        #     price = prices[i]
        #     prior_price = prices[i - 1]
        #     bid_diff = abs(price - prior_price)
        #     if trade_sign*price > naive_trade_prices[j]:
        #         j += 1

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

    def cancel_old_hodl_order(self, order_type, price):
        if self.trade_ids[order_type] != '':
            hodl_id = self.trade_ids[order_type]
            hodl_order = self.auth_client.get_order(hodl_id)

            if not ('status' in hodl_order.keys()):
                msg = 'waiting on outstanding orders'
                return msg

            if (hodl_order['status'] != 'done') & (self.trade_prices[order_type] != price):
                print('Canceled old hodl order for more fluidity')
                self.auth_client.cancel_order(hodl_id)
                self.trade_ids[order_type] = ''

        msg = 'waiting on outstanding orders'

        return msg

    def determine_trade_price(self, side, order_dict):
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
        trade_price_opp_type = round(float(order_dict[opposing_order_type][0][0]), 2) + 0.01*sign
        trade_price_type = round(float(order_dict[order_type][0][0]), 2) - 0.01*sign
        trade_price = np.abs(np.max(sign*np.array([trade_price_opp_type, trade_price_type])))


        return trade_price

    def place_limit_orders(self, err, const_diff, fit_coeff, fuzziness, fit_offset, order_type):
        #get min, max, and current price and order_book
        order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)
        sleep(0.4)
        min_future_price, max_future_price, current_price = self.find_spread_bounds(err, const_diff, fit_coeff, fuzziness, fit_offset, order_type, order_dict)
        if order_type == 'buy':
            cancel_type = 'sell'
        else:
            cancel_type = 'buy'

        if min_future_price is None and (self.trade_logic[cancel_type]):
            self.trade_logic[order_type] = False
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, cancel_type)
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            msg = 'Currently no value is expected from ' + order_type + 'ing ' + self.prediction_ticker + ' now'
            if self.trade_ids[order_type] != '':
                unused_msg = self.cancel_old_hodl_order(order_type, 0)
            return msg

        hodl = False

        self.trade_logic[order_type] = True
        #This determines whether to buy or sell
        if order_type == 'buy':
            dict_type = 'asks'
            order_type = 'sell'
            opposing_order_type = 'buy'
            stop_type = 'entry'
            sign = 1
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            usd_available, available = self.get_wallet_contents()
            price = self.determine_trade_price(order_type, order_dict)
            opposite_available = usd_available/price
            trade_size_lim = 10/price

        else:
            dict_type = 'bids'
            order_type = 'buy'
            opposing_order_type ='sell'
            stop_type = 'loss'
            self.cancel_out_of_bounds_orders(max_future_price, min_future_price, order_type)
            available, crypto_available = self.get_wallet_contents()
            opposite_available = crypto_available
            sign = -1
            price = self.determine_trade_price(order_type, order_dict)
            trade_size_lim = 0.01

        if not self.trade_logic[order_type]:
            hodl = True
            order_type = opposing_order_type
            price = self.determine_trade_price(order_type, order_dict)
            available = opposite_available
            if available < trade_size_lim:
                msg = self.cancel_old_hodl_order(order_type, price)
                return msg

        if hodl:
            # TODO replace with stop order (see below)
            # --Stop order instructions--
            # Stop orders become active and wait to trigger based on the movement of the last trade price. There are two
            # types of stop orders, stop loss and stop entry:
            #
            # stop: 'loss': Triggers when the last trade price changes to a value at or below the stop_price.
            #
            # stop: 'entry': Triggers when the last trade price changes to a value at or above the stop_price.
            #
            # The last trade price is the last price at which an order was filled. This price can be found in the latest
            # match message. Note that not all match messages may be received due to dropped messages.
            price_str = num2str(price, 2)
            stop_price_str = num2str(price + sign*0.02, 2)
            size_str = num2str(available, 8)
            order = self.auth_client.place_order(product_id=self.product_id, side=order_type, price=price_str, size=size_str, stop=stop_type, stop_price=stop_price_str, order_type='limit')
            self.trade_ids[order_type] = order['id']
            self.trade_prices[order_type] = price
            msg = order_type + 'ing at $' + price_str + 'due to lack of funds'
            return msg

        if True:
            msg = 'Spread trading has been manually disabled'
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

        while 20 < portfolio_value:
            if (current_time > (last_check + check_period)) & (current_time < (last_training_time + 2 * 3600)):
                # Scrape price from cryptocompae
                try:
                    err_counter = 0
                    last_check = current_time
                    if (current_time > (last_scrape + 65)):
                        self.spread_bot_predict()
                        last_scrape = current_time
                except Exception as e:
                    last_check = current_time + 5*60
                    err_counter += 1
                    print('failed to find new data due to error: ' + str(e))
                    print('number of consecutive errors: ' + str(err_counter))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # Plot returns
                try:
                    err_counter = 0
                    if (current_time > (last_plot + 5*60)):
                        price, portfolio_value = self.get_portfolio_value()
                        self.plot_returns()
                        last_plot = current_time
                except Exception as e:
                    last_check = current_time + 5*60
                    err_counter += 1
                    print('failed to plot due to error: ' + str(e))
                    print('number of consecutive errors: ' + str(err_counter))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # Make trades
                try:
                    err_counter = 0
                    err, fit_coeff, fit_offset, const_diff, fuzziness = self.find_fit_info()
                    buy_msg = self.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'buy')
                    sell_msg = self.place_limit_orders(err, const_diff, fit_coeff, fuzziness, fit_offset, 'sell')
                    print(buy_msg)
                    print(sell_msg)
                    if self.trade_logic['buy'] != self.trade_logic['sell']:
                        check_period = 2.5
                    else:
                        check_period = 60
                        if self.trade_ids['buy'] != '':
                            unused_msg = self.cancel_old_hodl_order('buy', 0)

                        if self.trade_ids['sell'] != '':
                            unused_msg = self.cancel_old_hodl_order('sell', 0)
                except Exception as e:
                    last_check = current_time + 5 * 60
                    err_counter += 1
                    print('failed to trade due to error: ' + str(e))
                    print('number of consecutive errors: ' + str(err_counter))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)


            # Update model training
            elif current_time > (last_training_time + 2*3600):
                try:
                    err_counter = 0
                    last_training_time = current_time
                    to_date = self.cp.create_standard_dates()
                    from_delta = timedelta(hours=2)
                    from_date = to_date - from_delta
                    fmt = '%Y-%m-%d %H:%M:%S %Z'
                    training_data = DataSet(date_from=from_date.strftime(fmt), date_to=to_date.strftime(fmt),
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
        usd_available, crypto_available = self.get_wallet_contents()
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



















