from CryptoPredict.CryptoPredict import CoinPriceModel
import cbpro
import numpy as np

class SpreadTradeBot:

    def __init__(self, minute_model, api_key, secret_key, passphrase, minute_len=15,
                 prediction_ticker='ETH', bitinfo_list=None, is_sandbox_api=True):

        temp = "2018-05-05 00:00:00 EST"

        if bitinfo_list is None:
            bitinfo_list = ['eth']

        self.cp = CoinPriceModel(temp, temp, days=minute_len, prediction_ticker=prediction_ticker,
                                 bitinfo_list=bitinfo_list, time_units='minutes', model_path=minute_model, need_data_obj=False)

        self.minute_length = minute_len
        self.prediction_ticker = prediction_ticker.upper()
        self.prediction = None
        self.price = None

        self.product_id = prediction_ticker.upper() + '-USD'

        if is_sandbox_api:
            self.api_base = 'https://public.sandbox.pro.coinbase.com/'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)
        else:
            self.api_base = 'https://api-public.sandbox.pro.coinbase.com'
            self.auth_client = cbpro.AuthenticatedClient(api_key, secret_key, passphrase, api_url=self.api_base)

    def predict(self):

        if self.price is None:
            full_minute_prediction, full_minute_price = self.cp.predict(time_units='minutes', show_plots=False)
        else:
            full_minute_prediction, full_minute_price = self.cp.predict(time_units='minutes', show_plots=False, old_prediction=self.prediction.values[::, 0], is_first_prediction=False)

        self.prediction = full_minute_prediction
        self.price = full_minute_price

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

    def find_point_expected_value(self, upper_buy, lower_buy, upper_sell, lower_sell, const_diff):
        ln_diff = (np.log(upper_buy) - np.log(lower_buy)) / const_diff
        sq_diff = ((upper_sell) ** 2 - (lower_sell) ** 2) / (2 * const_diff)
        check_val = sq_diff * ln_diff
        return check_val

    def find_expected_value(self, current_prediction, err, price_is_rising, const_diff, future_inds, fit_coeff, fuzziness, fit_offset):
        if price_is_rising:
            upper_buy = current_prediction + err
            lower_buy = current_prediction - err
            sell_now = False
        else:
            upper_sell = current_prediction + err
            lower_sell = current_prediction - err
            sell_now = True


        expected_return_arr = np.array([])

        for i in range(0, len(future_inds)):
            price = self.fuzzy_price(fit_coeff, int(future_inds[i]), fuzziness, fit_offset)
            if sell_now:
                upper_buy = price + err
                lower_buy = price - err
            else:
                upper_sell = price + err
                lower_sell = price - err

            current_expected_return = self.find_point_expected_value(upper_buy, lower_buy, upper_sell, lower_sell, const_diff)
            expected_return_arr = np.append(expected_return_arr, current_expected_return)

        expected_return = np.mean(expected_return_arr)

        return expected_return, err

    def find_spread_bounds(self, current_prediction, err, price_is_rising, const_diff, future_inds, fit_coeff, fuzziness, fit_offset, order_type):
        order_dict = self.auth_client.get_product_order_book(self.product_id, level=2)

        #Finds the expected return
        expected_return, err = self.find_expected_value(current_prediction, err, price_is_rising, const_diff, future_inds, fit_coeff, fuzziness, fit_offset)
        if expected_return < 0:
            return None


        current_price = order_dict[order_type][0][0]

        expected_future_price = expected_return*current_price
        min_future_price = expected_future_price - err
        max_future_price = expected_future_price + err

        return min_future_price, max_future_price, current_price

    def get_wallet_contents(self):
        # TODO get rid of cringeworthy repitition

        data = self.auth_client.get_accounts()
        USD_ind = [acc["currency"] == 'USD' for acc in data]
        usd_wallet = data[USD_ind.index(True)]

        crypto_ind = [acc["currency"] == self.prediction_ticker for acc in data]
        crypto_wallet = data[crypto_ind.index(True)]

        if (self.usd_id is None) or (self.crypto_id is None):
            self.usd_id = usd_wallet['id']
            self.crypto_id = crypto_wallet['id']

        return usd_wallet, crypto_wallet

    def price_loop(self, prices, max_price, min_price, trade_sign):
        #TODO fix this description
        #TODO write unit test
        # Price loop finds trade prices for trades made against the expected move (hence risky_trade_prices is for the
        # dangerous trades
        trade_prices = diff_arr = np.array([])

        for i in range(1, len(prices)):
            price = prices[i]
            prior_price = prices[i - 1]
            bid_diff = abs(price - prior_price)
            diff_arr = np.append(diff_arr, bid_diff)
            if (trade_sign*price > trade_sign*max_price):
                break

        #TODO have this loop return prices on order book for spread trade

    def find_spread_prices(self, current_prediction, err, price_is_rising, const_diff, future_inds, fit_coeff, fuzziness, fit_offset, order_type):
        #get min, max, and current price and order_book
        min_future_price, max_future_price, current_price = self.find_spread_bounds(current_prediction, err, price_is_rising, const_diff, future_inds, fit_coeff, fuzziness, fit_offset, order_type)

        order_dict = self.auth_client.get_product_order_book(self.product_id, level=1)

        #find the the largest gaps in the order book


        #Spread the price evenly between the largest gaps (if no gaps then spread evenly over 1 cent intervals