import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process
from multiprocessing import Queue
import CryptoBot.CryptoForecast as cf
from CryptoBot.CryptoStrategies import Strategy
from time import time
import multiprocessing, logging
import pickle
from CryptoBot.CryptoBot_Shared_Functions import rescale_to_fit
from CryptoBot.CryptoBot_Shared_Functions import progress_printer

class BackTestExchange:
    orders = {'bids': {}, 'asks': {}}
    time = 0
    min_order = {'bids': 0.01, 'asks': 0.01}

    def __init__(self, order_book_path):
        if order_book_path is not None:
            historical_order_books = pd.read_csv(order_book_path)
            if 'Unnamed: 0' in historical_order_books.columns.values:
                # Some data files have the above header containing the indices, this gets rid of it
                self.order_books = historical_order_books.drop(['Unnamed: 0'], axis=1)
            else:
                self.order_books = historical_order_books
        # self.model = keras.models.load_model(model_path)

    def get_top_order(self, side):
        ind = self.time
        if side == 'asks':
            col = '60'
        elif side == 'bids':
            col = '0'
        else:
            raise ValueError('Side must be either "asks" or "bids"')
        top_order = self.order_books.iloc[[ind]][col].values[0]
        return top_order

    def get_current_book(self):
        ind = self.time
        current_order_book_row = self.order_books.iloc[[ind]]
        return current_order_book_row

    def create_new_order_id(self, this_side_orders):
        if len(this_side_orders) > 0:
            existing_ids = np.array(list(this_side_orders.keys()))
            new_order_id = np.max(existing_ids) + 1
        else:
            new_order_id = 0

        return new_order_id

    def place_maker_order(self, price, side, size, coeff, top_opposing_order):
        new_order_id = None

        if (coeff*price >= coeff*top_opposing_order) and (size > self.min_order[side]):
            # This ensures the order is only placed onto the appropiate order book side
            this_side_orders = self.orders[side]
            new_order_id = self.create_new_order_id(this_side_orders)
            this_side_orders[new_order_id] = {'size': size, 'price': price, 'filled': False, 'is maker':True}

        return new_order_id

    def place_post_only_disabled_order(self, price, side, size, coeff, top_opposing_order):
        new_order_id = self.place_maker_order(price, side, size, coeff, top_opposing_order)

        if (new_order_id is None) and (size > self.min_order[side]):
            this_side_orders = self.orders[side]
            new_order_id = self.create_new_order_id(this_side_orders)
            this_side_orders[new_order_id] = {'size': size, 'price':  top_opposing_order, 'filled': True, 'is maker':False}


        return new_order_id

    def place_order(self, price, side, size, post_only=True):
        if side == 'asks':
            coeff = 1
            opposing_side = 'bids'
        elif side == 'bids':
            coeff = -1
            opposing_side = 'asks'
        else:
            raise ValueError(side + ' is not a valid orderbook side')
        top_opposing_order = self.get_top_order(opposing_side)

        if post_only:
            new_order_id = self.place_maker_order(price, side, size, coeff, top_opposing_order)
        else:
            new_order_id = self.place_post_only_disabled_order(price, side, size, coeff, top_opposing_order)

        return new_order_id

    def update_fill_status(self):
        sides = ['bids', 'asks']

        for side_ind in [0, 1]:
            # This loop cycles through the bids and asks books and fills any orders that are on the wrong book
            side = sides[side_ind]
            opposing_side = sides[not side_ind]
            coeff = (-1)**side_ind
            top_opposing_order = self.get_top_order(opposing_side)

            for order in self.orders[side].values():
                if (not order['filled']) and (coeff*order['price'] >= coeff*top_opposing_order):
                    order['filled'] = True

    def remove_order(self, side, order_id):
        self.orders[side].pop(order_id)

class BackTestPortfolio:
    value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
    # USD is total value stored in USD, SYM is total value stored in crypto, USD Hold is total value in bids, and SYM
    # Hold is total value in asks

    def __init__(self, order_book_path=None):
        self.exchange = BackTestExchange(order_book_path)

    def update_value(self, maker_fee=0.0015, taker_fee=0.0025):
        self.value['USD Hold'] = 0
        self.value['SYM Hold'] = 0
        for side in ['asks', 'bids']:
            ids_to_remove = []
            orders = self.exchange.orders[side]
            for order_id in orders.keys():
                order = orders[order_id]
                if side == 'bids':
                    from_val = order['size'] * order['price']
                    to_val = order['size']
                    to_sym = 'SYM'
                    from_sym = 'USD'
                else:
                    to_val = order['size'] * order['price']
                    from_val = order['size']
                    to_sym = 'USD'
                    from_sym = 'SYM'


                if order['filled']:
                    self.value[from_sym] -= from_val
                    ids_to_remove.append(order_id)
                    is_maker_order = order['is maker']
                    is_taker_order = not is_maker_order
                    self.value[to_sym] += to_val * (1 - maker_fee * is_maker_order - taker_fee * is_taker_order)
                else:
                    from_sym += ' Hold'
                    self.value[from_sym] += from_val

            for old_id in ids_to_remove:
                self.exchange.remove_order(side, old_id)

    def get_amnt_available(self, side):
        if side == 'asks':
            sym = 'SYM'
        elif side == 'bids':
            sym = 'USD'
        else:
            raise ValueError('side must be either "asks" or "bids"')
        available = self.value[sym] - self.value[sym + ' Hold']
        return available

class BackTestBot:
    current_price = {'asks': None, 'bids': None}
    order_stds = {}
    fills = None
    prior_price = None
    prior_value = 0
    order_books = None

    def __init__(self, model_path, strategy):
        # strategy is a class that tells to bot to either buy or sell or hold, and at what price to do so
        self.strategy = strategy
        self.model = cf.CryptoFillsModel('ETH', model_path=model_path, suppress_output=True)
        self.model.create_formatted_cbpro_data()
        self.portfolio = BackTestPortfolio()

    def load_model_data(self, historical_order_books_path, historical_fills_path, train_test_split):
        # Load all data
        historical_fills = pd.read_csv(historical_fills_path)
        historical_order_books = pd.read_csv(historical_order_books_path)
        if 'Unnamed: 0' in historical_order_books.columns.values:
            # Some data files have the above header containing the indices, this gets rid of it
            historical_order_books = historical_order_books.drop(['Unnamed: 0'], axis=1)
        if 'Unnamed: 0' in historical_fills.columns.values:
            # Some data files have the above header containing the indices, this gets rid of it
            historical_fills = historical_fills.drop(['Unnamed: 0'], axis=1)

        # Filter data so that only test data remains
        training_length = (int(len(historical_order_books) * (1 - train_test_split)))
        order_books = historical_order_books[training_length::]
        fills_ts_values = self.model.data_obj.str_list_to_timestamp(historical_fills.time.values)
        historical_fills_mask =  fills_ts_values > order_books.ts.values[0]
        self.fills = historical_fills[historical_fills_mask].values[0, ::]
        order_books = order_books.reset_index(drop=True)

        # Add filtered data to objects
        self.portfolio.exchange.order_books = order_books
        # self.model.data_obj.historical_order_books = order_books
        # self.model.data_obj.historical_fills = fills

    def get_order_book(self):
        order_book = self.portfolio.exchange.get_current_book()

        if self.order_books is None:
            self.order_books = order_book
        else:
            if len(self.order_books.index) >= 30:
                self.order_books = self.order_books.drop([0])
            self.order_books = self.order_books.append(order_book, ignore_index=True)

        return order_book

    def update_current_price(self):
        for side in ['asks', 'bids']:
            top_order = self.portfolio.exchange.get_top_order('asks')
            self.current_price[side] = top_order

    def place_order(self, price, side, size, allow_taker=True):
        order_id = self.portfolio.exchange.place_order(price, side, size, post_only=allow_taker)
        return order_id

    def fit_to_data(self, true_price, predicted):
        coeff = np.polyfit(true_price, predicted, 1)
        fit_data = coeff[0] * predicted + coeff[1]
        predict_point = fit_data[-1]

        return predict_point

    def predict(self):
        order_book = self.get_order_book()
        self.model.data_obj.historical_order_books = self.order_books
        full_prediction = self.model.model_actions('forecast')
        prices = self.order_books['0'].values
        int_len = 10
        if len(prices) > int_len:
            scaled_prediction = rescale_to_fit(full_prediction, prices)

            prediction = scaled_prediction
        else:
            prediction = np.array([0])

        if len(prediction) > 0:
            prediction_del = np.diff(prediction)
        else:
            prediction_del = 0

        return prediction_del, order_book

    def get_full_portfolio_value(self):
        self.update_current_price()
        price = np.mean([self.current_price['asks'], self.current_price['bids']])
        usd = self.portfolio.value['USD']
        sym = self.portfolio.value['SYM']
        full_value = usd + sym*price

        return full_value

    def cancel_out_of_bound_orders(self, side, price, order_std):
        # TODO take into account order reason (e.g. placing order at current price to hit future prediction vs placing order at projected future price)
        orders = self.portfolio.exchange.orders[side]
        keys_to_delete = []
        if side == 'bids':
            coeff = -1
        else:
            coeff = 1

        for id in orders.keys():
            # if coeff * orders[id]['price'] > coeff * price:
            if self.order_stds[id] > order_std:
                keys_to_delete.append(id)

        for id in keys_to_delete:
            self.portfolio.exchange.remove_order(side, id)

    def trade_action(self):
        prediction, order_book = self.predict()
        decision, order_std = self.strategy.determine_move(prediction, order_book, self.portfolio) # returns None for hold
        current_val = self.get_full_portfolio_value()
        if (self.portfolio.value['USD'] > 10):
            self.prior_value = current_val
        if (decision is not None):
            side = decision['side']
            price = decision['price']
            available = self.portfolio.get_amnt_available(side)
            # if (current_val >= self.prior_value * 1.035) or (side == 'bids'):
            if side == 'bids':
                size = available * decision['size coeff'] / decision['price']
            else:
                size = available * decision['size coeff']
            is_maker = decision['is maker']

            self.cancel_out_of_bound_orders(side, price, order_std)
            order_id = self.place_order(price, side, size, allow_taker=is_maker)
            self.order_stds[order_id] = order_std
            if order_id is None:
                price = self.prior_price
            self.prior_price = price
        else:
            price = self.prior_price

        return price

    def reset(self):
        self.portfolio.value = {'USD': 100, 'SYM': 0, 'USD Hold': 0, 'SYM Hold': 0}
        self.portfolio.exchange.orders = {'bids': {}, 'asks': {}}
        self.current_price = {'asks': None, 'bids': None}
        self.fills = None
        self.prior_price = None
        self.order_books = None

class MultiProcessingBackTestBot(BackTestBot):

    def __init__(self, model_path, strategy, predictions):
        BackTestBot.__init__(self, model_path, strategy)
        self.predictions = predictions

    def predict(self):
        # Keras's predict method does not support multiprocessing
        order_book = self.get_order_book()
        ind = self.portfolio.exchange.time+1
        prices = self.order_books['0'].values
        full_prediction = self.predictions[(ind - len(prices)):ind, 0]
        int_len = 10
        if len(prices) > int_len:
            scaled_prediction = rescale_to_fit(full_prediction, prices)

            prediction = scaled_prediction
        else:
            prediction = np.array([0])

        if len(prediction) > 0:
            prediction_del = np.diff(prediction)
        else:
            prediction_del = 0

        return prediction_del, order_book

def run_backtest(bot, data_queue, order_books, proc_id=0):

    times = np.array(order_books.index) - order_books.index[0]
    portfolio_history = np.array([])  # This will track the bot progress
    sym_start_portfolio_history = np.array([])
    predictions = np.array([])

    sym_start_portfolio = {'USD': 0, 'SYM': 1, 'USD Hold': 0, 'SYM Hold': 0}

    bot.portfolio.value = sym_start_portfolio
    sym_run = True
    ind = 0
    order_id = 0 # This allows segments of the history to be pushed early to avoid clogging the queue
    put_ind_limit = 1200
    next_put_ind = put_ind_limit
    did_sym_run_end_on_usd = None

    # TODO edit so that any orders remaining across segments can be used to determine the path of the next segment
    while ind < len(times):
        time = times[ind]
        ind += 1
        #progress_printer(len(times), time)
        bot.portfolio.exchange.time = time
        prediction = bot.trade_action()
        bot.portfolio.exchange.update_fill_status()
        bot.portfolio.update_value()
        val = bot.get_full_portfolio_value()
        # --This loop allows the bot to simulate what would happen if the last segment ended holding crypto--
        if sym_run:
            sym_start_portfolio_history = np.append(sym_start_portfolio_history, val)
            sym_val =  bot.portfolio.value['SYM']
            if sym_val <= 0:
                sym_run = False
                ind = 0
                bot.reset()
        else:
            portfolio_history = np.append(portfolio_history, val)
            predictions = np.append(predictions, prediction)

        if ind > next_put_ind:
            data_queue.put({'USD': portfolio_history, 'process id': proc_id, 'SYM': sym_start_portfolio_history, 'end state': None, 'sym run end state': None, 'seg id': order_id, 'predictions': predictions},
                           block=False)
            next_put_ind += put_ind_limit
            portfolio_history = np.array([])
            sym_start_portfolio_history = np.array([])
            predictions = np.array([])
            order_id += 1

        if sym_run & (ind == len(times)):
            sym_run = False
            ind = 0
            sym_val = bot.portfolio.value['SYM']
            usd_val = bot.portfolio.value['USD']
            bot.reset()
            did_sym_run_end_on_usd = sym_val <= usd_val


    sym_val = bot.portfolio.value['SYM']
    usd_val = bot.portfolio.value['USD']
    did_end_on_usd = sym_val <= usd_val  # This lets the program know which beginning to use
    if did_sym_run_end_on_usd is None:
        did_sym_run_end_on_usd = did_end_on_usd


    data_queue.put({'USD': portfolio_history, 'process id': proc_id, 'SYM': sym_start_portfolio_history, 'end state': did_end_on_usd, 'sym run end state': did_sym_run_end_on_usd, 'seg id': order_id, 'predictions': predictions}, block=False)

def stitch_process_segments(data):
    num_entries = len(data.keys())
    new_data = {'process id': data[0]['process id']}
    all_keys = data[0].keys()

    for key in all_keys:
        if key in new_data.keys():
            continue
        new_data[key] = np.array([])

    for ind in range(0, num_entries):
        current_data = data[ind]

        if current_data['end state'] is not None:
            # The end state is None for all but one entry
            new_data['end state'] = current_data['end state']

        if current_data['sym run end state'] is not None:
            # The end state is None for all but one entry
            new_data['sym run end state'] = current_data['sym run end state']

        for key in all_keys:
            if key in ['process id', 'end state', 'sym run end state']:
                continue
            new_data[key] = np.append(new_data[key], current_data[key])

    return new_data

def update_and_order_processes(procs, queue, full_len):
    data = [None for i in range(0, len(procs))]
    stop_loop = True
    completed_len = 0

    while stop_loop:

        stop_loop = np.any([( not x is None) for x in procs])

        for i in range(0, len(procs)):
            proc = procs[i]
            if proc is None:
                continue

            proc.join(timeout=1)
            while not queue.empty():
                temp_data = queue.get()
                usd_len = len(temp_data['USD'])
                sym_len = len(temp_data['SYM'])
                if usd_len > 0:
                    completed_len += usd_len
                    # TODO adjust full_len for short sym_len values
                else:
                    completed_len += sym_len

                if data[temp_data['process id']] is None:
                    data[temp_data['process id']] = {temp_data['seg id']: temp_data}  # Puts the segments in order
                else:
                    data[temp_data['process id']][temp_data['seg id']] = temp_data

                progress_printer(full_len, completed_len, digit_resolution=4, print_resolution=0)

            if not proc.is_alive():
                procs[i] = None

    for j in range(0, len(data)):
        new_entry = stitch_process_segments(data[j])
        data[j] = new_entry

    return data

def stitch_trade_histories(data):
    last_segment_did_end_on_usd = True
    last_segment_end_value = 100
    portfolio_history = np.array([])  # This will track the bot progress
    predictions = np.array([])

    for entry in data:
        if last_segment_did_end_on_usd:
            current_portfolio_history = entry['USD']
            norm_coeff = last_segment_end_value / entry['USD'][0]
            end_key = 'end state'
        else:
            usd_portfolio_history = entry['USD']
            sym_start_history = entry['SYM']
            usd_norm_coeff = entry['SYM'][0] / entry['USD'][0]
            norm_coeff = last_segment_end_value / entry['SYM'][0]
            current_portfolio_history = np.append(sym_start_history, usd_norm_coeff * usd_portfolio_history[len(sym_start_history)::])
            end_key = 'sym run end state'

        portfolio_history = np.append(portfolio_history, norm_coeff*current_portfolio_history)
        predictions = np.append(predictions, entry['predictions'])

        last_segment_did_end_on_usd = entry[end_key]
        last_segment_end_value = portfolio_history[-1]

    return portfolio_history, predictions

def run_backtests_in_parallel(model_path, strategy, historical_order_books_path, historical_fills_path, train_test_split=0.1, num_processes=1):

    # --Instantiate master bot which formats all data for the multiprocessing bots--
    bot = BackTestBot(model_path, strategy)
    print('Loading historical data')
    bot.load_model_data(historical_order_books_path, historical_fills_path, train_test_split)
    order_books = bot.portfolio.exchange.order_books
    full_len = len(order_books['0'].values)
    fills = bot.fills
    bot.model.data_obj.historical_order_books = order_books # The bot will make all predictions before the processing starts
    temp_input_arr = order_books.drop(['ts'], axis=1).values
    arr = temp_input_arr.reshape(temp_input_arr.shape[0], temp_input_arr.shape[1], 1)
    all_predictions = bot.model.model.predict(arr)

    # --Instantiate variables for process tracking--
    procs = []
    bots = []
    portfolio_history = np.array([])  # This will track the bot progress
    price_history = order_books['0'].values
    queue = Queue()
    order_book_chunks = np.array_split(order_books, num_processes)
    fill_chunks = np.array_split(fills, num_processes)
    prediction_chunks = np.array_split(all_predictions, num_processes)

    # --Setup logging for multuprocessing --
    logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG) # For some reason Pycharm does not recognize SUBDEBUG, however, it is valid
    logger.setLevel(logging.INFO)

    # --Start processing--
    start_time = time()
    for proc_id in range(0, num_processes):
        # Create temp bot
        bots.append(MultiProcessingBackTestBot(model_path, strategy, prediction_chunks[proc_id]))
        current_books = order_book_chunks[proc_id]
        current_fills = fill_chunks[proc_id]
        bots[proc_id].portfolio.exchange.order_books = current_books
        bots[proc_id].fills = current_fills
        proc = Process(target=run_backtest, args=(bots[proc_id], queue, current_books, proc_id))
        procs.append(proc)
        proc.start()

    data = update_and_order_processes(procs, queue, 2*full_len)

    portfolio_history, predictions = stitch_trade_histories(data)
    run_time = time() - start_time
    print(str(run_time))

    return portfolio_history, price_history, predictions



if __name__ == "__main__":


    model_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/Models/ETH/ETHmodel_3layers_30fill_leakyreluact_adamopt_mean_absolute_percentage_errorloss_60neurons_14epochs1553129847.019871.h5'
    strategy = Strategy()
    historical_order_books_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_historical_order_books_granular_031919.csv'
    historical_fills_path = '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/HistoricalData/order_books/ETH_fills_granular_031919.csv'

    algorithm_returns, market_returns, predictions = run_backtests_in_parallel(model_path, strategy, historical_order_books_path, historical_fills_path, train_test_split=0.003, num_processes=8)

    plt.plot(algorithm_returns, '--.r', label='algorithm')
    plt.plot(100*market_returns/market_returns[0], '--xb', label='market')
    plt.legend()
    plt.title('Returns')
    plt.ylabel('Value (%)')

    plt.figure()
    plt.plot(predictions, '--.r', label='prediction')
    plt.plot(market_returns, '--xb', label='market')
    plt.legend()
    plt.title('Price')
    plt.ylabel('Price ($)')

    plt.show()