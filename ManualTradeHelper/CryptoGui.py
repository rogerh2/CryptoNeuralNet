# Simple enough, just import everything from tkinter.
import numpy as np
import pandastable
from tkinter import *
from PIL import Image, ImageTk
from CryptoBot.SpreadBot import PSMPredictBot
from CryptoBot.CryptoBot_Shared_Functions import num2str
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from time import sleep
from time import time
from queue import Queue

# global constants
QUOTE_ORDER_MIN = 10
SYMS=('KNC', 'ATOM', 'OXT', 'LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP', 'BTC')

# Signals for the Queue
QUEUE_PAUSE_ORDER_TRACKER_UPDATES = 0
QUEUE_START_ORDER_TRACKER_UPDATES = 1
QUEUE_BUY = 2
QUEUE_SELL = 3
QUEUE_KILL_SIG = 4
QUEUE_PREDICTION_SLEEP = 5
QUEUE_PREDICTION_RESUME = 6

# Indicies for order dictionaries
SYM = 'sym'
LIMIT_PRICE = 'limit'
STOP_PRICE = 'stop'
TYPE = 'type'
SIZE = 'size'
SPREAD = 'spread'
CORRESPONDING_ID = 'corr'

# Names for Frame Widgets
WIDGET_HOME = 'home'
WIDGET_ORDERS = 'orders'
WIDGET_SLEEP = 'sleep'

# Supporting Classes
# These classes are for sub systems in individual GUI windows

# This class handles the menu bar

# This is a base frame to handle plots of cryptocurrency prices and their predictions
class CryptoCanvas:

    def __init__(self, master, bot, sym, size, row=0, column=0, column_span=4):
        # parameters that you want to send through the Frame class.
        self.master = master
        self.bot = bot
        self.sym = sym
        self.size = size
        self.row = row
        self.column_span = column_span
        self.column = column

        self.fig = figure = plt.Figure(figsize=self.size, dpi=100)
        self.ax = figure.add_subplot(111)

        # Plot the initial data
        self.raw_line, self.predict_line, self.previous_line, self.canvas, self.ax = self.plot()

    def plot(self):
        prediction = self.bot.predictions[self.sym]
        raw_data = self.bot.raw_data[self.sym]
        reversed_prediction = self.bot.reversed_predictions[self.sym]
        axis = self.ax

        raw_line, = axis.plot(np.arange(0, len(raw_data)), raw_data)
        predict_line, = axis.plot(np.arange(len(raw_data), len(prediction) + len(raw_data)), prediction)
        previous_line, = axis.plot(np.arange(len(raw_data) - len(reversed_prediction), len(raw_data)), reversed_prediction)

        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.show()
        canvas.get_tk_widget().grid(row=self.row, column=self.column, columnspan=self.column_span)

        return raw_line, predict_line, previous_line, canvas, axis

    def update_plot(self):
        # Get current prediction and price data (always plotted)
        raw_data = self.bot.raw_data[self.sym]
        prediction = self.bot.predictions[self.sym]
        reversed_prediction = self.bot.reversed_predictions[self.sym]

        # Update Lines
        self.previous_line.set_ydata(reversed_prediction)
        self.predict_line.set_ydata(prediction)
        self.raw_line.set_ydata(raw_data)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.canvas.draw()
        plt.pause(0.01)

#  This class is a link for the home page to go to the currency specific page
class CryptoLink:

    def __init__(self, master, bot, sym, target, row, column=0):
        self.master = master
        self.bot = bot
        self.wallet = bot.portfolio.wallets[sym]
        self.sym = sym
        self.row = row
        self.column = column
        self.avg_buy_price = 0
        self.command = lambda: target(sym)

        self.get_avg_buy_price()
        cmd_text = sym + '\nAvg Buy Price:' + num2str(self.avg_buy_price, digits=self.wallet.product.usd_decimal_num)
        self.button = Button(self.master, text=cmd_text, command=self.command)
        self.button.grid(row=row, column=column)

    def get_avg_buy_price(self):
        orders = self.bot.orders
        avg_buy_price = 0
        amnt_held = 0
        for id in orders.index:
            order = orders.loc[id]
            if order['side'] == 'sell':
                continue
            if not np.isnan(order['spread']):
                # If there is a spread listed the buy order is already taken care of
                continue
            if (self.sym in order['product_id']) and (order['side']=='buy'):
                avg_buy_price += float(order['price'])*order['filled_size']
                amnt_held += order['filled_size']

        if amnt_held:
            self.avg_buy_price =  avg_buy_price/amnt_held
        else:
            self.avg_buy_price =  0

    def update_text(self):
        self.get_avg_buy_price()
        cmd_text = self.sym + '\nAvg Buy Price:' + num2str(self.avg_buy_price, digits=self.wallet.product.usd_decimal_num)
        self.button.config(text=cmd_text)

# Frame Subclasses
# Here we add a class to represent the window for an individual currency
class CurrencyGui(Frame):

    def __init__(self, master, bot: PSMPredictBot, queue: Queue, sym: str):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master, name=sym.lower())
        self.master = master
        self.wallet = bot.portfolio.wallets[sym]
        self.bot = bot
        self.tracker = 0
        self.sym = sym
        self.queue = queue

        # parameters that are used only when set
        self.avg_buy_price = 0
        self.avg_buy_line = None # this is the line plotted to show the avg buy price
        self.min_sell_line = None

        # These lines show the current price entered, the minnimum sell to hit the specified profit (if you buy at that
        # price) and the maximum buy which could generate that profit (if you're selling
        self.current_price_line = None
        self.current_min_sell_line = None
        self.current_max_buy_line = None
        self.current_spread_line = None

        # Plot the initial data
        self.predict_plot_handler = CryptoCanvas(self, bot, sym, (6, 5))
        self.canvas = self.predict_plot_handler.canvas
        self.ax = self.predict_plot_handler.ax
        self.ax.set_title(self.sym + ' Prediction')
        self.ax.set_xlabel('Time (min)')
        self.ax.set_ylabel('Price ($)')
        self.plot()#self.showImg("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/test.png")

        # Create inputs for prices
        self.price_entry = self.display_text_entry('Price', 1, 0)
        self.spread_entry = self.display_text_entry('Spread', 2, 0)

        # Create buttons to trade
        self.buy_button = Button(self, text='Buy', command=self.buy)
        self.buy_button.grid(row=3, column=1)
        self.sell_button = Button(self, text='Sell', command=self.sell)
        self.sell_button.grid(row=3, column=2)

        # The refresh button updates the window information
        self.refresh_button = Button(self, text='Refresh', command=self.refresh)
        self.refresh_button.grid(row=3, column=3)

    def display_text_entry(self, text, row, column):
        Label(self, text=text).grid(row=row, column = column)
        entry = Entry(self)
        entry.grid(columnspan=2,row=row, column=(column+1))
        return entry

    def get_unhandled_buy_orders_ids(self):
        orders = self.bot.orders
        ids = []
        for id in orders.index:
            order = orders.loc[id]
            if order['side'] == 'sell':
                continue
            if not np.isnan(order['spread']):
                # If there is a spread listed the buy order is already taken care of
                continue
            if (self.sym in order['product_id']) and (order['side']=='buy'):
                ids.append(id)

        return ids

    def get_avg_buy_price(self):
        orders = self.bot.orders
        avg_buy_price = 0
        amnt_held = 0
        unhandled_ids = self.get_unhandled_buy_orders_ids()
        for id in unhandled_ids:
            order = orders.loc[id]
            avg_buy_price += float(order['price'])*order['filled_size']
            amnt_held += order['filled_size']

        if amnt_held:
            self.avg_buy_price =  avg_buy_price/amnt_held
        else:
            self.avg_buy_price =  0

    def get_price_and_spread(self):
        price = self.price_entry.get()
        spread = self.spread_entry.get()
        if price == '':
            price = 0
        else:
            price = float(price)
        if spread == '':
            spread = 0
        else:
            spread = float(spread)

        return price, spread

    def plot(self):
        axis = self.ax
        if self.avg_buy_price:
            xlimits = axis.get_xlim()
            self.avg_buy_line, = axis.plot(xlimits, self.avg_buy_price*np.ones(2), 'g')
            self.min_sell_line, = axis.plot(xlimits, self.bot.min_spread * self.avg_buy_price * np.ones(2), 'r')

        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.canvas.draw()
        plt.pause(0.01)

    def update_transient_line(self, line, value, color):
        #Plot straight horizontal lines that only appear a portion of the time

        # If the line exists and the price is non zero then update the line
        if (line is not None) and value:
            line.set_ydata(value * np.ones(2))
        # If the line does not exist and the price is non zero then create the line
        elif (line is None) and value:
            xlimits = self.ax.get_xlim()
            line, = self.ax.plot(xlimits, value * np.ones(2), color)
            self.ax.set_xlim(xlimits)
        # If the line exists but the price is zero then remove the line
        elif (line is not None) and not value:
            line.remove()
            line = None

        return line

    def plot_current_price_and_spread(self):
        price, spread = self.get_price_and_spread()
        self.current_price_line = self.update_transient_line(self.current_price_line, price, 'k')
        self.current_min_sell_line = self.update_transient_line(self.current_min_sell_line, self.bot.min_spread * price, 'r--')
        # if a spread is set then this is a buy order and we don't care about the max buy line
        if spread:
            self.current_spread_line = self.update_transient_line(self.current_spread_line, spread * price, 'k--')
            self.current_max_buy_line = self.update_transient_line(self.current_max_buy_line, 0, 'g--')
        else:
            self.current_spread_line = self.update_transient_line(self.current_spread_line, 0, 'k--')
            self.current_max_buy_line = self.update_transient_line(self.current_max_buy_line, price / self.bot.min_spread, 'g--')

    def update_plot(self):
        # Check for sometimes plotted lines
        self.avg_buy_line = self.update_transient_line(self.avg_buy_line, self.avg_buy_price, 'g')
        self.min_sell_line = self.update_transient_line(self.min_sell_line, self.bot.min_spread * self.avg_buy_price, 'r')
        self.plot_current_price_and_spread()

        self.predict_plot_handler.update_plot()

        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.canvas.draw()
        plt.pause(0.01)

    def buy(self):
        # This function places instructions for a buy order in the Queue
        price, spread = self.get_price_and_spread()
        if not spread:
            spread = np.nan
        if price:
            # Stop limit functionality available but not used as of now
            order = {TYPE:QUEUE_BUY, SYM:self.sym, LIMIT_PRICE:price, SPREAD:spread, STOP_PRICE:None}
            self.queue.put(order)
            print('Buy order queued\n')

    def sell(self):
        price, _ = self.get_price_and_spread()
        if price:
            ids = self.get_unhandled_buy_orders_ids()
            # pause the order tracking updates to add the new spreads to the order tracking
            self.queue.put({TYPE:QUEUE_PAUSE_ORDER_TRACKER_UPDATES})
            sleep(7) # wait 7 seconds to give the order_tracking time to respond
            for id in ids:
                order = self.bot.orders.loc[id]
                buy_price = order['price']
                nom_spread = price / buy_price
                size = order['filled_size']
                self.bot.orders.at[id, 'spread'] = nom_spread
                order = {TYPE: QUEUE_SELL, SYM: self.sym, LIMIT_PRICE: price, SIZE:size, STOP_PRICE: None}
                # self.queue.put(order)
                print('Sell Order for buy id: ' + id + ' queued\n')
            self.queue.put({TYPE: QUEUE_START_ORDER_TRACKER_UPDATES})


    def refresh(self):
        self.get_avg_buy_price()
        Label(self, text = self.sym + ' Available: ' + num2str(self.wallet.get_amnt_available('sell'), digits=self.wallet.product.base_decimal_num)).grid(row=1, column=3)
        Label(self, text = 'Average Buy Price: ' + num2str(self.avg_buy_price, digits=self.wallet.product.usd_decimal_num)).grid(row=2, column=3)
        self.update_plot()

# Here, we are creating a class to handle the home screen, which has links to all the individual currency screens
class HomeGui(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master, bot: PSMPredictBot, queue: Queue, syms=SYMS):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master, name=WIDGET_HOME)


        # reference to the master widget, which is the tk window
        self.master = master
        self.queue = queue

        canvases = [] # This holds the canvases that plot each prediction
        lables = [] # This holds the labels that show the current amount held
        col = 0
        row = 1


        for i in range(0, len(syms)):
            sym = syms[i]

            lable = CryptoLink(self, bot, sym, master.switch_frame, row+1, column=col)
            canvas = CryptoCanvas(self, bot, sym, (1.5,1.5), row=row, column=col, column_span=1)
            col += 1
            if col > 4:
                col = 0
                row += 2

            canvases.append(canvas)
            lables.append(lable)
        self.canvases = canvases
        self.lables = lables

    def refresh(self):
        for lable, canvas in zip(self.lables, self.canvases):
            lable.update_text()
            canvas.update_plot()

    def showImg(self, img_path, resize=False):
        load = Image.open(img_path)
        # Make the image smaller
        if resize:
            load = load.resize((250, 250), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.grid(row=1, column=0)

# Here we create a screen to allow editing the tracked orders
class Orders(Frame):
    def __init__(self, master, bot: PSMPredictBot, queue: Queue):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master, name=WIDGET_ORDERS)

        # reference to the master widget, which is the tk window
        self.master = master
        self.queue = queue
        self.bot = bot
        self.table = pt = pandastable.Table(self, dataframe=self.bot.orders, showtoolbar=True, showstatusbar=True)
        pt.show()

    def refresh(self):
        self.table.redraw()

# Here we create a screen to allow editing the tracked orders
class Sleep_Screen(Frame):
    def __init__(self, master):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master, name=WIDGET_SLEEP)

        # reference to the master widget, which is the tk window
        self.master = master
        label = Label(self, text='Sleep')
        label.grid()

    def refresh(self):
        pass

# This class controlls all the frames
class Controller(Tk):

    def __init__(self, bot: PSMPredictBot, handler, queue: Queue, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.bot = bot
        self.queue = queue
        self._frame = None
        self.geometry("800x600")
        self.switch_frame('Home')
        self.handler = handler
        self.menu()

    # Creats the menu
    def menu(self):
        # creating a menu instance
        menu = Menu(self)
        self.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option to refresh the window information
        file.add_command(label="Refresh", command=self.refresh)

        # adds a command to the menu option to close the window and stop the handler
        file.add_command(label="Exit", command=self.client_exit)

        # added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # create the navigate menu
        edit = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        home_command = lambda: self.switch_frame("Home")
        edit.add_command(label="Home", command=home_command)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        order_command = lambda: self.switch_frame("Orders")
        edit.add_command(label="Orders", command=order_command)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        order_command = lambda: self.switch_frame("Sleep")
        edit.add_command(label="Sleep", command=order_command)

        # added "file" to our menu
        menu.add_cascade(label="Navigate", menu=edit)

    def switch_frame(self, type):
        new_frame = None
        if type in SYMS:
            new_frame = CurrencyGui(self, self.bot, self.queue, type)
        elif (type=='Orders') and (WIDGET_ORDERS not in str(self._frame)):
            new_frame = Orders(self, self.bot, self.queue)
            self.queue.put({TYPE:QUEUE_PAUSE_ORDER_TRACKER_UPDATES})
        elif (type=='Sleep') and  (WIDGET_SLEEP not in str(self._frame)):
            new_frame = Sleep_Screen(self)
            self.queue.put({TYPE:QUEUE_PREDICTION_SLEEP})
        elif WIDGET_HOME not in str(self._frame):
            new_frame = HomeGui(self, self.bot, self.queue)

        if new_frame is not None:
            # If you are leaving the orders page start back auto updateing
            if (type != 'Orders') and  (WIDGET_ORDERS in str(self._frame)):
                self.queue.put({TYPE:QUEUE_START_ORDER_TRACKER_UPDATES})
            if (type != 'Sleep') and  (WIDGET_SLEEP in str(self._frame)):
                self.queue.put({TYPE:QUEUE_PREDICTION_RESUME})

            if self._frame is not None:
                self._frame.destroy()

            self._frame = new_frame
            self._frame.grid()

    def client_exit(self):
        self.queue.put({TYPE:QUEUE_KILL_SIG})
        exit()

    def refresh(self):
        self._frame.refresh()

# This class handles the execution of the frames and updates the bot
class Bot_Handler:

    keep_running = True
    desired_number_of_currencies = 5
    update_orders = True
    update_predictions = True

    def __init__(self, bot: PSMPredictBot):
        self.bot = bot
        # Ensure all values are initialized
        print('Initializing Values')
        predict_thread = Thread(target=self.bot.predict)
        portfolio_thread = Thread(target=self.bot.portfolio.update_value)
        predict_thread.start()
        portfolio_thread.start()
        portfolio_thread.join()
        predict_thread.join()
        print('Initialization Complete')
        self.gui_queue = Queue() # This queue handles passing information to the handler from the gui
        self.order_queue = Queue() # This queue handles pass order information internally
        self.root = Controller(self.bot, self, self.gui_queue)
        self.queued_orders = []

    def get_buy_order_size(self, wallet, buy_price):
        self.bot.portfolio.update_value()
        full_portfolio_value = self.bot.get_full_portfolio_value()
        usd_available = self.bot.portfolio.get_usd_available()
        # Determine the total order price
        nominal_order_price = full_portfolio_value / ( self.desired_number_of_currencies )

        if nominal_order_price < QUOTE_ORDER_MIN:
            # never try to place an order smaller than the minimum
            nominal_order_price = QUOTE_ORDER_MIN

        if nominal_order_price > (usd_available - QUOTE_ORDER_MIN):
            # If placing the nominal order leaves an unusable amount of money then only use available
            order_size = usd_available / buy_price
        else:
            order_size = nominal_order_price / buy_price

            # Always insure the order is small enough to go through in a reasonable amount of time
            mean_size, _, _ = wallet.product.get_mean_and_std_of_fill_sizes('asks', weighted=False)
            if order_size > (buy_price * mean_size):
                order_size = buy_price * mean_size

        return order_size

    def add_order_to_bot_tracking(self, order_id, sym, side, corresponding_order=0, spread=np.nan):
        if order_id is None:
            print(sym + ' ' + side + ' order rejected\n')
        else:
            print('Order placed\n')
            for i in range(0, 10):
                self.bot.add_order(order_id, sym, side, time(), corresponding_order, spread=spread)
                if order_id in self.bot.orders.index:
                    break
                else:
                    sleep(5)
            if order_id not in self.bot.orders.index:
                print(sym + ' order ' + order_id + ' did not save')

    def place_buy_order(self, order):
        sym = order[SYM]
        limit_price = order[LIMIT_PRICE]
        stop_price = order[STOP_PRICE]
        wallet = self.bot.portfolio.wallets[sym]
        mkr_fee, tkr_fee = self.bot.portfolio.get_fee_rate()

        size = self.get_buy_order_size(wallet, limit_price)
        size /= (1 + tkr_fee)

        order_id = self.bot.place_order(limit_price, 'buy', size, sym, post_only=False, stop_price=stop_price, time_out=True)
        self.add_order_to_bot_tracking(order_id, sym, 'buy', spread=order[SPREAD])

    def place_sell_order(self, order):
        sym = order[SYM]
        limit_price = order[LIMIT_PRICE]
        stop_price = order[STOP_PRICE]
        size = order[SIZE]

        order_id = self.bot.place_order(limit_price, 'sell', size, sym, post_only=False, stop_price=stop_price)
        self.add_order_to_bot_tracking(order_id, sym, 'sell', corresponding_order=order[CORRESPONDING_ID])


    def place_order(self, order):
        if order[TYPE] == QUEUE_BUY:
            self.place_buy_order(order)
        elif order[TYPE] == QUEUE_SELL:
            pass

    def queue_handler(self):
        Q = self.gui_queue
        # The bot handler is the only one that modifies the bot object except for the tracked orders
        while self.keep_running:
            q_input = Q.get()
            input_type = q_input['type']
            # When the order editor screen is up, pause the automatic order tracking updates
            if input_type == QUEUE_PAUSE_ORDER_TRACKER_UPDATES:
                self.update_orders = False
                print('Order Tracking Paused')
            elif input_type == QUEUE_START_ORDER_TRACKER_UPDATES:
                self.update_orders = True
                print('Order Tracking Resumed')
            elif input_type == QUEUE_PREDICTION_SLEEP:
                self.update_predictions = False
                print('Predictions Paused')
            elif input_type == QUEUE_PREDICTION_RESUME:
                self.update_predictions = True
                print('Predictions Resumed')
            elif (input_type == QUEUE_BUY) or (input_type == QUEUE_SELL):
                self.order_queue.put(q_input)
            elif input_type == QUEUE_KILL_SIG:
                self.keep_running = False

    def predict(self):
        prediction_refresh_time_s = 3 * 60
        t0 = time()
        while self.keep_running:
            if ((time() - t0) > prediction_refresh_time_s) and (self.update_predictions):
                self.bot.predict()
                t0 = time()
            sleep(5)

    def bot_thread(self):
        # This thread handles the interactions with coinbase, only one thread handles this to avoid timeouts
        refresh_time_s = 60
        t0 = time()
        while self.keep_running:
            if (time() - t0) > refresh_time_s:
                self.bot.portfolio.update_value()
                mkr_fee, tkr_fee = self.bot.portfolio.get_fee_rate()
                self.bot.update_min_spread(mkr_fee, tkr_fee=tkr_fee)
                if self.update_orders:
                    self.bot.update_orders()
                    # If you can currently able to update the tracked orders, place all queued orders
                    while not self.order_queue.empty():
                        order = self.order_queue.get_nowait()
                        self.place_order(order)
                    # This part handles automatic sell orders
                    self.bot.place_limit_sells()
            sleep(5)

    def run(self):
        predict_thread = Thread(target=self.predict)
        portfolio_thread = Thread(target=self.bot_thread)
        queue_thread = Thread(target=self.queue_handler)
        predict_thread.start()
        queue_thread.start()
        portfolio_thread.start()
        self.root.mainloop()


if __name__ == "__main__":
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows.
    # root = Tk()
    #
    # root.geometry("800x600")

    # creation of an instance
    api_input = input('What is the api key? ')
    secret_input = input('What is the secret key? ')
    passphrase_input = input('What is the passphrase? ')
    psmbot = PSMPredictBot(api_input, secret_input, passphrase_input)
    mkr_fee, tkr_fee = psmbot.portfolio.get_fee_rate()
    psmbot.update_min_spread(mkr_fee, tkr_fee=tkr_fee)
    # psmbot.predict(verbose_on=True)
    # for sym in SYMS:
    #     psmbot.predictions[sym] = np.arange(0, 30) + 10
    #     psmbot.reversed_predictions[sym] = np.arange(-30, 0) + 10
    #     psmbot.raw_data[sym] = np.arange(-480, 0) + 10

    bot_handler = Bot_Handler(psmbot)
    bot_handler.run()

    #116809