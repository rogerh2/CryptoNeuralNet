# Simple enough, just import everything from tkinter.
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from CryptoBot.SpreadBot import Product, Wallet, LiveRunSettings, CombinedPortfolio, Bot, PortfolioTracker, PSMPredictBot
from CryptoBot.CryptoBot_Shared_Functions import num2str
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from time import sleep
import pandastable

SYMS=('KNC', 'ATOM', 'OXT', 'LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP', 'BTC')

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

    def __init__(self, master, bot, sym):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.master = master
        self.wallet = bot.portfolio.wallets[sym]
        self.bot = bot
        self.tracker = 0
        self.sym = sym

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

    def get_avg_buy_price(self):
        orders = self.bot.orders
        avg_buy_price = 0
        amnt_held = 0
        for id in orders.index:
            order = orders.loc[id]
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
        price, spread = self.get_price_and_spread()
        if price:
            print('Price: ' + num2str(price))
            if spread:
                print('Spread: ' + num2str(spread))

    def sell(self):
        price, spread = self.get_price_and_spread()
        if price:
            print('Price: ' + num2str(price))

    def refresh(self):
        self.get_avg_buy_price()
        Label(self, text = self.sym + ' Available: ' + num2str(self.wallet.get_amnt_available('sell'), digits=self.wallet.product.base_decimal_num)).grid(row=1, column=3)
        Label(self, text = 'Average Buy Price: ' + num2str(self.avg_buy_price, digits=self.wallet.product.usd_decimal_num)).grid(row=2, column=3)
        self.update_plot()

# Here, we are creating a class to handle the home screen, which has links to all the individual currency screens
class HomeGui(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master, bot, syms=SYMS):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)


        # reference to the master widget, which is the tk window
        self.master = master

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
    def __init__(self, master, bot):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master
        self.bot = bot
        self.table = pt = pandastable.Table(self, dataframe=self.bot.orders, showtoolbar=True, showstatusbar=True)
        pt.show()

    def refresh(self):
        self.table.redraw()

# This class controlls all the frames
class Controller(Tk):

    def __init__(self, bot, handler, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.bot = bot
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

        # added "file" to our menu
        menu.add_cascade(label="Navigate", menu=edit)

    def switch_frame(self, type):
        if type in SYMS:
            new_frame = CurrencyGui(self, self.bot, type)
        elif type=='Orders':
            new_frame = Orders(self, self.bot)
        else:
            new_frame = HomeGui(self, self.bot)

        if self._frame is not None:
            self._frame.destroy()

        self._frame = new_frame
        self._frame.grid()

    def client_exit(self):
        # TODO Use a queue to send a stop signal
        exit()

    def refresh(self):
        self._frame.refresh()

# This class handles the execution of the frames and updates the bot
class Bot_Handler:

    keep_running = True

    def __init__(self, bot):
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
        self.root = Controller(self.bot, self)

    def predict(self):
        while self.keep_running:
            self.bot.predict()
            sleep(5*60)

    def update_portfolio(self):
        while self.keep_running:
            self.bot.portfolio.update_value()
            self.bot.update_orders()
            mkr_fee, tkr_fee = self.bot.portfolio.get_fee_rate()
            self.bot.update_min_spread(mkr_fee,tkr_fee=tkr_fee)
            sleep(60)

    def run(self):
        predict_thread = Thread(target=self.predict)
        portfolio_thread = Thread(target=self.update_portfolio)
        predict_thread.start()
        portfolio_thread.start()
        self.root.mainloop()
        self.keep_running = False


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
    # psmbot.predict(verbose_on=True)
    # for sym in SYMS:
    #     psmbot.predictions[sym] = np.arange(0, 30) + 10
    #     psmbot.reversed_predictions[sym] = np.arange(-30, 0) + 10
    #     psmbot.raw_data[sym] = np.arange(-480, 0) + 10

    bot_handler = Bot_Handler(psmbot)
    bot_handler.run()

    #116809