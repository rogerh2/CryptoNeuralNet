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

SYMS=('KNC', 'ATOM', 'OXT', 'LTC', 'LINK', 'ZRX', 'XLM', 'ALGO', 'ETH', 'EOS', 'ETC', 'XRP', 'XTZ', 'BCH', 'DASH', 'REP', 'BTC')

# Here we add a class to represent the window for an individual currency
class CurrencyGui(Frame):

    def __init__(self, master, controller, bot, sym):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.master = master
        self.controller = controller
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
        self.raw_line, self.predict_line, self.previous_line, self.canvas, self.ax = self.plot()#self.showImg("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/test.png")

        # Create inputs for prices
        self.price_entry = self.display_text_entry('Price', 1, 0)
        self.spread_entry = self.display_text_entry('Spread', 2, 0)

        # Create buttons to trade
        self.buy_button = Button(master, text='Buy', command=self.buy)
        self.buy_button.grid(row=3, column=1)
        self.sell_button = Button(master, text='Sell', command=self.sell)
        self.sell_button.grid(row=3, column=2)

        # The refresh button updates the window information
        self.refresh_button = Button(master, text='Refresh', command=self.refresh_portfolio)
        self.refresh_button.grid(row=3, column=3)

    def display_text_entry(self, text, row, column):
        Label(self.master, text=text).grid(row=row, column = column)
        entry = Entry(self.master)
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
        figure = plt.Figure(figsize=(6, 5), dpi=100)
        axis = figure.add_subplot(111)

        prediction = self.bot.predictions[self.sym]
        raw_data = self.bot.raw_data[self.sym]
        reversed_prediction = self.bot.reversed_predictions[self.sym]

        raw_line, = axis.plot(np.arange(0, len(raw_data)), raw_data)
        predict_line, = axis.plot(np.arange(len(raw_data), len(prediction) + len(raw_data)), prediction)
        previous_line, = axis.plot(np.arange(len(raw_data) - len(reversed_prediction), len(raw_data)), reversed_prediction)

        if self.avg_buy_price:
            xlimits = axis.get_xlim()
            self.avg_buy_line, = axis.plot(xlimits, self.avg_buy_price*np.ones(2), 'g')
            self.min_sell_line, = axis.plot(xlimits, self.bot.min_spread * self.avg_buy_price * np.ones(2), 'r')

        axis.set_title(self.sym + ' Prediction')
        axis.set_xlabel('Time (min)')
        axis.set_ylabel('Price ($)')

        canvas = FigureCanvasTkAgg(figure, master=self.master)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, columnspan=4)

        return raw_line, predict_line, previous_line, canvas, axis

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
        # Get current prediction and price data (always plotted)
        raw_data = self.bot.raw_data[self.sym]
        prediction = self.bot.predictions[self.sym]
        reversed_prediction = self.bot.reversed_predictions[self.sym]

        # Check for sometimes plotted lines
        self.avg_buy_line = self.update_transient_line(self.avg_buy_line, self.avg_buy_price, 'g')
        self.min_sell_line = self.update_transient_line(self.min_sell_line, self.bot.min_spread * self.avg_buy_price, 'r')
        self.plot_current_price_and_spread()

        # Update Lines
        self.previous_line.set_ydata(reversed_prediction)
        self.predict_line.set_ydata(prediction)
        self.raw_line.set_ydata(raw_data)
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

    def refresh_portfolio(self):
        self.bot.portfolio.update_value()
        self.bot.update_orders()
        self.get_avg_buy_price()
        Label(self.master, text = self.sym + ' Available: ' + num2str(self.wallet.get_amnt_available('sell'), digits=8)).grid(row=1, column=3)
        Label(self.master, text = 'Average Buy Price: ' + num2str(self.avg_buy_price, digits=8)).grid(row=2, column=3)
        self.update_plot()

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class HomeGui(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master, controller, bot, syms=SYMS):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        currency_windows = []
        for sym in syms:
            currency_windows.append(CurrencyGui(master, controller, bot, sym))

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Home")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        # added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # create the file object)
        edit = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        command = lambda: self.showImg("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/test.png")
        edit.add_command(label="Portfolio", command=command)

        # added "file" to our menu
        menu.add_cascade(label="Navigate", menu=edit)

    def showImg(self, img_path, resize=False):
        load = Image.open(img_path)
        # Make the image smaller
        if resize:
            load = load.resize((250, 250), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def client_exit(self):
        exit()

class Controller(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)
        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

def sleepy_predict(bot, root):
    sleep(180)
    print('starting next prediction')
    bot.predict(verbose_on=True)  #bot.predictions['BTC'] += 100  #
    root.update()

if __name__ == "__main__":
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows.
    root = Tk()

    root.geometry("800x600")

    # creation of an instance
    api_input = input('What is the api key? ')
    secret_input = input('What is the secret key? ')
    passphrase_input = input('What is the passphrase? ')
    psmbot = PSMPredictBot(api_input, secret_input, passphrase_input)
    # psmbot.predict(verbose_on=True)
    psmbot.predictions['LTC'] = np.arange(0, 30) + 10
    psmbot.reversed_predictions['LTC'] = np.arange(-30, 0) + 10
    psmbot.raw_data['LTC'] = np.arange(-480, 0) + 10
    app = CurrencyGui(root, None, psmbot, 'BCH')#Window(root)
    #
    # # mainloop
    # predict_thread = Thread(target=sleepy_predict, args=(psmbot,root))
    # predict_thread.start()
    root.mainloop()