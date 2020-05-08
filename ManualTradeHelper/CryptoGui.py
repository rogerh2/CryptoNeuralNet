# Simple enough, just import everything from tkinter.
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from CryptoBot.SpreadBot import Product, Wallet, LiveRunSettings, CombinedPortfolio, Bot, PortfolioTracker, PSMPredictBot
from CryptoBot.CryptoBot_Shared_Functions import num2str

# Here we add a class to represent the links for an individual currency
class CurrencyGui(Frame):

    def __init__(self, master, controller, wallet, bot, propogator):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.master = master
        self.controller = controller
        self.wallet = wallet
        self.bot = bot
        self.tracker = 0

        self.showImg("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/test.png")
        self.price_entry = self.display_text_entry('Price', 1, 0)
        self.spread_entry = self.display_text_entry('Spread', 2, 0)
        self.buy_button = Button(master, text='Buy', command=self.buy)
        self.buy_button.grid(row=3, column=1)
        self.sell_button = Button(master, text='Sell', command=self.sell)
        self.sell_button.grid(row=3, column=2)
        self.refresh_button = Button(master, text='Refresh', command=self.refresh_portfolio)
        self.refresh_button.grid(row=3, column=3)

    def display_text_entry(self, text, row, column):
        Label(self.master, text=text).grid(row=row, column = column)
        entry = Entry(self.master)
        entry.grid(columnspan=2,row=row, column=(column+1))
        return entry

    def showImg(self, img_path, resize=True, h=500, w=500):
        load = Image.open(img_path)
        # Make the image smaller
        if resize:
            load = load.resize((h, w), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self.master, image=render)
        img.image = render
        img.grid(row=0, columnspan=4)

    def get_price_and_spread(self):
        price = self.price_entry.get()
        spread = self.spread_entry.get()
        if price == '':
            price = None
        else:
            price = float(price)
        if spread == '':
            spread = None
        else:
            spread = float(spread)

        return price, spread

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
        Label(self.master, text='Sym in Portfolio: ' + num2str(self.tracker)).grid(row=1, column=3)
        Label(self.master, text='Average Buy Price: ' + num2str(self.tracker)).grid(row=2, column=3)
        self.tracker += np.random.rand()




# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("GUI")

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
        edit.add_command(label="Show Img", command=command)
        edit.add_command(label="Show Text", command=self.showText)

        # added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)

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

    def showText(self):
        text = Label(self, text="Hey there good lookin!")
        text.pack()

    def client_exit(self):
        exit()



if __name__ == "__main__":
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows.
    root = Tk()

    root.geometry("800x600")

    # creation of an instance
    app = CurrencyGui(root, None, None, None, None)#Window(root)

    # mainloop
    root.mainloop()