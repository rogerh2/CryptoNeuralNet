from CryptoPredict.CryptoPredict import increase_saved_dataset_length
from datetime import datetime
import numpy as np


def update_old_dataset():
    date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00 EST'
    pickle_path = input('Enter path to old data here: ')
    increase_saved_dataset_length(pickle_path, date_to)

def create_arrays_for_price_probability_scatter(prices, hours):
    prices = prices[-hours*60::]
    x = np.arange(np.min(prices), np.max(prices), 0.01)
    y = np.array([])

    for i in range(0, len(x)):
        probability_of_return = np.sum(prices > x[i])/len(prices)
        y = np.append(y, probability_of_return)

    return x, y