from CryptoPredict.CryptoPredict import increase_saved_dataset_length
from datetime import datetime


def create_new_dataset():
    date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00 EST'
    pickle_path = input('Enter path to old data here: ')
    increase_saved_dataset_length(pickle_path, date_to)
