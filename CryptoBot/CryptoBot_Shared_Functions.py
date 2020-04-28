import pytz
import numpy as np
import scipy.stats
import traceback
import dropbox
import sys
import keras
import pandas as pd
from tzlocal import get_localzone
from datetime import datetime
from datetime import timedelta
from scipy.signal import find_peaks
from time import sleep
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError
from matplotlib import pyplot as plt
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU



def num2str(num, digits=2, round_down=True):
    # This function formats numbers as strings with the desired number of digits

    if round_down:
        fmt_str = "{:0." + str(digits + 1) + "f}"
        num_str = fmt_str.format(num)[0:-1]
    else:
        fmt_str = "{:0." + str(digits) + "f}"
        num_str = fmt_str.format(num)


    return num_str


def convert_coinbase_timestr_to_timestamp(date_str):
    iso_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    if '.' not in date_str:
        date_str = date_str[0:-1] + '.000Z'

    trade_ts = datetime.timestamp(pytz.UTC.localize(datetime.strptime(date_str, iso_fmt)))

    return trade_ts


def get_current_tz():
    # This function returns the common symbol for a timezone (e.g. 'EST' instead of 'America/New_York')
    now = datetime.now(get_localzone())
    tz = now.strftime('%Z')

    return tz


def convert_time_to_uct(naive_date_from, tz_str=None):
    # This function converts any time object to utc

    if not tz_str:
        tz = get_localzone()
    else:
        tz = pytz.timezone(tz_str)
    sys_tz_date_from = tz.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = sys_tz_date_from.astimezone(utc)
    return utc_date


def str_list_to_timestamp(datetime_str_list, fmt='%Y-%m-%dT%H:%M:%S'):
    utc = pytz.UTC
    # TODO fix this so that timestamps without the float will not error
    localized_datetime_objects = [utc.localize(datetime.strptime(string[0:19], fmt)) for string in datetime_str_list]
    time_stamps = np.array([dt.timestamp() for dt in localized_datetime_objects])

    return time_stamps


def progress_printer(total_len, current_ind, start_ind=0, digit_resolution=1, print_resolution=None, tsk='Task', suppress_output=False):

    if print_resolution is None:
        # Print resolutions is the number of digits to print whereas digit resolution is how small of changes should be
        # registered, in most cases these are the same
        print_resolution = digit_resolution

    if not suppress_output:
        progress_percent = 100*(current_ind-start_ind)/(total_len-start_ind)
        resolution = 10**-(digit_resolution+2)

        if 1 >= (total_len - start_ind)*resolution: # Print everything if the resolution is smaller than the spacing
            print (tsk + ' is ' + num2str(progress_percent, print_resolution) + '% Complete')
        else:
            relevant_inds = range(start_ind, total_len, round((total_len - start_ind)*resolution))
            if current_ind in relevant_inds:
                print(tsk + ' is ' + num2str(progress_percent, print_resolution) + '% Complete')

    else:
        pass


def rescale_to_fit(data, data_to_scale_to):
    standard_data = (data - np.mean(data))/np.std(data)
    scaled_data = standard_data*np.std(data_to_scale_to) + np.mean(data_to_scale_to)

    return scaled_data


def create_number_from_bools(*args):

    bool_str = '0b'
    for arg in args:
        bool_str += str(int(arg))

    bool_num = eval(bool_str)

    return bool_num


def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def multiple_choice_question_with_prompt(prompt_str):

    input_val = 'maybe'

    while input_val not in ['yes', 'no']:
        input_val = input(prompt_str)
        if input_val == 'yes':
            bool_val = True
        elif input_val == 'no':
            bool_val = False

        if input_val not in ['yes', 'no']:
            print('Must answer yes or no')

    return bool_val


def fit_to_data(data, data_to_scale_to):
    coeff = np.polyfit(data, data_to_scale_to, 3)
    fit_data = coeff[1]*data + coeff[0]
    predict_point = fit_data[-1]

    return predict_point


def print_err_msg(section_text, e, err_counter):
    sleep(5)  # Most errors are connection related, so a short time out is warrented
    err_counter += 1
    print('failed to' + section_text + ' due to error: ' + str(e))
    print('number of consecutive errors: ' + str(err_counter))
    print(traceback.format_exc())

    return err_counter


def current_est_time():
    naive_date_from = datetime.utcnow()
    utc = pytz.timezone('UTC')
    est_date_from = utc.localize(naive_date_from)
    est = pytz.timezone('America/New_York')
    est_date = est_date_from.astimezone(est)
    return est_date


def offset_current_est_time(minute_offset, fmt=None):
    dt = current_est_time() - timedelta(minutes=minute_offset)
    if not fmt is None:
        return dt.strftime(fmt)
    else:
        return dt


def zero(data):
    return np.zeros(len(data))


def nth_max_ind(arr, n=1):
    return arr.argsort()[::-1][n-1]


def nth_max_peaks(arr, n=1):
    peak_inds, peak_data = find_peaks(arr, height=(None, None))
    nth_peak_ind = nth_max_ind(peak_data['peak_heights'], n=n)
    return peak_inds[nth_peak_ind]

def calculate_spread(buy_price, sell_price):
    return 1 + (sell_price - buy_price) / buy_price

def save_file_to_dropbox(data_path, file_path, access_token):
    dbx = dropbox.Dropbox(access_token)

    with open(data_path, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        try:
            dbx.files_upload(f.read(), file_path, mode=WriteMode('overwrite'))
            print(file_path + ' uploaded!')
        except ApiError as err:
            # This checks for the specific error where a user doesn't have
            # enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().reason.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()

class BaseNN:
    # This class is meant to serve as a parent class to neural net based machine learning classes

    def __init__(self, model_type=models.Sequential(), model_path=None, seed=7):
        np.random.seed(seed)
        if model_path is None:
            self.model = model_type
        else:
            self.model = models.load_model(model_path)

        self.seed = seed

    def train_model(self, training_input, training_output, epochs, file_name=None, retrain_model=False, shuffle=True, val_split=0.25, batch_size=96, training_patience=2, min_training_delta=0, training_e_stop_monitor='val_loss'):
        if retrain_model:
            print('re-trianing model')
            self.model.reset_states()

        estop = keras.callbacks.EarlyStopping(monitor=training_e_stop_monitor, min_delta=min_training_delta, patience=training_patience, verbose=0, mode='auto')

        hist = self.model.fit(training_input, training_output, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=shuffle, validation_split=val_split, callbacks=[estop])

        if not file_name is None:
            self.model.save(file_name)

        return hist

    def test_model(self, test_input, test_output, show_plots=True, x_indices=None, prediction_names=('Predicted', 'Measured'), prediction_styles=('rx--', 'b.--'), x_label=None, y_label=None, title=None):
        prediction = self.model.predict(test_input)
        if prediction.shape[1] == 0:
            prediction = prediction[::, 0] # For some reason the predictions come out 2D (e.g. [[p1,...,pn]] vs [p1,...,pn]]

        if show_plots:
            # Plot the price and the predicted price vs time
            if x_indices is None:
                plt.plot(prediction, prediction_styles[0])
                plt.plot(test_output, prediction_styles[1])
                plt.legend(prediction_names[0], prediction_names[1])
            else:
                df = pd.DataFrame(data={prediction_names[0]: test_output, prediction_names[1]: prediction}, index=x_indices)
                df.Predicted.plot(style=prediction_styles[0])
                df.Actual.plot(style=prediction_styles[1])

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)

            # Plot the correlation between price and predicted
            plt.figure()
            plt.plot(prediction, test_output, 'b.')
            plt.ylabel(prediction_names[0])
            plt.xlabel(prediction_names[1])
            plt.title('Correlation Between ' + prediction_names[0] + ' and ' + prediction_names[1])

            plt.show()


        return {prediction_names[0]:prediction, prediction_names[1]:test_output}

class ClassifierNN(BaseNN):

    def __init__(self, model_type=Sequential(), N=3, input_size=17, model_path=None, seed=7):
        super(ClassifierNN, self).__init__(model_type, model_path, seed)
        if model_path is None:
            self.model.add(Dense(30, input_dim=input_size, activation='relu'))
            self.model.add(LeakyReLU())
            self.model.add(Dense(N, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __call__(self):
        return self.model


# def rate_limited_get(func, limit, )