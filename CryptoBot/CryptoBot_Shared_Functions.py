import pytz
import numpy as np
import scipy.stats
from tzlocal import get_localzone
from datetime import datetime
from time import sleep
import traceback


def num2str(num, digits=2, round_down=True):
    # This function formats numbers as strings with the desired number of digits

    if round_down:
        fmt_str = "{:0." + str(digits + 1) + "f}"
        num_str = fmt_str.format(num)[0:-1]
    else:
        fmt_str = "{:0." + str(digits) + "f}"
        num_str = fmt_str.format(num)


    return num_str

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

        if 1 >= (total_len - start_ind)*resolution:
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
    naive_date_from = datetime.now()
    utc = pytz.timezone('UTC')
    est_date_from = utc.localize(naive_date_from)
    est = pytz.timezone('America/New_York')
    est_date = est_date_from.astimezone(est)
    return est_date

# def rate_limited_get(func, limit, )