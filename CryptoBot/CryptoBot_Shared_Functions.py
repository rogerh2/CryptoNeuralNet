import pytz
import numpy as np
import scipy.stats
from tzlocal import get_localzone
from datetime import datetime


def num2str(num, digits):
    # This function formats numbers as strings with the desired number of digits
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

def progress_printer(total_len, current_ind, start_ind=0, digit_resolution=1, tsk='Task', supress_output=False):

    if not supress_output:
        progress_percent = 100*(current_ind-start_ind)/(total_len-start_ind)
        resolution = 10**-(digit_resolution+2)

        if 1 >= (total_len - start_ind)*resolution:
            print (tsk + ' is ' + num2str(progress_percent, digit_resolution) + '% Complete')
        else:
            relevant_inds = range(start_ind, total_len, round((total_len - start_ind)*resolution))
            if current_ind in relevant_inds:
                print(tsk + ' is ' + num2str(progress_percent, digit_resolution) + '% Complete')

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