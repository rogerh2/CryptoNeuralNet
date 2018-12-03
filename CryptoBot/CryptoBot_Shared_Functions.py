from tzlocal import get_localzone
import pytz
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