from tzlocal import get_localzone
import pytz
from datetime import datetime

def get_current_tz():
    now = datetime.now(get_localzone())
    tz = now.strftime('%Z')

    return tz

def convert_time_to_uct(naive_date_from, tz_str=None):
    if not tz_str:
        tz = get_localzone()
    else:
        tz = pytz.timezone(tz_str)
    sys_tz_date_from = tz.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = sys_tz_date_from.astimezone(utc)
    return utc_date