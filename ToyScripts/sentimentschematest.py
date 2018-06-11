import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt
from CryptoPredict.CryptoPredict import CryptoCompare
from textblob import TextBlob as txb
from datetime import datetime

cryp_obj = CryptoCompare(date_from="2018-06-9 23:00:00 EST", date_to="2018-06-10 23:00:00 EST")
price_df = cryp_obj.minute_price_historical(symbol='ETH')
sentiment_schema_1 = [] #sentiment sum
sentiment_schema_2 = [] #number of articles in last 5hrs
sentiment_schema_3 = [] #number of articles in last 5hrs plus news sum in last 5 hrs

def convert_time_to_uct(naive_date_from):
    est = pytz.timezone('America/New_York')
    est_date_from = est.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = est_date_from.astimezone(utc)
    return utc_date
total_len = len(price_df)
price_df = price_df.set_index('date')
iterations_complete = 0

def roge_normalization(arr):
    arr = np.array(arr)
    zerod_arr = arr - np.min(arr)
    norm_arr = zerod_arr/np.max(zerod_arr)
    return norm_arr

def convert_time_to_uct(naive_date_from):
    est = pytz.timezone('America/New_York')
    est_date_from = est.localize(naive_date_from)
    utc = pytz.UTC
    utc_date = est_date_from.astimezone(utc)
    return utc_date


n = 4500

last_news = None

date_len = len(price_df.index.values)

for i in range(1, date_len + 1):
    ind = date_len - i
    current_dt = price_df.index.values[ind]
    current_dt = pd.to_datetime(current_dt)
    utc_current_dt = convert_time_to_uct(current_dt)
    current_ts = utc_current_dt.timestamp()

    if last_news is not None:
        last_news_publication_times = [news['published_on'] < current_ts for news in last_news]
        if all(last_news_publication_times):
            current_news = last_news
        else:
            current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' EST')
    else:
        current_news = cryp_obj.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S') + ' EST')

    last_news = current_news

    current_sentiment = [n*txb(news['title']).sentiment.polarity/(n + current_ts - news['published_on']) for news in current_news] #14400 converts to time stamp

    sentiment_sum = np.sum(current_sentiment)
    sentiment_schema_1.insert(0, sentiment_sum)

    utc_current_dt = convert_time_to_uct(current_dt)
    delta_ts = utc_current_dt.timestamp() - 5*3600
    #news_count = np.sum([news['published_on'] > delta_ts for news in current_news])
    weighted_news_count = np.sum([n * (news['published_on'] > delta_ts) / (n + current_ts - news['published_on']) for news in current_news])
    sentiment_schema_2.insert(0, weighted_news_count)

    delta_ts = utc_current_dt.timestamp() - 5 * 3600
    news_count = np.sum([(news['published_on'] > delta_ts) for news in current_news])
    news_count_sentiment_sum = np.mean(current_sentiment[0:(news_count)])

    if np.isnan(news_count_sentiment_sum):
        news_count_sentiment_sum = 0


    sentiment_schema_3.insert(0, news_count_sentiment_sum)

    iterations_complete += 1
    print(str(round(100*iterations_complete/total_len, 1)) + '% complete')


open_price_df = pd.DataFrame({'Price':roge_normalization(price_df.ETH_high.values)}, index=price_df.index)
df_1 = pd.DataFrame({'Score':roge_normalization(sentiment_schema_1)}, index=price_df.index)
df_2 = pd.DataFrame({'Score':roge_normalization(sentiment_schema_2)}, index=price_df.index)
df_3 = pd.DataFrame({'Score':roge_normalization(sentiment_schema_3)}, index=price_df.index)

ax1 = df_1.plot(style='r--')
open_price_df.plot(ax=ax1)
plt.title('Schema 1')

ax2 = df_2.plot(style='r--')
open_price_df.plot(ax=ax2)
plt.title('Schema 2')

ax3 = df_3.plot(style='r--')
open_price_df.plot(ax=ax3)
plt.title('Schema 3')

plt.show()