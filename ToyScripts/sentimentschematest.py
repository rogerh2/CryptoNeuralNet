import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt
from CryptoPredict.CryptoPredict import cryptocompare
from textblob import TextBlob as txb
from datetime import datetime

cp = cryptocompare(date_from="2018-05-03 08:30:00 EST")
price_df = cp.hourly_price_historical(symbol='ETH', aggregate=1)
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

for current_dt in price_df.index.values:
    current_dt = pd.to_datetime(current_dt)
    current_news = cp.news('ETH', date_before=current_dt.strftime('%Y-%m-%d %H:%M:%S')  + ' EST')
    current_sentiment = [txb(news['title']).sentiment.polarity for news in current_news]

    sentiment_sum = np.sum(current_sentiment)
    sentiment_schema_1.append(sentiment_sum)

    utc_current_dt = convert_time_to_uct(current_dt)
    delta_ts = utc_current_dt.timestamp() - 5*3600
    news_count = np.sum([news['published_on'] > delta_ts for news in current_news])
    sentiment_schema_2.append(news_count)

    news_count_sentiment_sum = np.sum(current_sentiment[0:news_count])
    sentiment_schema_3.append(news_count+news_count_sentiment_sum)

    iterations_complete += 1
    print(str(int(100*iterations_complete/total_len)) + '% complete')


open_price_df = pd.DataFrame({'Price':roge_normalization(price_df.ETH_open.values)}, index=price_df.index)
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