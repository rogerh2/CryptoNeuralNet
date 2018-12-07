import unittest
from CryptoBot.CryptoForecast import DataScraper
from CryptoBot.CryptoBot_Shared_Functions import convert_time_to_uct
from datetime import datetime
from datetime import timedelta
import pandas as pd


class DataScraperTestCase(unittest.TestCase):

    def setUp(self):
        self.fmt = '%Y-%m-%d %H:%M:%S'
        self.test_obj = DataScraper("2018-05-15 09:00:00", date_to="2018-05-16 09:00:00", exchange='Coinbase')

    # --Tests for the create_data_frame Method (which actually reads price data from the url and outputs it as a Pandas DataFrame)--

    def test_can_create_proper_data_headers(self):
        url = 'https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=ETH&limit=30&aggregate=1&e=CCCAGG'
        symbol = 'ETH'

        df = self.test_obj.create_data_frame(url, symbol)

        for att in ['high', 'low', 'open', 'close', 'volumeto', 'volumefrom']:
                self.assertIn(symbol + '_' + att, df.columns)

    def test_does_not_output_data_containing_nan(self):
        url = 'https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=ETH&limit=30&aggregate=1&e=CCCAGG'
        symbol = 'ETH'

        df = self.test_obj.create_data_frame(url, symbol)

        for data_col in df.columns:
            data = df[data_col].values
            self.assertFalse(any(pd.isnull(data)))

    # --Tests for the Price Data Output Method--

    def test_can_scrape_hourly_coinbase_prices(self):
        ref_ts = convert_time_to_uct(datetime.strptime("2018-05-16 09:00:00", self.fmt)).timestamp()
        df = self.test_obj.get_historical_price(symbol='ETH', unit='hr')

        # Ensure the correct number of checks are returned
        self.assertEqual(len(df.index), 25)

        # Ensure there are no duplicate dates
        self.assertEqual(len(df.date[df.date.duplicated(keep=False)]), 0)

        # Ensures that the scraper returs points up to the correct date
        self.assertEqual(ref_ts, convert_time_to_uct(df.date[24]).timestamp())

    def test_can_scrape_minutely_coinbase_prices(self):
        date_to = datetime.now().strftime('%Y-%m-%d %H:%M:') + '00'
        date_from = (datetime.now() - timedelta(minutes=2999)).strftime('%Y-%m-%d %H:%M:') + '00'
        ref_ts = convert_time_to_uct(datetime.strptime(date_to, self.fmt)).timestamp()
        ds = DataScraper(date_from=date_from, date_to=date_to, exchange='Coinbase')
        df = ds.get_historical_price(symbol='ETH', unit='min')

        # Ensure the correct number of checks are returned, the asserted number must be off by 2 presumably because 1 extra trailing minute is added after every 2000
        self.assertEqual(len(df.index), 3001)

        # Ensure there are no duplicate dates
        self.assertEqual(len(df.date[df.date.duplicated(keep=False)]), 0)

        # Ensures that the scraper returs points up to the correct date
        self.assertEqual(ref_ts, convert_time_to_uct(df.date[3000]).timestamp())

    # --Tests for the News Data Output Method--

    def test_can_scrape_news_within_specified_limits(self):
        ref_date_from = "2018-11-30 07:01:00"
        ds = DataScraper(date_from=ref_date_from, date_to="2018-12-01 23:00:00", exchange='Coinbase')
        ds_date_from_ts = convert_time_to_uct(datetime.strptime(ref_date_from, self.fmt), tz_str='America/New_York').timestamp()
        a = ds.iteratively_scrape_news(['ETH', 'BTC'])

        self.assertGreater(a[0]['published_on'], ds_date_from_ts)


if __name__ == '__main__':
    unittest.main()
