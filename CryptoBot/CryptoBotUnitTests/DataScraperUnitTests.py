import unittest
from CryptoBot.CryptoForecast import DataScraper
from CryptoBot.CryptoBot_Shared_Functions import convert_time_to_uct
from datetime import datetime
from datetime import timedelta


class DataScraperTestCase(unittest.TestCase):

    def setUp(self):
        self.fmt = '%Y-%m-%d %H:%M:%S'
        self.test_obj = DataScraper("2018-05-15 09:00:00", date_to="2018-05-16 09:00:00", exchange='Coinbase')

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

    def test_can_scrape_news_within_specified_limits(self):
        ref_date_from = "2018-11-30 07:01:00"
        ds = DataScraper(date_from=ref_date_from, date_to="2018-12-01 23:00:00", exchange='Coinbase')
        ds_date_from_ts = convert_time_to_uct(datetime.strptime(ref_date_from, self.fmt), tz_str='America/New_York').timestamp()
        a = ds.iteratively_scrape_news(['ETH', 'BTC'])

        self.assertGreater(a[0]['published_on'], ds_date_from_ts)


if __name__ == '__main__':
    unittest.main()
