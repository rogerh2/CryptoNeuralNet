import unittest
from CryptoPredict.CryptoPredict import CryptoCompare


class TestCryptoCompare(unittest.TestCase):
    def setUp(self):
        self.cp = CryptoCompare(date_from="2018-05-15 09:00:00 EST", date_to="2018-05-16 09:00:00 EST")
        self.minute_cp = CryptoCompare(date_from="2018-05-16 08:00:00 EST", date_to="2018-05-16 09:00:00 EST")

    def test_datedelta(self):
        #Difference in minutes
        del_date = self.cp.datedelta('minutes')
        self.assertEqual(del_date, 1440.0)

        #Difference in hours
        del_date = self.cp.datedelta('hours')
        self.assertEqual(del_date, 24.0)

        #Difference in days
        del_date = self.cp.datedelta('days')
        self.assertEqual(del_date, 1.0)

    def test_daily_price_historical_returns_correct_number_of_rows(self):
        df = self.cp.daily_price_historical(symbol='ETH')
        self.assertEqual(len(df.index), 2) #2 instead of 1 because both start and finish hours are returned

    def test_hourly_price_historical_returns_correct_number_of_rows(self):
        df = self.cp.hourly_price_historical(symbol='ETH')
        self.assertEqual(len(df.index), 25) #see test_daily_price_historical_returns_correct_number_of_columns

    def test_minute_price_historical_returns_correct_number_of_rows(self):
        df = self.minute_cp.minute_price_historical(symbol='ETH')
        self.assertEqual(len(df.index), 61) #see test_daily_price_historical_returns_correct_number_of_columns

    def test_news(self):
        news_data = self.cp.news('ETH', date_before="2018-05-16 08:00:00 EST")
        self.assertEqual(len(news_data), 50)

if __name__ == '__main__':
    unittest.main()