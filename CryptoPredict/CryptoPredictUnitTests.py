import unittest
from CryptoPredict.CryptoPredict import CryptoCompare
from CryptoPredict.CryptoPredict import DataSet

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

class TestDataSet(unittest.TestCase):
    def setUp(self):
        date_from = "2018-05-15 09:00:00 EST"
        date_to = "2018-05-16 09:00:00 EST"
        bitinfo_list = ['eth']
        prediction_ticker = 'ETH'
        time_units = 'hours'

        self.data_obj = DataSet(date_from=date_from, date_to=date_to, days=6, bitinfo_list=bitinfo_list, prediction_ticker=prediction_ticker, time_units=time_units)
        self.data_obj.create_arrays()

    def test_create_price_prediction_columns_creates_correct_number_of_columns_and(self):
        self.assertEqual(len(self.data_obj.final_table.columns), 9)

    def test_create_price_prediction_columns_creates_correct_number_of_rows(self):
        self.assertEqual(len(self.data_obj.final_table.index), 25)

    def test_dataset_creates_correct_predictions(self):
        self.assertEqual(self.data_obj.fin_table['ETH_open'][7], self.data_obj.output_array[1])


if __name__ == '__main__':
    TestCryptoCompare.run()
    TestDataSet.run()