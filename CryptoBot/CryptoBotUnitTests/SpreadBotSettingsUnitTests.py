import unittest
from CryptoBot.SpreadBot import LiveRunSettings


class LiveRunSettingsTestCase(unittest.TestCase):
    def setUp(self):
        self.settings = LiveRunSettings('/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/unittest_bot_settings.txt')

    def test_saves_and_reads_data(self):
        syms={'ATOM':1,
              'OXT':2,
              'LTC':3,
              'LINK':4,
              'ZRX':5}
        for sym in syms.keys():
            self.settings.write_setting_to_file(sym + ' buy price', syms[sym])
            self.settings.write_setting_to_file(sym + ' sell price', syms[sym])
            self.settings.write_setting_to_file(sym + ' spread', syms[sym])

            self.settings.update_settings()
        for sym in syms.keys():
            self.assertEqual(syms[sym], self.settings.read_setting_from_file(sym + ' buy price'), msg=sym)

    def tests_returns_none_if_entry_not_found(self):
        self.assertIsNone(self.settings.read_setting_from_file('Hello World! buy price'))


if __name__ == '__main__':
    unittest.main()
