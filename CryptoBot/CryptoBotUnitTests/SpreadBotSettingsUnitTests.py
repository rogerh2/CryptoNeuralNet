import unittest
from CryptoBot.SpreadBot import LiveRunSettings


class LiveRunSettingsTestCase(unittest.TestCase):

    def test_saves_and_reads_data(self):
        settings = LiveRunSettings(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/unittest_bot_settings.txt')
        syms={'ATOM':1,
              'OXT':2,
              'LTC':3,
              'LINK':4,
              'ZRX':5}
        for sym in syms.keys():
            settings.write_setting_to_file(sym + ' buy price', syms[sym])
            settings.write_setting_to_file(sym + ' sell price', syms[sym])
            settings.write_setting_to_file(sym + ' spread', syms[sym])

            settings.update_settings()
        for sym in syms.keys():
            self.assertEqual(syms[sym], settings.read_setting_from_file(sym + ' buy price'), msg=sym)

    def tests_returns_none_if_entry_not_found(self):
        settings = LiveRunSettings(
            '/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/CryptoBot/CryptoBotUnitTests/UnitTestData/unittest_bot_settings.txt')
        settings.update_settings()
        settings.read_setting_from_file('Hello World! buy price')
        self.assertIsNone(settings.settings['Hello World! buy price'])


if __name__ == '__main__':
    unittest.main()
