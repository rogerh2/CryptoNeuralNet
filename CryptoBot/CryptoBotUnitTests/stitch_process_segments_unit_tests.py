import unittest
import numpy as np
from CryptoBot.BackTest import stitch_process_segments


PORTFOLIO_HISTORY = (np.arange(0, 10), np.arange(10, 20), np.arange(20, 30))
SYM_START_PORTFOLIO_HISTORY = PORTFOLIO_HISTORY
PREDICTIONS = PORTFOLIO_HISTORY


class MyTestCase(unittest.TestCase):
    test_list = {0:
        {'USD': PORTFOLIO_HISTORY[0], 'process id': 0, 'SYM': SYM_START_PORTFOLIO_HISTORY[0], 'end state': None,
         'seg id': 0, 'predictions': PREDICTIONS[0]},
                 1:
        {'USD': PORTFOLIO_HISTORY[1], 'process id': 0, 'SYM': SYM_START_PORTFOLIO_HISTORY[1], 'end state': None,
         'seg id': 0, 'predictions': PREDICTIONS[1]},
                 2:
        {'USD': PORTFOLIO_HISTORY[2], 'process id': 0, 'SYM': SYM_START_PORTFOLIO_HISTORY[2], 'end state': True,
         'seg id': 0, 'predictions': PREDICTIONS[2]}
    }

    def test_can_output_correctly_formatted_dictionary(self):
        stitched_arr = np.array([])
        for arr in PORTFOLIO_HISTORY:
            stitched_arr = np.append(stitched_arr, arr)

        stitched_dict = stitch_process_segments(self.test_list)

        for key in ['USD', 'SYM', 'predictions']:
            np.testing.assert_array_equal(stitched_dict[key], stitched_arr)



if __name__ == '__main__':
    unittest.main()
