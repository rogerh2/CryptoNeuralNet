import unittest
import pickle
from ODESolvers.MachineLearningFourier import FourierNeuralNetworkGenerator

RAW_DATA = pickle.load(open("/Users/rjh2nd/PycharmProjects/CryptoNeuralNet/ODESolvers/SolversUnitTests/UnitTestData/psm_test.pickle", "rb"))

class MyTestCase(unittest.TestCase):
    def test_does_create_training_data_with_correct_number_of_rows_and_columns(self):
        generator = FourierNeuralNetworkGenerator(RAW_DATA[3][0:990], 60, 30, 3)
        training_data = generator.create_training_data('a', 1, step_size=10)
        self.assertEqual(training_data.shape, (90, 14))


if __name__ == '__main__':
    unittest.main()
