import unittest
import logging
import numpy as np
from src.hands_on_02_linear_regression.BasicLinearRegression import BasicLinearRegression

# Configure the logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


class MyTestCase(unittest.TestCase):
  """
   Just trying out unittest.
  """
  def test_initialization(self):
    y = np.array([1, 2, 3])
    x = np.array([[1, 2], [3, 4], [5, 6]])

    model: BasicLinearRegression = BasicLinearRegression(x=x, y=y)
    b = model.B
    w = model.W
    logging.debug(f"{b = }")
    logging.debug(f"{w = }")
    self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
  unittest.main()
