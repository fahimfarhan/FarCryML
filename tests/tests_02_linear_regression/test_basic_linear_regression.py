import numpy as np
import pytest
import logging

from src.hands_on_02_linear_regression.BasicLinearRegression import BasicLinearRegression

# Configure the logging
logging.basicConfig(level=logging.NOTSET,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


def test_initialization():
  Y_1_3 = np.array([
    [1, 2, 3]
  ])  # y = np.array([1, 2, 3]) error cz weird python syntax I guess
  X_3_2 = np.array([
    [1, 2],
    [3, 4],
    [5, 6]]
  )

  model: BasicLinearRegression = BasicLinearRegression(x=X_3_2, y=Y_1_3)
  b = model.B
  w = model.W
  logging.debug(f"{b = }")
  logging.debug(f"{w = }")

  assert b.shape == (1, 3)
  assert w.shape == (2, 3)
  assert np.all(model.W == 1)  # Check if W is initialized to 1
  assert np.all(model.B == 0)  # Check if B is initialized to 0
  pass
