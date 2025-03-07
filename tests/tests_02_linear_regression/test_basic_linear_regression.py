import numpy as np
import pytest
import logging

from src.hands_on_02_linear_regression.BasicLinearRegression import BasicLinearRegression

# Configure the logging
logging.basicConfig(level=logging.NOTSET,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


def test_dimensionality_x():
  """
  x should be Mx1, a column matrix
  """
  x = np.array([
    [0],
    [1],
    [2]
  ])

  assert x.shape == (3, 1)
  pass


def test_dimensionality_y():
  """
  y should be Nx1, a column matrix
  """
  y = np.array([
    [0],
    [1],
    [2],
    [3]
  ])

  assert y.shape == (4, 1)
  pass


def test_initialization():
  """
  @brief: recall that linear regression is defined as:
  y = wx + b,
  or Y_nx1 = X_nxm * W_mx1 + B_nx1 , [capital means matrices]
  """
  X_4x3 = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
  ])

  Y_4x1 = np.array([
    [0],
    [1],
    [2],
    [3]
  ])

  model: BasicLinearRegression = BasicLinearRegression(x=X_4x3, y=Y_4x1)
  b_4x1 = model.B
  w_3x1 = model.W
  logging.debug(f"{b_4x1=}")
  logging.debug(f"{w_3x1=}")

  assert b_4x1.shape == (4, 1)
  assert w_3x1.shape == (3, 1)
  assert np.all(model.W == 1)  # Check if W is initialized to 1
  assert np.all(model.B == 0)  # Check if B is initialized to 0
  pass
