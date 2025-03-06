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
  y = wx + b,
  or Y_nx1 = W_nxm * X_mx1 + B_nx1 , [capital means matrices]

  """
  X_3x1 = np.array([
    [0],
    [1],
    [2]
  ])

  Y_4x1 = np.array([
    [0],
    [1],
    [2],
    [3]
  ])

  model: BasicLinearRegression = BasicLinearRegression(x=X_3x1, y=Y_4x1)
  b_4x1 = model.B
  w_4x3 = model.W
  logging.debug(f"{b_4x1=}")
  logging.debug(f"{w_4x3=}")

  assert b_4x1.shape == (4, 1)
  assert w_4x3.shape == (4, 3)
  assert np.all(model.W == 1)  # Check if W is initialized to 1
  assert np.all(model.B == 0)  # Check if B is initialized to 0
  pass
