import numpy as np
import pytest
import logging

from src.hands_on_02_linear_regression.BasicLinearRegression import BasicLinearRegression, MSELoss, GradientDescent, \
  L1Regularization, L2Regularization

# Configure the logging
# not printing in pytest :/ use print() for now
logging.basicConfig(level=logging.NOTSET,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


def test_dimensionality_x():
  """
  x should be NxM, a data matrix
  """
  X_4x3 = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
  ])

  assert X_4x3.shape == (4, 3)
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
    [0.0, 1.0, 2.0],
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
  ])

  Y_4x1 = np.array([
    [0.1],
    [1.1],
    [2.2],
    [3.5]
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

  #  test mseLoss, L1, L2, and optim
  #  let's create random Y, and pretend it to be Y_predicted
  Y_4x1_predicted = np.array([
    [0.5],
    [1.1],
    [2.4],
    [3.1]
  ])

  print("\n----------\n")
  someMseLoss = MSELoss(Y_actual=Y_4x1, Y_predicted=Y_4x1_predicted)
  print(f"{someMseLoss=}")
  print(f"{w_3x1=}")
  Wj = GradientDescent(W_mx1=w_3x1, learning_rate=0.05, mse=someMseLoss)
  print(f"{Wj=}")

  l1 = L1Regularization(someMseLoss, w_3x1, someLambda=0.05)
  print(f"{l1=}")
  l2 = L2Regularization(someMseLoss, w_3x1, someLambda=0.05)
  print(f"{l2=}")
  pass
