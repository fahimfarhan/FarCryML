import logging

import numpy as np

# Configure the logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


class BasicLinearRegression:
  """
  @brief: recall that linear regression is defined as:
    y = wx + b,
    or Y_nx1 = X_nxm * W_mx1 + B_nx1 , [capital means matrices]

    so, Y is column matrix, x is matrix.
    We have n equations cz y has n samples. if there are m features, then 1 eqn has m xs. thus n eqns have nxm xs.

    data type? np.ndarray
  """
  W_mx1: np.ndarray = None
  B: float = None
  M: int = None  # features_count
  N: int = None  # target_count

  def __init__(self, x: np.ndarray = None, y: np.ndarray = None, features_count: int = None, target_count: int = None):
    # detect M, N
    if features_count is not None and target_count is not None:
      self.init_using_dimensions(features_count=features_count, target_count=target_count)
    elif x is not None and y is not None:
      self.init_using_data(x, y)
    else:
      raise Exception("something went wrong while initializing ")

    self.init_weights_and_biases()
    pass

  def init_using_data(self, x: np.ndarray, y: np.ndarray):
    logging.debug(f"f{x.shape=}")
    logging.debug(f"f{y.shape=}")

    self.M: int = x.shape[1]
    self.N: int = y.shape[0]
    pass

  def init_using_dimensions(self, features_count, target_count):
    self.M = features_count
    self.N = target_count
    pass

  def init_weights_and_biases(self):
    m: int = self.M
    n: int = self.N
    logging.debug(f"feature count {m=}")
    logging.debug(f"target count {n=}")
    # init weights
    self.W_mx1: np.ndarray = np.full(shape=(m, 1), fill_value=1)
    # init biases
    # self.B: np.ndarray = np.full(shape=(n, 1), fill_value=0)
    self.B = 0
    pass

  def fit(self, X_nxm, Y_nx1, lr: float = 0.05, epochs: int = 100):
    for _ in range(epochs):
      self.W_mx1 = GradientDescent(W_mx1=self.W_mx1, X_nxm=X_nxm, Y_nx1=Y_nx1, learning_rate=lr)
    pass

  def predict(self, X_nxm: np.ndarray):
    Y_pred_nx1 = np.matmul(X_nxm, self.W_mx1) + self.B
    return Y_pred_nx1

  """
     todo: 
      * calculate loss? MSE
      * optimization gradient descent
      * regularization Lasso L1, Ridge L2
      * function fit (x, y)
      * function predict (x, y)
  """

  """
    recall MSE(Y_actual, Y_predicted) = sum( (Yi_actual - Yi_predicted )**2 / M
    then W = W - dMse
  """


def MSELoss(Y_actual: np.ndarray, Y_predicted: np.ndarray) -> float:
  """
  :param Y_actual: np.ndarray,
  :param Y_predicted: np.ndarray,
  :return: MSE(y), a floating number
  recall MSE(Y_actual, Y_predicted) = sum( (Yi_actual - Yi_predicted )**2 / M
    then W = W - dMse

  """
  n = Y_actual.shape[0]  # for simplicity, let's assume Y_nx1 a column matrix. then shape[0] is the n

  totalDySquared = 0
  for i in range(0, n):
    ithDY: float = (float(Y_actual[i][0]) - float(Y_predicted[i][0]))
    ithDySquared = ithDY ** 2
    totalDySquared += ithDySquared

  meanDySquared = totalDySquared / n
  mseY = meanDySquared
  return mseY  # so the loss function returns a number?


def GradientDescent(W_mx1: np.ndarray, X_nxm: np.ndarray, Y_nx1: np.ndarray, learning_rate: float, bias: float) -> (np.ndarray, float):
  """
   the formula W:=W−α⋅(2/n)XT(XW−Y)
  :param bias:
  :param W_mx1:
  :param X_nxm:
  :param Y_nx1:
  :param learning_rate:
  :return:
  """
  n = Y_nx1.shape[0]
  # find prediction
  Y_pred_nx1 = np.matmul(X_nxm, W_mx1) + bias
  # find W_new
  # find delta = xw - y
  delta_nx1 = np.matmul(X_nxm, W_mx1) - Y_nx1
  # find gradient
  X_mxn = X_nxm.T
  gradient_mx1 = (2.0 / n) * np.matmul(X_mxn, delta_nx1)
  # find W_new
  W_mx1_new = W_mx1 - learning_rate * gradient_mx1

  # find B_new
  gradient_b = (2.0 / n) * np.sum(delta_nx1)  # Gradient for bias
  new_bias = bias - learning_rate * gradient_b
  return W_mx1_new, new_bias


def L1Regularization(mse: float, W_mx1: np.ndarray, someLambda: float):
  """
  L1 = mse + someLambda * sum ( |Wj| )
  """

  m = W_mx1.shape[0]
  total: float = 0.0

  for j in range(0, m):
    wj = W_mx1[j][0]
    total += np.abs(wj)

  l1 = mse + someLambda * total
  return l1


def L2Regularization(mse: float, W_mx1: np.ndarray, someLambda: float):
  """
  L2 = mse + someLambda * sum ( Wj**2 )
  """
  m = W_mx1.shape[0]
  total: float = 0.0

  for j in range(0, m):
    wj = W_mx1[j][0]
    total += wj ** 2

  l2 = mse + someLambda * total
  return l2


## regularization end

def start():
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
  bias = model.B
  w_3x1 = model.W_mx1
  logging.debug(f"{bias=}")
  logging.debug(f"{w_3x1=}")

  assert w_3x1.shape == (3, 1)
  assert np.all(model.W_mx1 == 1)  # Check if W is initialized to 1
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
  Wj, bj = GradientDescent(W_mx1=w_3x1, X_nxm=X_4x3, Y_nx1=Y_4x1, learning_rate=0.05, bias=bias)
  print(f"{Wj=}")
  print(f"{bj=}")

  l1 = L1Regularization(someMseLoss, w_3x1, someLambda=0.05)
  print(f"{l1=}")
  l2 = L2Regularization(someMseLoss, w_3x1, someLambda=0.05)
  print(f"{l2=}")
  pass


if __name__ == '__main__':
  start()

"""
mseLoss as array:
2025-03-06 23:46:55,377 - DEBUG - mseLoss=array([0.145])
2025-03-06 23:46:55,377 - DEBUG - someW_4x3=array([[0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275]])
       
       
 --------
 mse loss as a number:
 2025-03-06 23:47:49,377 - DEBUG - mseLoss=np.float64(0.14500000000000005)
2025-03-06 23:47:49,378 - DEBUG - someW_4x3=array([[0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275],
       [0.99275, 0.99275, 0.99275]])
"""
