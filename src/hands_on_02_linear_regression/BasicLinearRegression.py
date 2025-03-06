import numpy as np
import logging

# Configure the logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


class BasicLinearRegression:
  """
  @brief: recall that linear regression is defined as:
    y = wx + b,
    or Y_nx1 = W_nxm * X_mx1 + B_nx1 , [capital means matrices]

    so, Y is column matrix, x is column matrix.
    data type? np.ndarray
  """
  W: np.ndarray = None
  B: np.ndarray = None
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

    self.M: int = x.shape[0]
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
    self.W: np.ndarray = np.full(shape=(n, m), fill_value=1)
    # init biases
    self.B: np.ndarray = np.full(shape=(n, 1), fill_value=0)
    pass


def start():
  y = np.array([[1, 2, 3]])  # y = np.array([1, 2, 3])  # error cz weird python syntax I guess
  x = np.array([[1, 2], [3, 4], [5, 6]])

  model: BasicLinearRegression = BasicLinearRegression(x=x, y=y)
  b = model.B
  w = model.W
  logging.debug(f"{b=}")
  logging.debug(f"{w=}")
  pass


if __name__ == '__main__':
  start()
