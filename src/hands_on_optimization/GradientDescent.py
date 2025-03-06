import numpy as np


def GradientDescent(W_nxm: np.ndarray, learning_rate: float, mse: float) -> np.ndarray:
  W = W_nxm - learning_rate * mse
  return W
