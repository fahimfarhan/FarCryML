import numpy as np


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
    ithDY = (Y_actual[i] - Y_predicted[i])
    ithDySquared = ithDY ** 2
    totalDySquared += ithDySquared

  meanDySquared = totalDySquared / n
  mseY = meanDySquared
  return mseY  # so the loss function returns a number?
