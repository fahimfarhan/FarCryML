import numpy as np


def L1Regularization(mse: float, W_nxmList: list[np.ndarray], someLambda: float):
  """
  L1 = mse + someLambda * sum ( |Wj| )
  """
  w0 = W_nxmList[0]
  n = w0.shape[0]
  m = w0.shape[1]
  total = np.full(shape=(n, m), fill_value=0)

  for wj in W_nxmList:
    total += np.abs(wj)

  l1 = mse + someLambda * total
  return l1


def L2Regularization(mse: float, W_nxmList: list[np.ndarray], someLambda: float):
  """
  L2 = mse + someLambda * sum ( Wj**2 )
  """
  w0 = W_nxmList[0]
  n = w0.shape[0]
  m = w0.shape[1]
  total = np.full(shape=(n, m), fill_value=0)

  for wj in W_nxmList:
    total += wj * wj

  l2 = mse + someLambda * total
  return l2

