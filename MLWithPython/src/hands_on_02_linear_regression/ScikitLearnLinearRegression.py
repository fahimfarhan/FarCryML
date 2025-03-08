import numpy as np
import polars as pl
import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import logging
import matplotlib
from sklearn.linear_model import LinearRegression

from sklearn.utils import Bunch

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)  # Enable debug logs

  housing_dataset: Bunch = fetch_california_housing()
  logging.debug(f"{type(housing_dataset)=}")  # bunch
  logging.debug(f"{type(housing_dataset.data)=}")  # np.ndarray :/

  # prepare dataframe
  # convert input data into a hashmap
  hashmap = {}

  for jthColumn, feature_name in enumerate(housing_dataset.feature_names):
    hashmap[feature_name] = housing_dataset.data[:, jthColumn]  # just google np.ndarray get jth column

  hashmap["PRICE"] = housing_dataset.target
  dataframe: pl.DataFrame = pl.DataFrame(data=hashmap)

  # X, Y
  X_nxm = dataframe.drop("PRICE").to_numpy().astype(np.float64)
  Y = dataframe["PRICE"].to_numpy().astype(np.float64)

  logging.debug(f"{X_nxm.shape=}")  # (20640, 8)
  logging.debug(f"{Y.shape=}")  # (20640, )

  Y_nx1 = Y.reshape(-1, 1)
  logging.debug(f"{Y_nx1.shape=}")  # (20640, 1)

  trainX_nxm, testX_nxm, trainY_nx1, testY_nx1 = train_test_split(X_nxm, Y_nx1, test_size=0.2, random_state=42)

  model = LinearRegression()
  model.fit(X=trainX_nxm, y=trainY_nx1)

  # Make Predictions
  y_pred = model.predict(testX_nxm)

  # Evaluate Model
  r2 = r2_score(y_true=testY_nx1, y_pred=y_pred)
  print(f"R² Score: {r2:.4f}")  # R² score closer to 1 means better fit

  # visualization
  matplotlib.use("TkAgg")  # Set backend to Tkinter

  plt.scatter(x=testY_nx1, y=y_pred, alpha=0.2)
  plt.xlabel("Actual Price")
  plt.ylabel("Predicted Price")
  plt.title("Actual vs. Predicted Prices")
  plt.savefig("prediction_plot.png")
  plt.show()
  a = input("Press any key to exit: ")  # to prevent exiting immediately
  pass
