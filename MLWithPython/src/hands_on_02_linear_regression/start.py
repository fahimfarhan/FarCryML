from sklearn.model_selection import train_test_split

from src.hands_on_02_linear_regression.BasicLinearRegression import BasicLinearRegression, MSELoss, GradientDescent, \
  L1Regularization, L2Regularization

from sklearn.datasets import fetch_california_housing
import logging
import polars as pl

logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log message format


def loadDataFrame() -> pl.DataFrame:
  housing = fetch_california_housing()
  logging.debug(f"{type(housing)=}")  # bunch
  logging.debug(f"{type(housing.data)=}")  # np.ndarray :/

  # convert houisng into  hashmap
  mp = {}

  for i, name in enumerate(housing.feature_names):
    mp[name] = housing.data[:, i]

  mp["PRICE"] = housing.target

  data_frame: pl.DataFrame = pl.DataFrame(mp)
  return data_frame


if __name__ == '__main__':
  df = loadDataFrame()
  logging.debug(df.head())

  logging.debug(f"{df.shape=}")  # (20640, 9) so, X_20640x8, Y_2964x1, W_8x1, bias = bias

  X_20640x8 = df.drop("PRICE").to_numpy()
  Y_20640x1 = df["PRICE"].to_numpy()

  X = X_20640x8
  y = Y_20640x1
  # X_train_165120x8, X_test_4128x8, y_train_165120x1, y_test_4128x1 = train_test_split(X, y, test_size=0.2, random_state=42)

  N = X_20640x8.shape[0]  # target count
  M = X_20640x8.shape[1]  # feature count
  model = BasicLinearRegression(features_count=M, target_count=N)  # oops! my model doesn't handle train test etc

  model.fit(X_nxm=X_20640x8, Y_nx1=Y_20640x1)
  Y_predicted_20640x1 = model.predict(X_nxm=X_20640x8)
  logging.debug(f"{Y_predicted_20640x1}")
  pass

"""
def foo():
  df = pd.DataFrame(data.data, columns=data.feature_names)
  df['PRICE'] = data.target

  # 2️⃣ Create a dictionary (like a HashMap in Java)
  data_dict = {}

  # 3️⃣ Fill dictionary with feature columns
  for i, name in enumerate(data.feature_names):
    data_dict[name] = data.data[:, i]  # Assign each column

  # 4️⃣ Add target column (PRICE)
  data_dict["PRICE"] = data.target

  # 5️⃣ Convert to Polars DataFrame
  df = pl.DataFrame(data_dict)
  pass
"""
