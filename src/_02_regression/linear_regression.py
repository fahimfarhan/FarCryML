from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


def main():
    x, y = load_diabetes(return_X_y=True, as_frame=False)
    print(f"{x = }")
    print(f"{y = }")

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = LinearRegression()
    trained_model = model.fit(train_x, train_y)
    print(f"{trained_model = }")

    predicted_y = trained_model.predict(test_x)

    print(f"{predicted_y = }")

    mse = mean_squared_error(test_y, predicted_y)
    rmse = root_mean_squared_error(test_y, predicted_y)
    r2 = r2_score(test_y, predicted_y)

    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

    pass

if __name__ == "__main__":
    main()
    pass
