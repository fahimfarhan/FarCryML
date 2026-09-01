from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

def main():
    x, y = load_diabetes(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = SVR()
    trainedmodel = model.fit(train_x, train_y)

    predicted_y = trainedmodel.predict(test_x)
    mse = mean_squared_error(y_true=test_y, y_pred=predicted_y)
    rmse = root_mean_squared_error(y_true=test_y, y_pred=predicted_y)
    r2 = r2_score(y_true=test_y, y_pred=predicted_y)

    print(f"{ mse = }")
    print(f"{ rmse = }")
    print(f"{ r2 = }")
    pass

if __name__ == "__main__":
    main()
    pass
