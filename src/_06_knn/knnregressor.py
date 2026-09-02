from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    x, y = load_diabetes(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = KNeighborsRegressor()
    trainedmodel = model.fit(train_x, train_y)
    predicted_y = trainedmodel.predict(test_x)

    mse = metrics.mean_squared_error(y_true=test_y, y_pred=predicted_y)
    rmse = metrics.root_mean_squared_error(y_true=test_y, y_pred=predicted_y)
    r2 = metrics.r2_score(y_true=test_y, y_pred=predicted_y)

    print(f"{ mse= }")
    print(f"{ rmse= }")
    print(f"{ r2= }")

    pass

if __name__ == "__main__":
    main()
    pass