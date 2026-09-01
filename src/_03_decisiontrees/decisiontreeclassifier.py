from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    x, y = load_iris(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = DecisionTreeClassifier(max_depth=4)
    trainedmodel = model.fit(X=train_x, y=train_y)

    predicted_y = trainedmodel.predict(test_x)

    accuracy = accuracy_score(y_true=test_y, y_pred=predicted_y)
    report = classification_report(y_true=test_y, y_pred=predicted_y)
    print(f"{accuracy = }")
    print(report)
    pass

if __name__ == "__main__":
    main()
    pass
