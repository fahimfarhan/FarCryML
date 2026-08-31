from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    iris_dataset = load_iris(return_X_y=True, as_frame=True)

    x, y = iris_dataset

    print(type(iris_dataset))
    print(iris_dataset)

    print(iris_dataset[0])

    print(f"{x = }")
    print(f"{y = }")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
   #  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, train_size=0.5)

    model: GaussianNB = GaussianNB()
    # scores = cross_val_score(model, x_train, y_train, cv=5) # alternative to model.fit() # didn't work :/

    # print(f"Cross-Validation Scores: {scores}")
    # print(f"Mean Validation Score: {scores.mean()}")

    model = model.fit(X=x_train, y=y_train)

    y_pred = model.predict(X=x_test)


    # 1. Calculate general accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- Naive Bayes Model Performance ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    # 2. Detailed Breakdown (Precision, Recall, F1-Score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 3. Confusion Matrix Breakdown
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    pass

if __name__ == "__main__":
    main()
    pass