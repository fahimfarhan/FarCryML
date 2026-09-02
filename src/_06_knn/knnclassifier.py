from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics

def main():
    x, y = load_iris(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = KNeighborsClassifier()
    trainedmodel = model.fit(train_x, train_y)
    predicted_y = trainedmodel.predict(test_x)

    accuracy = metrics.accuracy_score(test_y, predicted_y)
    classification_report = metrics.classification_report(test_y, predicted_y)

    print(classification_report)

    
    pass

if __name__ == "__main__":
    main()
    pass
