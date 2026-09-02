from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

def start():
    # breakdown steps
    print("PCA breakdown steps")
    x, y = load_iris(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    pca = PCA(n_components=2)

    pca_x = pca.fit_transform(scaled_x)

    train_x, test_x, train_y, test_y = train_test_split(pca_x, y, train_size=0.8)

    model = KNeighborsClassifier()

    trainedmodel = model.fit(train_x, train_y)
    predicted_y = trainedmodel.predict(test_x)


    classification_report = metrics.classification_report(y_true=test_y, y_pred=predicted_y)
    print(classification_report)

    pass

def main():
    # combine all steps
    print("PCA using pipeline")
    x, y = load_iris(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        KNeighborsClassifier(),
    )

    trainedmodel = model.fit(train_x, train_y)
    predicted_y = trainedmodel.predict(test_x)

    classification_report = metrics.classification_report(y_true=test_y, y_pred=predicted_y)
    print(classification_report)
    pass

if __name__ == "__main__":
    start()
    main()
    pass
