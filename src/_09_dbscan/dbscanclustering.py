from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN


def main():
    X, y = load_iris(return_X_y=True)

    model = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    model.fit(X)

    labels = model.labels_

    print("Cluster labels:", labels)
    pass

if __name__ == "__main__":
    main()
    pass
