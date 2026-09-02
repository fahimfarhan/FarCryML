from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics

def main():
    x, y = load_iris(return_X_y=True, as_frame=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    model = KMeans(n_clusters=3)
    trainedmodel = model.fit(train_x)
    labels = trainedmodel.labels_

    print(f"{ labels = }")
    print(f"{ train_y = }")
    
    pass

if __name__ == "__main__":
    main()
    pass
