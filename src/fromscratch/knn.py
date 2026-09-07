import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        n = X.shape[0]
        y = np.zeros((n, 1))

        for i in range(0, n):
            xi = X[i]
            di = self._predict_one(xi)
            indices_i = np.argsort(di)
            subindices_i = indices_i[0:self.k]
            labels_i = self.y[subindices_i]

            prediction_i = (np.mean(labels_i) >= 0.5).astype(int)
            y[i] = prediction_i

        return y

    def _predict_one(self, xi):
        X = self.X
        
        n = X.shape[0]
        di = np.zeros(n)
        for j in range(0, n):
            distij = np.linalg.norm(xi - X[j])
            di[j] = distij
        return di

def testcase1():
    print("test case 1")
    X_train = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ], dtype=float)

    y_train = np.array([
        [0],
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    model = KNN(k=3)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1.5],
        [2.5],
        [4.5],
        [5.5]
    ])

    print(model.predict(X_test))
    pass

def testcase2(k = 3):
    print("test case 2")
    X_train = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],

        [4, 4],
        [4, 5],
        [5, 4],
        [5, 5]
    ], dtype=float)

    y_train = np.array([
        [0],
        [0],
        [0],
        [0],

        [1],
        [1],
        [1],
        [1]
    ])

    model = KNN(k=k)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1.5, 1.5],
        [2, 2],
        [4.5, 4.5],
        [5, 5]
    ])

    print(model.predict(X_test))
    pass

if __name__ == "__main__":
    testcase1()
    testcase2(1)
    testcase2(2)
    testcase2(3)
    testcase2(4)
    testcase2(5)
    pass
