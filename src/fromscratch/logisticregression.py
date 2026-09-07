import numpy as np

class LogisticRegression:
    def __init__(self, alpha = 0.01, iteration = 10_000):
        self.w: np.ndarray = None
        self.b: float = 0.0
        self.alpha = alpha
        self.iteration = iteration
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        Xnm = x
        Yn1 = y
        n = Xnm.shape[0]
        m = Xnm.shape[1]

        if n != Yn1.shape[0]:
            raise Exception("Invalid matrix sizes")

        Wm1: np.ndarray = np.random.rand(m, 1)
        b = 0.0

        for i in range(0, self.iteration):
            z = Xnm @ Wm1 + b
            yp = self.sigmoid(z)

            e = yp - y
            dJdw = Xnm.T @ e
            dJdb = np.sum(e)

            Wm1 = Wm1 - self.alpha * dJdw
            b = b - self.alpha * dJdb

        self.w = Wm1
        self.b = b
        pass

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        probability = self.sigmoid(z)
        return probability

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability = self.predict_probability(x)
        return (probability >= 0.5).astype(int)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


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

    model = LogisticRegression(alpha=0.01, iteration=10_000)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1.5],
        [2.5],
        [4.5],
        [5.5]
    ])

    print(model.predict(X_test))

    print(f"{model.w = }")
    print(f"{model.b = }")

    pass

def testcase2():
    print("test case 2")
    X_train = np.array([
        [-3],
        [-2],
        [-1],
        [ 1],
        [ 2],
        [ 3]
    ], dtype=float)

    y_train = np.array([
        [0],
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    model = LogisticRegression(alpha=0.01, iteration=10_000)
    model.fit(X_train, y_train)

    X_test = np.array([
        [-2.5],
        [-0.5],
        [ 0.5],
        [ 2.5]
    ])

    print(model.predict(X_test))
    pass

def testcase3():
    print("test case 3")
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

    model = LogisticRegression(alpha=0.01, iteration=10_000)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1, 3],
        [2, 2],
        [4, 3],
        [5, 5]
    ], dtype=float)

    print(model.predict(X_test))

    prob = model.predict_probability(X_test)

    assert np.all(prob >= 0)
    assert np.all(prob <= 1)

    print("Probability test passed")

    assert model.w.shape == (2, 1)
    assert model.b is not None

    print("Shape test passed")
    pass

if __name__ == "__main__":
    testcase1()
    testcase2()
    testcase3()
    pass
