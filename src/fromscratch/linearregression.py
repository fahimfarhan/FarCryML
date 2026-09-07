import numpy as np

class LinearRegression:
    
    def __init__(self, alpha=0.001, iterations=10_000):
        self.w = None
        self.b = 0.0
        self.alpha = alpha
        self.iterations = iterations
        pass

    """
    Assuming X has dimensions (n,m), and Y has dimensions (n, 1) a column vector for convenience.
    """
    def fit(self, x: np.ndarray, y: np.ndarray):
        Xnm = x
        Yn1 = y

        n = Xnm.shape[0]
        m = Xnm.shape[1]

        if n != y.shape[0]:
            raise Exception("Invalid matrix shapes. Xnm and Yn1 both should have n rows.")
        Wm1 = np.ones((m, 1))
        b: float = 0.0
        for i in range(0, self.iterations):
            Yp = Xnm @ Wm1 + b
            e = Yn1 - Yp

            # calculate dJdw
            dJdw = -2 * (Xnm.T) @ e

            # calculate dJdb
            # dJdb = 0.0
            # for j in range(0, n):
            #     dJdb += e[j][0]
            # dJdb = -2 * dJdb

            dJdb = -2 * np.sum(e) # simple and faster built-in method

            Wm1 = Wm1 - self.alpha * dJdw
            b = b - self.alpha * dJdb

        self.w = Wm1
        self.b = b
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        predicted_y: np.ndarray = x @ self.w + self.b
        return predicted_y


def testcase1():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ], dtype=float)

    y = np.array([
        [3],
        [5],
        [7],
        [9],
        [11]
    ], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    predicted_y = model.predict(X)

    print("Testcase 1")
    print("Expected:\n", y)
    print("Predicted:\n", predicted_y)
    print()
    pass

def testcase2():
    X = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [2, 3]
    ], dtype=float)

    y = np.array([
        [10],
        [12],
        [13],
        [17],
        [18]
    ], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    predicted_y = model.predict(X)

    print("Testcase 2")
    print("Expected:\n", y)
    print("Predicted:\n", predicted_y)
    print()
    pass

def testcase3():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ], dtype=float)

    y = np.array([
        [9],
        [7],
        [5],
        [3],
        [1]
    ], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    predicted_y = model.predict(X)

    print("Testcase 3")
    print("Expected:\n", y)
    print("Predicted:\n", predicted_y)
    print()
    pass

def testcase4():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
        [4]
    ], dtype=float)

    y = np.array([
        [7],
        [10],
        [13],
        [16],
        [19]
    ], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    predicted_y = model.predict(X)

    print("Testcase 4")
    print("Expected:\n", y)
    print("Predicted:\n", predicted_y)
    print()
    pass

def testcase5():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ], dtype=float)

    y = np.array([
        [4.2],
        [6.1],
        [8.5],
        [9.8],
        [12.3],
        [13.7]
    ], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    predicted_y = model.predict(X)

    print("Testcase 5")
    print("Actual:\n", y)
    print("Predicted:\n", predicted_y)
    print()
    pass

def testcase6():
    # Training data
    X_train = np.array([
        [1],
        [2],
        [3],
        [4],
    ], dtype=float)

    y_train = np.array([
        [3],
        [5],
        [7],
        [9],
    ], dtype=float)

    # Unseen test data
    X_test = np.array([
        [5],
        [6],
        [7],
    ], dtype=float)

    y_test = np.array([
        [11],
        [13],
        [15],
    ], dtype=float)

    model = LinearRegression()

    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    print("Testcase 6")
    print("Expected:")
    print(y_test)

    print("Predicted:")
    print(predicted_y)
    print()

    assert np.allclose(predicted_y, y_test, atol=1e-5)
    print("Testcase 6 passed!")
    pass

if __name__ == "__main__":
    testcase1()
    testcase2()
    testcase3()
    testcase4()
    testcase5()
    testcase6()
    pass
    