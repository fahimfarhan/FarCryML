import numpy as np

class SimpleLayer:
    def __init__(
        self,
    ):
        self.w: np.ndarray = None
        self.b: float = 0.0
        self.z: np.ndarray = None
        self.a: np.ndarray = None
        self.x: np.ndarray = None

        self.dW: np.ndarray = None
        self.db: float = 0.0
        pass

    def relu(self, x): return np.maximum(0, x)

    def drelu(self, x):
        n = x.shape[0]
        z = np.zeros((n,1))

        for i in range(n):
            if x[i, 0] > 0.0:
                z[i, 0] = 1.0
            else:
                z[i, 0] = 0.0 
        return z

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        self.x = x
        Xnm = x
        n = Xnm.shape[0]
        m = Xnm.shape[1]

        if self.w is None:
            self.w = np.random.rand(m, 1)

        Wm1 = self.w

        Zn1 = Xnm @ Wm1 + self.b
        An1 = self.relu(Zn1)

        self.z = Zn1
        self.a = An1
        
        return An1

    def backward(
        self,
        y
    ):
        Yn1 = y
        An1 = self.a
        Zn1 = self.z
        Xnm = self.x

        dA = -2*(Yn1 - An1)
        dZ = dA * self.drelu(Zn1)
        dW = Xnm.T @ dZ
        db = np.sum(dZ)

        self.dW = dW
        self.db = db

        # Gradient to the previous layer
        dA_prev = dZ @ self.w.T

        return dA_prev

    def update(
        self,
        learningrate: float,
    ):
        self.w = self.w - learningrate * self.dW
        self.b = self.b - learningrate * self.db
        pass

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, y, prediction):
        dA = -2 * (y - prediction)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)


def testcase1():
    X = np.array([
        [1.0],
        [2.0]
    ])

    Y = np.array([
        [2.0],
        [4.0]
    ])

    prediction = None
    layer = SimpleLayer()
    for epoch in range(1000):
        prediction = layer.forward(X)
        # calculate loss
        layer.backward(Y)
        layer.update(0.01)

        loss = np.sum((Y - prediction) ** 2)

        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={loss}")

    print(f"{prediction = }")

    # print(f"{layer.forward(X) = }")

    # print(f"{layer.backward(Y) = }")
    print(f"{layer.dW = }")
    print(f"{layer.db = }")

    print(f"{layer.z = }")
    print(f"{layer.a = }")
    print(f"{layer.dW = }")
    print(f"{layer.db = }")

    pass

if __name__ == "__main__":
    testcase1()
    pass
