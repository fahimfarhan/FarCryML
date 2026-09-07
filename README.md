# FarCryML

The bare minimum implement from scratch

| # | Algorithm               | Implement with | What it teaches                                          |
| - | ----------------------- | -------------- | -------------------------------------------------------- |
| 1 | **Linear Regression**   | NumPy          | Loss, optimization, matrix math                          |
| 2 | **Logistic Regression** | NumPy          | Classification, sigmoid, cross-entropy, gradient descent |
| 3 | **K-Nearest Neighbors** | NumPy          | Distance, decision boundaries, lazy learning             |
| 4 | **Naive Bayes**         | NumPy          | Probability, Bayes theorem, distributions                |
| 5 | **Decision Tree**       | NumPy          | Entropy/Gini, information gain, recursive splitting      |
| 6 | **Neural Network**      | NumPy          | Forward pass, backpropagation, gradients                 |


* What kind of problem?
- classification
- regression
- clustering
- dimensionality reduction
- anomaly detection
* What sklearn class do I use?
* What does the basic code look like?
* What are 2–4 important parameters?
* What metric do I use to see whether it worked?

```python
class Model:
    def __init__(self):
        pass

    """
    Assuming X has dimensions (n,m), and Y has dimensions (n, 1) a column vector for convenience.
    """
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        predicted_y: np.ndarray = None
        return predicted_y
```
