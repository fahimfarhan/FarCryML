from sklearn.datasets import load_iris
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np

"""
The important thing to remember:

Self-training = model labels some unlabeled data itself, then learns from those pseudo-labels.
"""


X, y = load_iris(return_X_y=True)

# Pretend most labels are unavailable
rng = np.random.RandomState(42)
y_semi = y.copy()

mask = rng.rand(len(y)) < 0.7
y_semi[mask] = -1       # -1 means "unlabeled"

model = SelfTrainingClassifier(
    SVC(probability=True),
    threshold=0.8
)

model.fit(X, y_semi)

predictions = model.predict(X)

print(predictions)

