import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

# Two "views" of the data
X_a = X[:, :2]
X_b = X[:, 2:]

# Hide most labels
rng = np.random.RandomState(42)
y_semi = y.copy()

mask = rng.rand(len(y)) < 0.6
y_semi[mask] = -1

# Start with labeled data
labeled = y_semi != -1

model_a = LogisticRegression(max_iter=1000)
model_b = LogisticRegression(max_iter=1000)

model_a.fit(X_a[labeled], y[labeled])
model_b.fit(X_b[labeled], y[labeled])

# Each model predicts unlabeled examples
unlabeled = ~labeled

pred_a = model_a.predict(X_a[unlabeled])
pred_b = model_b.predict(X_b[unlabeled])

# Simple agreement
agreement = pred_a == pred_b

print("Agreed predictions:", pred_a[agreement])