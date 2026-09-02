from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load sample data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate using the Sklearn wrapper
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    eval_metric='logloss',
    random_state=42
)

# Fit and evaluate using native sklearn methods
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"mse: {mean_squared_error(y_test, preds):.4f}")