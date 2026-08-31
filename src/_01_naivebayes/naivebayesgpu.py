import os

# Must be set before importing scipy / sklearn
os.environ["SCIPY_ARRAY_API"] = "1"

import torch

from sklearn import config_context
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    device = get_device()

    print(f"Using device: {device}")

    # -------------------------
    # Dataset
    # -------------------------

    iris_dataset = load_iris(return_X_y=True, as_frame=True)

    x, y = iris_dataset

    print(f"Original X type: {type(x)}")
    print(f"Original y type: {type(y)}")

    # Convert pandas -> PyTorch
    x = torch.tensor(
        x.values,
        dtype=torch.float32,
        device=device,
    )

    y = torch.tensor(
        y.values,
        dtype=torch.int64,
        device=device,
    )

    print(f"X: {x.shape}, {x.device}")
    print(f"y: {y.shape}, {y.device}")

    # -------------------------
    # Train / test split
    # -------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"X_train device: {x_train.device}")
    print(f"X_test device:  {x_test.device}")

    # -------------------------
    # Model
    # -------------------------

    model = GaussianNB()

    with config_context(array_api_dispatch=True):

        model.fit(
            X=x_train,
            y=y_train,
        )

        y_pred = model.predict(
            X=x_test,
        )

        # -------------------------
        # Evaluation
        # -------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

    print("\n--- Naive Bayes Model Performance ---")

    print(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(classification_report(y_test.cpu(), y_pred.cpu()))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test.cpu(), y_pred.cpu()))

    print(f"\ny_pred type: {type(y_pred)}")

    if isinstance(y_pred, torch.Tensor):
        print(f"y_pred device: {y_pred.device}")


if __name__ == "__main__":
    main()
    pass