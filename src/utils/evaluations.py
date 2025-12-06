from typing import Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


def evaluate_ml_model(
    model: LinearSVC,
    X_test: list,
    y_test: list
) -> dict[str, Any]:
    """
    Evaluates the performance of the trained model on the test dataset.

    Args:
        model (LinearSVC): The trained Linear SVM classifier.
        X_test (list): The feature vectors for the test dataset.
        y_test (list): The true labels for the test dataset.

    Returns:
        dict: A dictionary containing accuracy, precision, recall,
            F1-score, and confusion matrix.
    """
    # Fazer predições
    y_pred = model.predict(X_test)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)

    # Precisão, Recall e F1-Score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Relatório completo
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def evaluate_ml_model_cross_validation(
    model: LinearSVC,
    X: list,
    y: list
):
    """
    Evaluates the model using cross-validation.

    Args:
        model (LinearSVC): The Linear SVM classifier.
        X (list): The feature vectors.
        y (list): The true labels.

    Returns:
        None
    """
    # Validação cruzada com 5 folds
    scores = cross_val_score(
        model,
        X,  # type: ignore
        y,
        cv=5,
        scoring='accuracy'
    )
    accuracy_mean = scores.mean()
    accuracy_std = scores.std()

    scores = cross_val_score(
        model,
        X,  # type: ignore
        y,
        cv=5,
        scoring='precision_weighted'
    )
    precision_mean = scores.mean()
    precision_std = scores.std()

    scores = cross_val_score(
        model,
        X,  # type: ignore
        y,
        cv=5,
        scoring='recall_weighted'
    )
    recall_mean = scores.mean()
    recall_std = scores.std()

    scores = cross_val_score(
        model,
        X,  # type: ignore
        y,
        cv=5,
        scoring='f1_weighted'
    )
    f1_mean = scores.mean()
    f1_std = scores.std()

    return {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std
    }
