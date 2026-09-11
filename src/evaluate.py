"""Classification evaluation with metrics requested in the project abstract."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_models(models, X_train, X_test, y_train, y_test, verbose=False):
    minimum_class_count = int(y_train.value_counts().min())
    folds = min(5, minimum_class_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="weighted", zero_division=0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = {
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "weighted_precision": float(precision), "weighted_recall": float(recall),
            "weighted_f1": float(f1), "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        }
        if verbose:
            print(f"\n{name} classification report:")
            print(classification_report(y_test, predicted, zero_division=0))
            print(f"{folds}-fold cross-validation accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    return results
