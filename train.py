from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_dataset():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()

    X.columns = [str(c).strip() for c in X.columns]

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target column contains unexpected values.")

    return X, y


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Exclude duration because it is only known after the call ends.
    if "duration" in X.columns:
        X = X.drop(columns=["duration"])

    # Normalize string columns.
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str).str.strip().str.lower()

    return X


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(
        include=["int16", "int32", "int64", "float16", "float32", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[("onehot", onehot)]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }


def evaluate_pipeline(name, pipeline, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    result = {
        "model": name,
        "cv_accuracy": float(np.mean(cv_results["test_accuracy"])),
        "cv_precision": float(np.mean(cv_results["test_precision"])),
        "cv_recall": float(np.mean(cv_results["test_recall"])),
        "cv_f1": float(np.mean(cv_results["test_f1"])),
        "cv_roc_auc": float(np.mean(cv_results["test_roc_auc"])),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }
    return result, pipeline


def get_feature_importance(trained_pipeline):
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])
    else:
        return None

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    return importance_df


def save_plot_confusion_matrix(cm, file_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    classes = ["No", "Yes"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plot_roc(fpr, tpr, auc_value, file_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plot_feature_importance(importance_df, file_path: Path):
    if importance_df is None or importance_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = importance_df.sort_values("importance", ascending=True)
    ax.barh(ordered["feature"], ordered["importance"])
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    X_raw, y = load_dataset()
    X = clean_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    models = build_models()

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        result, fitted_pipeline = evaluate_pipeline(
            model_name, pipeline, X_train, y_train, X_test, y_test
        )
        results.append(result)
        trained_pipelines[model_name] = fitted_pipeline
        print(f"Done: {model_name}")
        print(f"Test F1: {result['test_f1']:.4f}")

    comparison_df = (
        pd.DataFrame(results)
        .sort_values(by="test_f1", ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    best_model_name = comparison_df.loc[0, "model"]
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, ARTIFACT_DIR / "best_model.joblib")

    best_result = next(item for item in results if item["model"] == best_model_name)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    with open(ARTIFACT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": int(len(X)),
                "features_used": list(X.columns),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "positive_cases": int(y.sum()),
                "negative_cases": int((1 - y).sum()),
                "positive_rate": float(y.mean()),
                "best_model": best_model_name,
                "dropped_feature": "duration",
            },
            f,
            indent=2,
        )

    cm = np.array(best_result["confusion_matrix"])
    save_plot_confusion_matrix(cm, ARTIFACT_DIR / "confusion_matrix.png")
    save_plot_roc(
        best_result["fpr"],
        best_result["tpr"],
        best_result["test_roc_auc"],
        ARTIFACT_DIR / "roc_curve.png",
    )

    importance_df = get_feature_importance(best_pipeline)
    if importance_df is not None:
        importance_df.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
        save_plot_feature_importance(
            importance_df, ARTIFACT_DIR / "feature_importance.png"
        )

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(
        comparison_df[
            ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        ]
    )


if __name__ == "__main__":
    main()
