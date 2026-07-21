"""Train and evaluate a leakage-aware property-age classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/byuidatascience/data4dwellings/"
    "master/data-raw/dwellings_ml/dwellings_ml.csv"
)
TARGET = "before1980"
EXCLUDED_FEATURES = {TARGET, "yrbuilt", "parcel"}
RANDOM_STATE = 42


def load_dataset(source: str | Path = DATA_URL) -> pd.DataFrame:
    """Load the upstream CSV from a URL or a local path."""
    return pd.read_csv(source)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the schema and return a target-leakage-free feature matrix."""
    missing = EXCLUDED_FEATURES - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features = data.drop(columns=sorted(EXCLUDED_FEATURES))
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Model features must be numeric: {non_numeric}")
    if features.isna().any().any():
        raise ValueError("Model features contain missing values")

    target = data[TARGET].astype(int)
    if set(target.unique()) - {0, 1}:
        raise ValueError("Target must contain only 0 and 1")
    return features, target


def build_model() -> RandomForestClassifier:
    """Create the deterministic, class-balanced portfolio model."""
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate_predictions(
    y_true: pd.Series, predictions: pd.Series, probabilities: pd.Series
) -> dict[str, float | list[list[int]]]:
    """Return complementary classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, predictions)), 4
        ),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def run_analysis(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Train on 80% of the data, evaluate on 20%, and save report figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features, target = prepare_features(data)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, 1]
    metrics = evaluate_predictions(test_y, predictions, probabilities)

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        test_y,
        predictions,
        display_labels=["1980 or later", "Before 1980"],
        ax=axis,
        colorbar=False,
        cmap="Purples",
    )
    axis.set_title("Property-age classification on the holdout set")
    figure.tight_layout()
    figure.savefig(output_path / "confusion-matrix.png", dpi=180)
    plt.close(figure)

    importances = (
        pd.Series(model.feature_importances_, index=features.columns)
        .sort_values(ascending=False)
        .head(12)
        .sort_values()
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    importances.plot.barh(ax=axis, color="#6d5ce8")
    axis.set_title("Top 12 random-forest feature importances")
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output_path / "feature-importance.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    data.loc[data[TARGET] == 1, "yrbuilt"].plot.hist(
        bins=35, alpha=0.72, ax=axis, label="Before 1980", color="#6d5ce8"
    )
    data.loc[data[TARGET] == 0, "yrbuilt"].plot.hist(
        bins=35, alpha=0.72, ax=axis, label="1980 or later", color="#ffb84d"
    )
    axis.axvline(1980, color="#222", linestyle="--", linewidth=1.5)
    axis.set_title("Build-year distribution and target boundary")
    axis.set_xlabel("Year built")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path / "build-year-distribution.png", dpi=180)
    plt.close(figure)

    result: dict[str, object] = {
        "rows": len(data),
        "features": features.shape[1],
        "holdout_rows": len(test_y),
        "random_state": RANDOM_STATE,
        "excluded_features": sorted(EXCLUDED_FEATURES),
        "metrics": metrics,
        "top_features": {
            name: round(float(value), 6)
            for name, value in importances.sort_values(ascending=False).items()
        },
    }
    (output_path / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=DATA_URL, help="Upstream CSV URL or local CSV path"
    )
    parser.add_argument(
        "--output", default="reports/figures", help="Directory for metrics and PNGs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(load_dataset(args.data), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
