from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "model" / "heart_model.joblib"
METADATA_PATH = ROOT / "model" / "metadata.json"
DATA_URL = "https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def download_dataset(path: Path, url: str = DATA_URL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "heart-disease-fastapi-assignment/1.0"})
    try:
        with urlopen(request, timeout=30) as response, path.open("wb") as destination:
            destination.write(response.read())
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Could not download the heart.csv dataset. Download it from "
            "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset "
            f"and place the CSV at {path}. Details: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--download-if-missing", action="store_true",
        help="Download a public mirror of the Kaggle dataset when the CSV is missing.",
    )
    parser.add_argument("--data-url", default=DATA_URL, help="Dataset CSV URL for automated builds.")
    args = parser.parse_args()

    if not args.data_path.is_file():
        if not args.download_if_missing:
            print(
                f"Dataset not found: {args.data_path}\n"
                "Download heart.csv from the assignment Kaggle page, or run with --download-if-missing.",
                file=sys.stderr,
            )
            return 2
        print(f"Downloading dataset from {args.data_url}")
        download_dataset(args.data_path, args.data_url)

    data = pd.read_csv(args.data_path)
    missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing:
        print(f"Dataset is missing required columns: {missing}", file=sys.stderr)
        return 2
    data = data[FEATURES + [TARGET]].dropna()
    labels = sorted(data[TARGET].unique().tolist())
    if labels != [0, 1]:
        print(f"Expected binary target values 0 and 1; found {labels}", file=sys.stderr)
        return 2
    if len(data) < 20:
        print(f"Dataset appears too small ({len(data)} rows); refusing to train.", file=sys.stderr)
        return 2

    X = data[FEATURES]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metrics = {
        "model_type": "StandardScaler + LogisticRegression",
        "features": FEATURES,
        "target": {"column": TARGET, "0": "absence", "1": "presence"},
        "dataset": "Kaggle johnsmith88/heart-disease-dataset (heart.csv)",
        "dataset_source": "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset",
        "training_data_mirror": args.data_url,
        "dataset_rows_used": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_probability)), 4),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_sklearn_version": sklearn.__version__,
        "note": "Metrics are for this educational dataset split; this is not a clinical model.",
    }
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Rows: {len(data)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Test accuracy: {metrics['test_metrics']['accuracy']}")
    print(f"Test ROC-AUC: {metrics['test_metrics']['roc_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
