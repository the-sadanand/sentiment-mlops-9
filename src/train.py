import mlflow
import mlflow.sklearn
import joblib
import tempfile
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data import load_data
from src.config import MODEL_NAME, EXPERIMENT_NAME

np.random.seed(42)
random.seed(42)


def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, _, y_train, y_val, _ = load_data()

    with mlflow.start_run(run_name="logreg-tfidf"):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_val_vec)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        mlflow.log_params({
            "max_features": 5000,
            "ngram_range": (1, 2),
            "model": "LogisticRegression"
        })

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(model, "sentiment_model")

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.pkl"
            joblib.dump(vectorizer, path)
            mlflow.log_artifact(path, "tfidf_vectorizer")

        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    train()
