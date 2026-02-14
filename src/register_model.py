import mlflow
from src.config import MODEL_NAME, EXPERIMENT_NAME

def register_best_model():
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(exp.experiment_id)

    best_run = max(runs, key=lambda r: r.data.metrics.get("f1_score", 0))

    model_uri = f"runs:/{best_run.info.run_id}/sentiment_model"

    result = mlflow.register_model(model_uri, MODEL_NAME)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Registered version:", result.version)


if __name__ == "__main__":
    register_best_model()
