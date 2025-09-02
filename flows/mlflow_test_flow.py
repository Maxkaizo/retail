from prefect import flow, task
import mlflow
import mlflow.sklearn
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Tarea de entrenamiento
@task
def train_and_log():
    mlflow.set_tracking_uri("http://localhost:5000")  # usa nombre de servicio en docker
    mlflow.set_experiment("prefect_test_experiment")

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    with mlflow.start_run(run_name="rf-prefect-test"):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("rmse", rmse)

        # Save & log model as artifact
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/rf_model"
        mlflow.sklearn.save_model(model, model_path)
        mlflow.log_artifacts(model_path, artifact_path="model")

    return rmse

# Flow orquestado
@flow(name="mlflow_test_flow")
def main_flow():
    rmse = train_and_log()
    print(f"✅ Flow finished. RMSE={rmse}")

if __name__ == "__main__":
    main_flow()
