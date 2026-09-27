import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/student_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/model.pkl"


def train_with_mlflow():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Remove ID
    df = df.drop("student_id", axis=1)

    # Target
    target = "exam_score"

    X = df.drop(target, axis=1)
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Load preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Start MLflow experiment
    mlflow.set_experiment("Student_Performance_Prediction")

    with mlflow.start_run():

        # Model parameters
        n_estimators = 100
        random_state = 42

        # Create model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train
        model.fit(X_train_processed, y_train)

        # Predictions
        predictions = model.predict(X_test_processed)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        # Save model locally too
        joblib.dump(model, MODEL_PATH)

        print("\nMLflow run completed successfully!")
        print("-------------------------------")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.2f}")

        print("\nModel saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_with_mlflow()