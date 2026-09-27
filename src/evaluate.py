import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


DATA_PATH = "data/student_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/model.pkl"


def evaluate_model():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Remove ID
    df = df.drop("student_id", axis=1)

    # Target
    target = "exam_score"

    X = df.drop(target, axis=1)
    y = df[target]

    # Same train-test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Load preprocessor and model
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test_processed)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")


if __name__ == "__main__":
    evaluate_model()