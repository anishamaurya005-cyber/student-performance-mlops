import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/student_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/model.pkl"


def train_model():

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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Load preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform data
    X_train_processed = preprocessor.transform(X_train)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("\nModel training completed successfully!")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()