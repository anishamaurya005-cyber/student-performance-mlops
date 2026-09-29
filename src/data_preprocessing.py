import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# File paths
# -----------------------------

DATA_PATH = "data/student_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


# -----------------------------
# Load dataset
# -----------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# -----------------------------
# Preprocess dataset
# -----------------------------

def preprocess_data():

    # Load data
    df = load_data()

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    # Remove unnecessary ID column
    df = df.drop("student_id", axis=1)

    # Target variable
    target = "exam_score"

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessing pipeline
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print("\nPreprocessing completed successfully.")

    print("Training data shape:", X_train_processed.shape)
    print("Testing data shape:", X_test_processed.shape)

    print(
        "\nPreprocessor saved at:",
        PREPROCESSOR_PATH
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )

# -----------------------------
# Run preprocessing
# -----------------------------

if __name__ == "__main__":

    preprocess_data()