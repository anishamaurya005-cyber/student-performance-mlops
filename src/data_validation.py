import pandas as pd
import os


DATA_PATH = "data/student_data.csv"


REQUIRED_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "academic_level",
    "study_hours",
    "self_study_hours",
    "online_classes_hours",
    "social_media_hours",
    "gaming_hours",
    "sleep_hours",
    "screen_time_hours",
    "exercise_minutes",
    "caffeine_intake_mg",
    "part_time_job",
    "upcoming_deadline",
    "internet_quality",
    "mental_health_score",
    "focus_index",
    "burnout_level",
    "productivity_score",
    "exam_score"
]


def validate_data():

    # Check whether file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Check columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Column validation: PASSED")

    # Check missing values
    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    print("Missing value validation: PASSED")

    # Check duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Dataset contains {duplicates} duplicate rows."
        )

    print("Duplicate validation: PASSED")

    # Check target column
    if not pd.api.types.is_numeric_dtype(df["exam_score"]):
        raise ValueError(
            "exam_score must be numeric."
        )

    print("Target validation: PASSED")

    # Check exam score range
    if (df["exam_score"] < 0).any() or \
       (df["exam_score"] > 100).any():

        raise ValueError(
            "exam_score must be between 0 and 100."
        )

    print("Exam score range validation: PASSED")

    print("\nDATA VALIDATION PASSED SUCCESSFULLY!")

    return True


if __name__ == "__main__":
    validate_data()