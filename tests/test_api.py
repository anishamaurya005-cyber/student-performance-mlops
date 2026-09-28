from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json()["message"] == "Student Performance Prediction API is running"


def test_predict():
    sample_data = {
        "features": {
            "age": 21,
            "gender": "Female",
            "academic_level": "Undergraduate",
            "study_hours": 4,
            "self_study_hours": 2,
            "online_classes_hours": 2,
            "social_media_hours": 2,
            "gaming_hours": 1,
            "sleep_hours": 7,
            "screen_time_hours": 5,
            "exercise_minutes": 30,
            "caffeine_intake_mg": 100,
            "part_time_job": 0,
            "upcoming_deadline": 0,
            "internet_quality": "Good",
            "mental_health_score": 7,
            "focus_index": 7,
            "burnout_level": 2,
            "productivity_score": 70
        }
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "predicted_exam_score" in response.json()