from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    with open("sample_leaf.jpg", "rb") as img:
        response = client.post("/api/predict", files={"file": ("sample_leaf.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
