from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# pour la bonne reponse, l'app doit envoyer le code 200 avec la bonne reponse
def test_predict_success():
    response = client.post("/predict", json={
        "features": [3.5, 1, 4.9]
    })

    assert response.status_code == 200
    assert "predictions" in response.json()



# envoyer un mauvais json, l'app doit envoyer le code 422 

def test_predict_unprocessable_entity():
    response = client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4.9
    })

    assert response.status_code == 422