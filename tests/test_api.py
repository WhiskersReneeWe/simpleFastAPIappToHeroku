import pytest
from fastapi.testclient import TestClient
from main import app

# comment
testApp = TestClient(app)

lessthan50KPayload = {
  "age": 29,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}

morethan50KPayload = {
          "age": 50,
          "workclass": "Federal-gov",
          "fnlgt": 251585,
          "education": "Bachelors",
          "education_num": 13,
          "marital_status": "Divorced",
          "occupation": "Exec-managerial",
          "relationship": "Not-in-family",
          "race": "White",
          "sex": "Male",
          "capital_gain": 0,
          "capital_loss": 0,
          "hours_per_week": 55,
          "native_country": "United-States"
}

badPayload = {}

@pytest.fixture
def lessthan50KPayload_fixture():
    return lessthan50KPayload

@pytest.fixture
def morethan50KPayload_fixture():
    return morethan50KPayload

def test_get():
    res = testApp.get("/")
    assert res.json()["result"] == "Welcome to CENSUS API. You can type in a JSON body containing 14 attributes to get back a salary prediction"
    assert res.status_code == 200

def test_post_correct(lessthan50KPayload_fixture):
    res = testApp.post("/predict", json=lessthan50KPayload_fixture)
    assert res.status_code == 200
    assert res.json()["prediction"] == "Income < 50k"

def test_post_wrong(morethan50KPayload_fixture):
    res = testApp.post("/predict", json=morethan50KPayload_fixture)
    assert res.status_code == 200
    assert res.json()["prediction"] != "Income > 50k"
