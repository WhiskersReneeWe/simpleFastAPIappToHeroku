import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from pathlib import Path

@pytest.fixture
def model_encoder_fixture():
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    print(BASE_DIR)

    with open('starter/saved_model/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
        f.close()

    with open('starter/saved_model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        f.close()

    return model, encoder


@pytest.fixture
def data_fixture():
    data = pd.read_csv("data/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    return train, test

# Add code to load in the data.
#data = pd.read_csv("/Users/reneeliu/Desktop/mlops/deployMLFastAPI/starter/data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
#train, test = train_test_split(data, test_size=0.20)

def test_train_model(model_encoder_fixture):
    random_state = 1234
    #training, testing = data_fixture
    model, encoder =  model_encoder_fixture

    # cat_features = [
    #     "workclass",
    #     "education",
    #     "marital-status",
    #     "occupation",
    #     "relationship",
    #     "race",
    #     "sex",
    #     "native-country",
    # ]
    # X_test, y_test, _, lb = process_data(
    #     testing, categorical_features=cat_features, label="salary", training=False, encoder=encoder
    # )
   # model = LogisticRegression(random_state=random_state).fit(X_train, y_train)

    # model = RandomForestClassifier(random_state)
    # model.fit(X_train, y_train)
    assert model is not None
    assert encoder is not None

def test_compute_metrics(data_fixture, model_encoder_fixture):
    """
    This functions tests the computation of the metrics
    """
    random_state = 1234
    training, testing = data_fixture
    model, encoder = model_encoder_fixture
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_test, y_test, encoder, lb = process_data(
        testing, categorical_features=cat_features, label="salary", training=False
    )

    predictions = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert precision >= 0.2
    assert recall >= 0.2
    assert fbeta >= 0

def test_inference(data_fixture, model_encoder_fixture):
    model, encoder = model_encoder_fixture
    training, testing = data_fixture
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_test, y_test, _, lb = process_data(
        testing, categorical_features=cat_features, label="salary", training=False, encoder=encoder
    )
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)  # Assert that the length is the same as x_train

def test_model_types(model_encoder_fixture):
    model, encoder = model_encoder_fixture
    assert model is not None