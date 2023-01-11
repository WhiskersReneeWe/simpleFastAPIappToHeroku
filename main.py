from typing import Union
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
from starter.starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

app = FastAPI()

# load the models as global constants
BASE_DIR = Path(__file__).resolve(strict=True).parent
print(BASE_DIR)

with open(f'{BASE_DIR}/starter/saved_model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
    f.close()

with open(f'{BASE_DIR}/starter/saved_model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
    f.close()

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

class Predictor(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.post("/predict")
async def predict(payload: Predictor):
    payload_dict = dict(payload)
    payload_dataframe = pd.DataFrame(payload_dict, columns=payload_dict.keys(), index=[0])

    # convert all the "_ " to "-" in the dataframe
    payload_dataframe.columns = payload_dataframe.columns.str.replace("_", "-")

    # prepare the data
    processed_epoch, _, _, _ = process_data(
        payload_dataframe,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        label=None,
    )
    # Calling the inference function to make a prediction
    raw_pred = model.predict(processed_epoch)

    # Return the prediction in the expected format
    if raw_pred == 0:
        pred = "Income < 50k"
    elif raw_pred == 1:
        pred = "Income > 50k"

    res = {"prediction": pred}
    return res

@app.get("/")
async def welcome():
    result = "Welcome to CENSUS API. " \
             "You can type in a JSON body containing 14 attributes to get back a salary prediction"

    return {"result": result, "health_check": "OK"}
