from typing import Union
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import numpy as np
from starter.ml.data import process_data
from humps import camelize
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

with open(f'{BASE_DIR}/starter/saved_model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)
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

# reference: https://medium.com/analytics-vidhya/camel-case-models-with-fast-api-and-pydantic-5a8acb6c0eee
def to_hyphen(underscore_string):
    return f"{underscore_string}".replace('_', '-')

class Predictor(BaseModel):
    age: int = Field(..., example=31)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=45781)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., example=14)
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=5000)
    capital_loss: int = Field(..., example=5000)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")

    class Config:
        alias_generator = to_hyphen
        allow_population_by_field_name = True

@app.post("/predict")
async def predict(payload: Predictor):

    payload_dict = payload_dict = payload.dict(by_alias=True)
    payload_dataframe = pd.DataFrame(payload_dict, columns=payload_dict.keys(), index=[0])


    # prepare the data
    processed, _, _, _ = process_data(
        payload_dataframe,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Calling the inference function to make a prediction
    raw_pred = model.predict(processed)

    # Return the prediction in the expected format
    if raw_pred == 0:
        pred = "Income <= 50k"
    elif raw_pred == 1:
        pred = "Income > 50k"

    res = {"prediction": pred}
    return res



@app.get("/")
async def welcome():
    result = "Welcome to CENSUS API. " \
             "You can type in a JSON body containing 14 attributes to get back a salary prediction"

    return {"result": result, "health_check": "OK"}
