# Script to train machine learning model.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from starter.starter.ml.data import process_data
# from starter.starter.ml.model import train_model
from ml.data import process_data
from ml.model import train_model
import pickle
from pathlib import Path


# Add code to load in the data.
data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "data", "census.csv")
                      )


# Proces the test data with the process_data function.

if __name__ == "__main__":
    print("Training started ... ")

    # load model
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    # with open(f'{BASE_DIR}/saved_model/trained_model.pkl', 'rb') as f:
    #     model = pickle.load(f)

    train, test = train_test_split(data, test_size=0.20)

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

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    print("Saving the data encoder ...")
    with open(f'{BASE_DIR}/saved_model/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    trained_model = train_model(X_train, y_train)
    print("Saving the logistic regression model ...")
    with open(f'{BASE_DIR}/saved_model/trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
