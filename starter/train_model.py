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

# Add the necessary imports for the starter code.

# Add code to load in the data.
#data = pd.read_csv("starter/data/census.csv")
data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "data", "census.csv")
                      )
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

print(encoder)

# Proces the test data with the process_data function.

if __name__ == "__main__":
    print("Training started ... ")
    # Train and save a model.
    #print(X_train.shape)
    # sample_test = X_train[0].reshape(1, -1)
    #
    # load model
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    # with open(f'{BASE_DIR}/saved_model/trained_model.pkl', 'rb') as f:
    #     model = pickle.load(f)

    trained_model = train_model(X_train, y_train)
    print("Saving the logistic regression model ...")
    with open(f'{BASE_DIR}/saved_model/trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    with open(f'{BASE_DIR}/saved_model/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
