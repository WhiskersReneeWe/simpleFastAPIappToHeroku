import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
import numpy as np
import logging
import os


# Add code to load in the data, model and encoder
df = pd.read_csv("data/raw/census.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
model = pd.read_pickle(r"model.pkl")
encoder = pd.read_pickle(r"encoder.pkl")
lb = pd.read_pickle(r"lb.pkl")


slice_metrics = []
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


_, test_set = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df.salary)


for cat in cat_features:
    for cls in test_set[cat].unique():
        df_temp = test_set[test_set[cat] == cls]
        encoder = pd.read_pickle(r"encoder.pkl")
        X_test, y_test, _, _ = process_data(
            df_temp,
            cat_features,
            label=None, encoder=encoder, lb=lb, training=False)

        y_preds = inference(model, X_test)
        y = df_temp.iloc[:, -1:]
        lb = LabelEncoder()
        y = lb.fit_transform(np.ravel(y))
        prc, rcl, fb = compute_model_metrics(y, y_preds)
        line = "[%s->%s] Precision: %s " \
               "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
        logging.info(line)
        slice_metrics.append(line)


with open('slice_output.txt', 'w') as out:
    for slice_metric in slice_metrics:
        out.write(slice_metric + '\n')