from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import pickle

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

BASE_DIR = Path(__file__).resolve(strict=True).parent
print(BASE_DIR)
# Add code to load in the data, model and encoder
df = pd.read_csv(f"{BASE_DIR}/data/census.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# load saved models

with open(f'{BASE_DIR}/starter/saved_model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
    f.close()

with open(f'{BASE_DIR}/starter/saved_model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
    f.close()

with open(f'{BASE_DIR}/starter/saved_model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)
    f.close()

_, test = train_test_split(df, test_size=0.20)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(precision,recall,fbeta)
