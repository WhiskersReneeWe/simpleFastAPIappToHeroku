import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
import numpy as np
import json
import pickle
from pathlib import Path

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

_, test = train_test_split(df, test_size=0.20)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)


for cate in cat_features:
    result = {}
    for cate_j in df[cate].unique():
        is_slice = (df[cate] == cate_j).values
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        result[cate_j] = {"precision": precision, "recall": recall, "fbeta": fbeta}

with open(f'{BASE_DIR}/slice_performance.txt', 'w') as out:
    out.write(json.dumps(result, indent = 4))

    # return result
    # for cls in test[cat].unique():
    #     df_cls = test[test[cat] == cls]
    #
    #
    #
    #     y_preds = model.predict(X_test)
    #     y = df_cls.iloc[:, -1:]
    #     #lb = LabelEncoder()
    #     y = lb.fit_transform(np.ravel(y))
    #     prc, rcl, fb = compute_model_metrics(y, y_preds)
    #     line = "[%s->%s] Precision: %s " \
    #            "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
    #     logging.info(line)
    #     slice_metrics.append(line)


#with open(f'{BASE_DIR}/slice_performance.txt', 'w') as out:
    # for slice_metric in slice_metrics:
    #     out.write(slice_metric + '\n')