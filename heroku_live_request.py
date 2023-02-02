import json

import requests

r = requests.get('https://nameless-brushlands-32546.herokuapp.com/')

print(r.status_code)
print(r.json())

data = {
    "age": 33,
    "workclass": "Private",
    "fnlgt": 22222,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married_civ_spouse",
    "occupation": "Prof_specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# row = {}
# for k, v in data.items():
#     row[k.replace("-", '_')] = v
#
# print(row)

r = requests.post('https://nameless-brushlands-32546.herokuapp.com/predict', json=data)
#
print(r.status_code)
print(r.json())