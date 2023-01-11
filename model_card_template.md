# Model Card


## Model Details
Renee Liu created the model. It is logistic regression using the default hyperparameters in scikit-learn.

## Intended Use
This model should be used to predict the salary of US citizens based on a list of attributes.

## Training Data
Data was obtained from: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
Data was obtained from: https://archive.ics.uci.edu/ml/datasets/census+income
Evaluation data was extracted from the original data by random splitting.

## Metrics
The model was evaluated using precision, recall, fbeta, respectively.

## Ethical Considerations
No comment.

## Caveats and Recommendations
We need a more balanced data because the gender is skewed.
We also need to train a varieties of models to see which one fits better to this dataset.
