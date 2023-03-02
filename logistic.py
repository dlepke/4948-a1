"""
Basic logistic model to get baseline for accuracy, precision, f1, etc.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

# print(df.head())

X = df.copy()
y = df[['Group']]
del X['Group']

# print(df.describe())
# missing values for SES, MMSE, need imputer

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

kfold = KFold(n_splits=5, shuffle=True)

accuracies = []

for train_index, test_index in kfold.split(X):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	model = LogisticRegression(fit_intercept=True, solver="liblinear")

	model.fit(X_train, y_train.values.ravel())
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)

	y_test_array = np.array(y_test['Group'])
	
	accuracy = accuracy_score(y_test_array, y_pred)
	accuracies.append(accuracy)

	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)
	print(classification_report(y_test, y_pred))
	
print(accuracies)

"""
Results:

Accuracy: 0.92

precision: 1/0.25/0.4
recall: 0.9/1/0.95
f1: 0.93/1/0.96

confusion matrix:
2/3/3
0/28/0
0/0/39
"""
