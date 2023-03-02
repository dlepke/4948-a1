"""
Bagging classifier model using random forests.

4948 Lesson 3.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

X = df.copy()
y = df[['Group']]
del X['Group']

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

kfold = KFold(n_splits=5, shuffle=True)

accuracies = []

for train_index, test_index in kfold.split(X_smote):
	X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2)
	
	rf = RandomForestClassifier(n_estimators=1000)
	
	rf.fit(X_train, y_train.values.ravel())
	y_pred = rf.predict(X_test)
	# y_prob = rf.predict_proba(X_test)
	
	y_test_array = np.array(y_test['Group'])
	
	accuracies.append(accuracy_score(y_test_array, y_pred))
	
	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)
	print(classification_report(y_test, y_pred))

print(accuracies)




