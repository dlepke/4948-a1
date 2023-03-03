"""
Bagging classifier model.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

X = df.copy()
y = df[['Group']]
del X['Group']
del X['EDUC']
del X['MMSE']

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


"""
Without kfold
"""
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# smote = SMOTE()
# X_smote, y_smote = smote.fit_resample(X, y)
#
# X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2)
#
# clf = BaggingClassifier(n_estimators=1000)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# y_test_array = np.array(y_test['Group'])
#
# accuracy = accuracy_score(y_test_array, y_pred)
#
# cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(cm)
# print(classification_report(y_test, y_pred))
#
# print(accuracy)

"""
Kfold
"""
kfold = KFold(n_splits=5, shuffle=True)

accuracies = []

for train_index, test_index in kfold.split(X):
	X_train = X.loc[X.index.intersection(train_index), :]
	X_test = X.loc[X.index.intersection(test_index), :]
	y_train = y.loc[y.index.intersection(train_index), :]
	y_test = y.loc[y.index.intersection(test_index), :]

	smote = SMOTE()
	X_smote, y_smote = smote.fit_resample(X_train, y_train)

	# scaler = StandardScaler()
	# X = scaler.fit_transform(X)

	model = BaggingClassifier(n_estimators=1000)

	model.fit(X_smote, y_smote.values.ravel())
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)

	y_test_array = np.array(y_test['Group'])

	accuracies.append(accuracy_score(y_test_array, y_pred))

	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)
	print(classification_report(y_test, y_pred))

print(accuracies)