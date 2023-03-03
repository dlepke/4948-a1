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
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, f_classif

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

X = df.copy()
y = df[['Group']].copy()
del X['Group']
del X['EDUC']

# based on visualization observations
del X['Age']
del X['eTIV']
del X['ASF']
# del X['MMSE']

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# model = LogisticRegression(fit_intercept=True, solver="liblinear")

"""
Recursive feature selection:
"""
# rfe = RFE(model)
#
# rfe = rfe.fit(X_smote, y_smote.values.ravel())
#
# print("RFE FEATURES SELECTED")
# print(rfe.support_)
#
# columns = list(X_smote.keys())
# for i in range(0, len(columns)):
# 	if rfe.support_[i]:
# 		print(columns[i])
		
		
"""
Forward feature selection:
"""
# ffs = f_classif(X_smote, y_smote.values.ravel())
#
# featuresDf = pd.DataFrame()
# for i in range(0, len(X.columns)):
# 	featuresDf = featuresDf.append({"feature": X.columns[i], "ffs": ffs[0][i]}, ignore_index=True)
# featuresDf = featuresDf.sort_values(by=['ffs'])
# print("FFS FEATURES SELECTED")
# print(featuresDf)

"""
Non-kfold
"""
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# smote = SMOTE()
# X_smote, y_smote = smote.fit_resample(X_train, y_train)
#
# # scaler = StandardScaler()
# # X = scaler.fit_transform(X)
#
# model = LogisticRegression(fit_intercept=True, solver="liblinear")
#
# model.fit(X_smote, y_smote.values.ravel())
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)
#
# y_test_array = np.array(y_test['Group'])
#
# cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(cm)
# print(classification_report(y_test, y_pred))

"""
Kfold
"""
# kfold = KFold(n_splits=5, shuffle=True)
#
# accuracies = []

# for train_index, test_index in kfold.split(X):
# 	X_train = X.loc[X.index.intersection(train_index), :]
# 	X_test = X.loc[X.index.intersection(test_index), :]
# 	y_train = y.loc[y.index.intersection(train_index), :]
# 	y_test = y.loc[y.index.intersection(test_index), :]
#
# 	smote = SMOTE()
# 	X_smote, y_smote = smote.fit_resample(X_train, y_train)
#
# 	# scaler = StandardScaler()
# 	# X = scaler.fit_transform(X)
#
# 	model = LogisticRegression(fit_intercept=True, solver="liblinear")
#
# 	model.fit(X_smote, y_smote.values.ravel())
# 	y_pred = model.predict(X_test)
# 	y_prob = model.predict_proba(X_test)
#
# 	y_test_array = np.array(y_test['Group'])
#
# 	accuracies.append(accuracy_score(y_test_array, y_pred))
#
# 	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
# 	print(cm)
# 	print(classification_report(y_test, y_pred))
#
# print(accuracies)
