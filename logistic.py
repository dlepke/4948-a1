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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, f_classif

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

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

model = LogisticRegression(fit_intercept=True, solver="liblinear")

"""
Between the two feature selection methods,
the following features are strongly recommended:

ASF
M/F_F
nWBV

They disagree on:
CDR
SES

Feature importance (in bagging classifier module) showed:
importance    feature
3    0.482053     CDR
2    0.236816    MMSE
1    0.091915     SES
4    0.067490    eTIV
5    0.066786     ASF
0    0.054942     Age

So it agrees with RFE on CDR, SES, and ASF
and it agrees with FFS on ASF, Age, and eTIV

Overall selection of features:
ASF, nWBV, CDR, SES, Age, eTIV, M/F_M, and M/F_F
"""

"""
Recursive feature selection:
RFE FEATURES SELECTED
[False False  True False  True False  True  True  True False]
SES
CDR
nWBV
ASF
M/F_F
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
feature         ffs
7     ASF    1.430156
5    eTIV    2.684062
0     Age   14.149083
8   M/F_F   18.656900
9   M/F_M   18.656900
6    nWBV   28.090159
1    EDUC   28.760364
2     SES   41.344708
3    MMSE  170.239290
4     CDR  496.623041
"""

# ffs = f_classif(X_smote, y_smote.values.ravel())
#
# featuresDf = pd.DataFrame()
# for i in range(0, len(X.columns)):
# 	featuresDf = featuresDf.append({"feature": X.columns[i], "ffs": ffs[0][i]}, ignore_index=True)
# featuresDf = featuresDf.sort_values(by=['ffs'])
# print("FFS FEATURES SELECTED")
# print(featuresDf)

kfold = KFold(n_splits=5, shuffle=True)

accuracies = []

for train_index, test_index in kfold.split(X_smote):
	# X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2)
	X_train = X.loc[X.index.intersection(train_index), :]
	X_test = X.loc[X.index.intersection(test_index), :]
	y_train = y.loc[y.index.intersection(train_index), :]
	y_test = y.loc[y.index.intersection(test_index), :]

	model = LogisticRegression(fit_intercept=True, solver="liblinear")

	model.fit(X_train, y_train.values.ravel())
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)

	y_test_array = np.array(y_test['Group'])

	accuracies.append(accuracy_score(y_test_array, y_pred))

	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)
	print(classification_report(y_test, y_pred))

print(accuracies)
