"""
Ensemble classifier model using mlxtend classifier.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

X = df.copy()
y = df[['Group']]
del X['Group']
# based on feature importance ratings
del X['M/F']
del X['EDUC']
del X['nWBV']

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

kfold = KFold(n_splits=5, shuffle=True)

accuracies = []

ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting="hard")
# classifiers = [ada_boost, grad_boost, xgb_boost, eclf]

# for clf in classifiers:
for train_index, test_index in kfold.split(X_smote):
	# X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2)
	X_train = X.loc[X.index.intersection(train_index), :]
	X_test = X.loc[X.index.intersection(test_index), :]
	y_train = y.loc[y.index.intersection(train_index), :]
	y_test = y.loc[y.index.intersection(test_index), :]
	print(eclf.__class__.__name__)
	eclf.fit(X_train, y_train.values.ravel())
	y_pred = eclf.predict(X_test)
	report = classification_report(y_test, y_pred)
	print(report)
	
	y_test_array = np.array(y_test['Group'])

	accuracies.append(accuracy_score(y_test_array, y_pred))

	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)

print(accuracies)
