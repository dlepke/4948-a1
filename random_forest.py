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
from sklearn.model_selection import RandomizedSearchCV

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

# X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# smote = SMOTE()
# X_smote, y_smote = smote.fit_resample(X_train, y_train)
#
# # # scaler = StandardScaler()
# # # X = scaler.fit_transform(X)
#
# rf = RandomForestClassifier(n_estimators=10000, max_features=2, max_depth=4)
#
# rf.fit(X_smote, y_smote.values.ravel())
# y_pred = rf.predict(X_test)
# y_prob = rf.predict_proba(X_test)
#
# y_test_array = np.array(y_test['Group'])
#
# cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(cm)
# print(classification_report(y_test, y_pred))


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

	model = RandomForestClassifier(n_estimators=10000, max_features=2, max_depth=4)

	model.fit(X_smote, y_smote.values.ravel())
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)

	y_test_array = np.array(y_test['Group'])

	accuracies.append(accuracy_score(y_test_array, y_pred))

	cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
	print(cm)
	print(classification_report(y_test, y_pred))

print(accuracies)


"""
Get numerical feature importances
"""
# importances = list(rf.feature_importances_)
#
# df_importance = pd.DataFrame()
#
# for i in range(0, len(importances)):
# 	df_importance = df_importance.append({"importance": importances[i], "feature": X.columns[i]}, ignore_index=True)
#
# df_importance = df_importance.sort_values(by=['importance'], ascending=False)
# print(df_importance)

"""
Grid searching for best model params found:
Best parameters:
{'n_estimators': 10000, 'max_features': 2, 'max_depth': 4}
"""
# estimator_count = [100, 500, 1000, 10000]
# max_feature_count = ['auto', 2, 3, 4, 5, None]
# max_depths = [1, 2, 3, 4, 5]
#
# random_grid = {
# 	'n_estimators': estimator_count,
# 	'max_features': max_feature_count,
# 	'max_depth': max_depths
# }
#
# test_rf = RandomForestClassifier()
# rf_random = RandomizedSearchCV(
# 	estimator=test_rf, param_distributions=random_grid,
# 	n_iter=100, cv=3, verbose=2, n_jobs=-1)
#
# rf_random.fit(X_train, y_train)
#
# print("Best parameters:")
# print(rf_random.best_params_)


