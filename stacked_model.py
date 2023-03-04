import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./alzheimer.csv', skiprows=1, names=(
	'Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
))

X = df.copy()
y = df[['Group']].copy()
del X['Group']
del X['EDUC']

X = pd.get_dummies(X)  # convert m/f to dummy columns

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

smote = SMOTE()
X, y = smote.fit_resample(X, y)

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

kfold = KFold(n_splits=5, shuffle=True)

logistic_accuracies = []
rf_accuracies = []
bag_accuracies = []
eclf_accuracies = []
stacked_accuracies = []

for train_index, test_index in kfold.split(X):
	X_train = X.loc[X.index.intersection(train_index), :]
	X_test = X.loc[X.index.intersection(test_index), :]
	y_train = y.loc[y.index.intersection(train_index), :]
	y_test = y.loc[y.index.intersection(test_index), :]
	X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50)

	print("***** Base Models *****")
	
	""" Logistic regression """
	
	X_logistic_train = X_train.copy()
	# based on visualization observations
	del X_logistic_train['Age']
	del X_logistic_train['eTIV']
	del X_logistic_train['ASF']
	
	X_test_logistic = X_test.copy()
	del X_test_logistic['Age']
	del X_test_logistic['eTIV']
	del X_test_logistic['ASF']
	
	logistic_model = LogisticRegression(fit_intercept=True, solver="liblinear")
	
	logistic_model.fit(X_logistic_train, y_train.values.ravel())
	y_pred_logistic = logistic_model.predict(X_test_logistic)
	
	y_test_array = np.array(y_test['Group'])
	
	cm = pd.crosstab(y_test_array, y_pred_logistic, rownames=['Actual'], colnames=['Predicted'])
	print("Logistic regression")
	print(cm)
	print(classification_report(y_test, y_pred_logistic))
	
	""" Random forest """
	rf = RandomForestClassifier(n_estimators=10000, max_features=2, max_depth=4)
	
	X_train_rf = X_train.copy()
	del X_train_rf['M/F_F']
	del X_train_rf['M/F_M']
	del X_train_rf['nWBV']
	
	X_test_rf = X_test.copy()
	del X_test_rf['M/F_F']
	del X_test_rf['M/F_M']
	del X_test_rf['nWBV']

	rf.fit(X_train_rf, y_train.values.ravel())
	y_pred_rf = rf.predict(X_test_rf)
	
	y_test_array = np.array(y_test['Group'])
	
	cm = pd.crosstab(y_test_array, y_pred_rf, rownames=['Actual'], colnames=['Predicted'])
	print("Random forest")
	print(cm)
	print(classification_report(y_test, y_pred_rf))
	
	""" Bagging classifier """
	bag_clf = BaggingClassifier(n_estimators=1000)
	
	X_train_bag = X_train.copy()
	del X_train_bag['MMSE']
	
	X_test_bag = X_test.copy()
	del X_test_bag['MMSE']
	
	bag_clf.fit(X_train_bag, y_train)
	y_pred_bag = bag_clf.predict(X_test_bag)
	
	y_test_array = np.array(y_test['Group'])

	cm = pd.crosstab(y_test_array, y_pred_bag, rownames=['Actual'], colnames=['Predicted'])
	print("Bagging classifier")
	print(cm)
	print(classification_report(y_test, y_pred_bag))
	
	""" Ensemble classifier """
	ada_boost = AdaBoostClassifier()
	grad_boost = GradientBoostingClassifier()
	xgb_boost = XGBClassifier()
	eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting="hard")
	
	X_train_eclf = X_train.copy()
	del X_train_eclf['M/F_F']
	del X_train_eclf['M/F_M']
	
	X_test_eclf = X_test.copy()
	del X_test_eclf['M/F_F']
	del X_test_eclf['M/F_M']
	
	eclf.fit(X_train_eclf, y_train.values.ravel())
	y_pred_eclf = eclf.predict(X_test_eclf)
	
	y_test_array = np.array(y_test['Group'])
	
	cm = pd.crosstab(y_test_array, y_pred_eclf, rownames=['Actual'], colnames=['Predicted'])
	print("Ensemble classifier")
	print(cm)
	print(classification_report(y_test, y_pred_eclf))
	
	""" Stacked model """
	def convert_to_dummy(cat_df):
		columns = cat_df.columns
		
		for k in columns:
			imputed_column = []
			
			for j in range(len(cat_df)):
				if cat_df.loc[j][k] == 'Demented':
					imputed_column.append(0)
				elif cat_df.loc[j][k] == 'Nondemented':
					imputed_column.append(1)
				else:
					imputed_column.append(2)
				
			cat_df[k] = imputed_column
		return cat_df


	""" These predictions go into building the stacked model """
	models = [logistic_model, rf, bag_clf, eclf]
	df_val_predictions = pd.DataFrame()
	
	X_val_logistic = X_val.copy()
	del X_val_logistic['Age']
	del X_val_logistic['eTIV']
	del X_val_logistic['ASF']
	df_val_predictions[0] = logistic_model.predict(X_val_logistic)
	
	X_val_rf = X_val.copy()
	del X_val_rf['M/F_F']
	del X_val_rf['M/F_M']
	del X_val_rf['nWBV']
	df_val_predictions[1] = rf.predict(X_val_rf)
	
	X_val_bag = X_val.copy()
	del X_val_bag['MMSE']
	df_val_predictions[2] = bag_clf.predict(X_val_bag)
	
	X_val_eclf = X_val.copy()
	del X_val_eclf['M/F_F']
	del X_val_eclf['M/F_M']
	df_val_predictions[3] = eclf.predict(X_val_eclf)
	
	df_val_predictions = convert_to_dummy(df_val_predictions)
	
	stacked_model = LogisticRegression()
	stacked_model.fit(df_val_predictions, y_val)
	
	""" These predictions are for testing the stacked model """
	df_validation_predictions = pd.DataFrame()

	X_test_logistic = X_test.copy()
	del X_test_logistic['Age']
	del X_test_logistic['eTIV']
	del X_test_logistic['ASF']
	df_validation_predictions[0] = logistic_model.predict(X_test_logistic)
	
	X_test_rf = X_test.copy()
	del X_test_rf['M/F_F']
	del X_test_rf['M/F_M']
	del X_test_rf['nWBV']
	df_validation_predictions[1] = rf.predict(X_test_rf)
	
	X_test_bag = X_test.copy()
	del X_test_bag['MMSE']
	df_validation_predictions[2] = bag_clf.predict(X_test_bag)
	
	X_test_eclf = X_test.copy()
	del X_test_eclf['M/F_F']
	del X_test_eclf['M/F_M']
	df_validation_predictions[3] = eclf.predict(X_test_eclf)
	
	df_validation_predictions = convert_to_dummy(df_validation_predictions)
	
	# Evaluate stacked model with validation data.
	stacked_predictions = stacked_model.predict(df_validation_predictions)
	print("Stacked Model")
	print(cm)
	print(classification_report(y_test, stacked_predictions))
	
	stacked_accuracies.append(accuracy_score(y_test, stacked_predictions))

print("stacked accuracies: ", stacked_accuracies)
