import pandas as pd
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

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

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

demented = df[df['Group'] == 'Demented']
not_demented = df[df['Group'] == 'Nondemented']
converted = df[df['Group'] == 'Converted']

plt.figure(figsize=(30, 15))

plt.subplot(331)
plt.hist([demented['M/F'], not_demented['M/F'], converted['M/F']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by gender")

plt.subplot(332)
plt.hist([demented['Age'], not_demented['Age'], converted['Age']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by age")

plt.subplot(333)
plt.hist([demented['EDUC'], not_demented['EDUC'], converted['EDUC']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by education")

plt.subplot(334)
plt.hist([demented['SES'], not_demented['SES'], converted['SES']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by SES")

plt.subplot(335)
plt.hist([demented['MMSE'], not_demented['MMSE'], converted['MMSE']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by MMSE")

plt.subplot(336)
plt.hist([demented['CDR'], not_demented['CDR'], converted['CDR']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by CDR")

plt.subplot(337)
plt.hist([demented['eTIV'], not_demented['eTIV'], converted['eTIV']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by eTIV")

plt.subplot(338)
plt.hist([demented['nWBV'], not_demented['nWBV'], converted['nWBV']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by nWBV")

plt.subplot(339)
plt.hist([demented['ASF'], not_demented['ASF'], converted['ASF']], rwidth=0.8)
plt.legend(['Demented', 'Not demented', 'Converted'])
plt.title("Frequency of demented, non-demented, and converted patients by ASF")

plt.show()


"""
Notes

methods used:
- imputing using KNNImputer for missing values in SES and MMSE
- dummy variables for the categorical M/F column
- SMOTE due to the 'Converted' target value being a significant minority and consistently miscategorized without SMOTE
- StandardScaler was tested for logistic regression with little to no improvement in results
- TODO: try binning age

Feature selection:
- RFE (for basic logistic regression)
- FFS (for basic logistic regression)
- Feature importance (in random forest module)
- Based on the results from these methods, removed from logistic and bagging model:
	- EDUC
	- MMSE
- For mlxtend and random forest, removed:
	- M/F
	- EDUC
	- nWBV
- For stacked model, TBD

"""