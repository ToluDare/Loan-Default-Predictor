
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


#loading data
data = pd.read_csv('bank_marketing.csv')

#---DATA CLEANING---
# printing the columns names, data type and size to understand what we are dealing with
#print(data.info())
#print(data.shape)  # 41188 rows and 21 columns

#-cleaning up column names
data.columns = data.columns.str.strip().str.lower().str.replace(
    ' ', '_').str.replace('(', '').str.replace(')', '')

#-removing non-useful and leakage columns to prevent incorrect predictions
cols_to_drop = [
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome"
]
data = data.drop(columns=cols_to_drop)

#-checking misssing values
#sorting the columns (in table copy) based off the percent of missing values (descending order)

data.isna().sum().sort_values(ascending=False)
missing_percent = round(data.isna().mean() * 100, 2)

print(
    missing_percent
)  #since i picked such a nice dataset, there are only 0.02% missing vlaues in the age column >_<
#but usally columns with 40%+ missing values are dropped

#-fixing missing values - putting n/a
data["age"] = data["age"].fillna(data["age"].median())
#median to preserve the distribution of age.

#-removing duplicates
data.duplicated().sum()  # 12 duplicates
data = data.drop_duplicates()

#-fixing data types - all colums are already in the correct data type.

#--SAVING DATA FOR TABLEAU--
data.to_csv('cleaned_bank_data.csv', index=False)

#--DATA PREPARATION--

#-mapping and removing values in leaking column

y = data["default"].str.strip().map({"no": 1, "unknown": 0})
y = y.fillna(0).astype(int) #target
x = data.drop(columns=["default"]) #feature

#-one hot encoding the categorical columns
x = pd.get_dummies(x, drop_first=True)

#--SAFETY CHECKS--
#checking for missing values and data types

#x.info()
assert x.isna().sum().sum() == 0
#y.info()
assert y.isna().sum().sum() == 0

#--SAVING THE CLEANED DATA--
data.to_csv('cleaned_encoded_bank_data.csv', index=False)

#--TRAINING/TESTING THE DATA--
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, #20% testing data, 80% training data
    random_state=42,
    stratify=y  #stratifying to ensure the same distribution of the target variable in both training and testing sets
)
#--SCALING THE DATA--
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--TRAINING THE MODEL--
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

#--EVALUATING THE MODEL--
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#--METRICS--
print("ROC-AUC:", roc_auc_score(y_test, y_prob))#0.65 would be a good baseline
print(classification_report(y_test, y_pred)) # num of defaulters

#--!!PREDICTIONS!!--
coef_data = pd.DataFrame({
    "feature": x.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)
print("-----")
coef_data.head(10)    
coef_data.info()
