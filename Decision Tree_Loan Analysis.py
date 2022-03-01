# # Decision Tree & Random Forest Project 
# Analyzing loan data-- How can we predict whether a borrower will fully pay back their loan? 

### Imports Library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib inline

### Get the Data
# It is a lending data from 2007-2010 from "LendingClub.com".  
loans = pd.read_csv('loan_data.csv')


### Brief view on the Data
loan.head()
loan.describe()
loan.info()


### Exploratory Data Analysis
# 1. Compare FICO score and the credit.policy outcome with histogram
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
# There's a trend that people with lower FICO are less likely to meet the credit underwriting criteria.


# 2. Compare FICO score and the not.fully.pay outcome with histogram
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
# There's a trend that people with lower FICO are more likely to fail to pay pack the loan in the end.


# 3. Calculate the counts of loans by purpose, with the color hue defined by "not.fully.paid" with countplot.
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# 4. Observe the trend between FICO score and interest rate with jointplot. 
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# 5. Examine the trend differed between not.fully.paid and credit.policy with lmplot.
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

### Adjusting Data
# The purpose column as categorical, therefore I transformed them using dummy variables in order to
# let sklearn understand the feature.
new_loan = pd.get_dummies(loan, columns = ['purpose'],drop_first=True)
new_loan.head()


### Create Training and Testing Data
# Split the data into training and testing sets.
from sklearn.model_selection import train_test_split
X = new_loan.drop(columns = ['not.fully.paid'],axis = 1)
y = new_loan['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


### Training the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
### Train/fit on the training data
tree.fit(X_train, y_train)

### Predicting Test Data
tree_prediction = tree.predict(X_test)


### Evaluating the Model
# Evaluate model performance by confusion matrix and classification_report.
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,tree_prediction))
print(classification_report(y_test,tree_prediction))
# Accuracy is around 73%.


### Training the Random Forest Model
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(n_estimators=200)
### Train/fit on the training data
RF.fit(X_train, y_train)

### Predicting Test Data
RF_prediction1 = RF.predict(X_test)


### Evaluating the Model
# Evaluate model performance by confusion matrix and classification_report.
print(confusion_matrix(y_test,RF_prediction1))
print(classification_report(y_test,RF_prediction1))
# Accuracy is improved to 85%, I wanna check whether enhancing n_estimators would help.

### n_estimators=500
RF= RandomForestClassifier(n_estimators=500)
RF.fit(X_train, y_train)
RF_prediction2 = RF.predict(X_test)
print(confusion_matrix(y_test,RF_prediction2))
print(classification_report(y_test,RF_prediction2))
# Accuracy remains 85%, but precision of not.fully.paid=1 reduce, so no better
# I wanna change the direction to try reducing n_estimators.

### n_estimators=100
RF= RandomForestClassifier(n_estimators=100)
RF.fit(X_train, y_train)
RF_prediction3 = RF.predict(X_test)
print(confusion_matrix(y_test,RF_prediction3))
print(classification_report(y_test,RF_prediction3))
# Accuracy remains 85%, but precision of not.fully.paid=1 reduce, so no better
# It seems that we cannot further improve ideal precision by adjusting n_estimators.

### Adjusting variables for model training 
# Next I would like to know, whether features we put in will affect model performance.
nX = new_loan[['credit.policy','int.rate','fico','purpose_credit_card','purpose_debt_consolidation','purpose_educational','purpose_home_improvement','purpose_major_purchase','purpose_small_business']]
ny = new_loan['not.fully.paid']
nX_train, nX_test, ny_train, ny_test = train_test_split(nX, ny, test_size=0.30, random_state=101)

RF= RandomForestClassifier(n_estimators=200)
RF.fit(nX_train, ny_train)
RF_prediction4 = RF.predict(nX_test)
print(confusion_matrix(ny_test,RF_prediction4))
print(classification_report(ny_test,RF_prediction4))
# Accuracy is reduced to 80%.
### Conclusion: From the view of model accuracy, n_estimators and features we put in the model would affect 
### the model performance.



