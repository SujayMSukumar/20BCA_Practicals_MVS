#Logistic regression using titanic dataset

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.chdir("C:\Users\Sujay M S\Desktop\DATASETS")
data = pd.read_csv("titanic.csv")
data.head()
data.info()
sex = pd.get_dummies(data['Sex'])
embark = pd.get_dummies(data['Embarked'])
data.drop(['Sex','Embarked','Name','Ticket','Cabin','Age'],axis=1,inplace=True)
data = pd.concat([data,sex,embark],axis=1)
data.info()
X = data.drop(['Survived'], axis=1)
y = data['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test,y_pred))