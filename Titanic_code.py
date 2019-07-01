import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
df = pd.read_csv('train.csv')
c = df.iloc[:,6] + df.iloc[:,7]
df["Family"] = c
df.drop[]

df = pd.get_dummies(df,columns = ['Embarked'], drop_first = True)


X = df.iloc[:, [2,4,5,-1,-2,-3]].values
y = df.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

from sklearn.svm import SVC
classifier_SVC = SVC(random_state = 0)

from sklearn.model_selection import GridSearchCV

parameters = [{"C":[0.001,0.01,0.1,1,10,100], "kernel" : ["poly"], "gamma" : [1,0.5,0.1,0.01,0.001], 'degree':[2,3,4,5] }, {"C":[0.001,0.01,0.1,1,10,100],"kernel" : ["rbf"], "gamma" : [0.5,0.1,0.01,0.001] }]

searcher = GridSearchCV(classifier_SVC, param_grid = parameters, scoring = "accuracy", cv = 5, n_jobs = -1)

searcher.fit(X, y)

best_accuracy = searcher.best_score_
best_para = searcher.best_params_

classifier_SVC = SVC(C = 100, kernel = "rbf", gamma = 0.01, random_state = 0)
classifier_SVC.fit(X, y)
print(classifier_SVC.score(X, y))


df_test = pd.read_csv('test_1.csv')
df_test["Family"] = df_test["SibSp"] + df_test["Parch"]

df_test = pd.get_dummies(df_test,columns = ['Embarked'], drop_first = True)

X_test = df_test.iloc[:,[1,3,4,-1,-2,-3]].values

imputer = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

y_pred = classifier_SVC.predict(X_test)

df_ans = pd.read_csv('My_result.csv')

df_ans["Survived"] = y_pred

df_ans.to_csv('My_result.csv', index = False)



