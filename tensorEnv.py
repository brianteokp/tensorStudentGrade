import tensorflow
import keras

# run in terminal
# activate tensor
# pip install sklearn
# pip install pandas
# pip install numpy

import pandas as pd
import sklearn
import numpy as np

from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import model_selection

data = pd.read_csv("student-mat.csv", sep =";")
# original data separated by ;
print(data.head)

data = data[["G1","G2","G3","studytime","failures","absences"]]
# subsetting dataset (note currently using only integer data)

print(data.head)

predict = "G3"
# Predicting final grade based on other features

# Defining features
X = np.array(data.drop([predict],1))
X_label = ["G1", "G2", "studytime", "failures", "absences"]
y = np.array(data[predict])
# Create an array and remove our output
print(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# Model creation
linear = linear_model.LinearRegression()

linear.fit(X_train,y_train)
acc = linear.score(X_test,y_test)
print(acc)

print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

print(y_test - predictions)

G1 = int(input("Insert G1 here: "))
G2 = int(input("Insert G2 here: "))
st = int(input("Insert studytime here: "))
fail = int(input("Insert number of failures here: "))
absence = int(input("Insert number of absences here: "))

predict_feature = ([[G1,G2,st,fail,absence]])
predict_grade = float((linear.predict(predict_feature)))
predict_grade = round(predict_grade)
print("Predicted grade is: " + str((predict_grade)))






