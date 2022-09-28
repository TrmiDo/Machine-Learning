import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
import pickle
from matplotlib import style

data = pd.read_csv("../../student-mat.csv", sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head())
predict = "G3"
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(x,y, test_size =0.1)
"""
This step is for importing. Take out the quotation to do thiss
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print(acc)

#Saving model
with open("studentmodel.pickle","wb") as f:
    pickle.dump(linear,f)
"""
pickle_in = open("../../studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
#Saving best model. Make a for loop then set best. if acc> best then we save the pickle

print("coeficent:", linear.coef_)
print("intercept:", linear.intercept_)
prediction= linear.predict(x_test)
for x in range(len(prediction)):
    print(prediction[x],x_test[x],y_test[x])

p='G2'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

