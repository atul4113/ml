import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

daibe = datasets.load_diabetes()
db_x = daibe.data[:, np.newaxis,2]
db_x_train = db_x[:-200]
db_x_test = db_x[-200:]
db_y_train = daibe.target[:-200]
db_y_test = daibe.target[-200:]

model = linear_model.LinearRegression()
model.fit(db_x_train,db_y_train)
db_y_predict = model.predict(db_x_test)

print("Mean Squred Error : ", mean_squared_error(db_y_test,db_y_predict))
print("Weight : ",model.coef_)
print("Intersept : ",model.intercept_)

plt.scatter(db_x_test,db_y_test)
plt.plot(db_x_test,db_y_predict)
plt.show()
