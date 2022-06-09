# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:14:50 2022

@author: Sai pranay
"""


import pandas as pd
import numpy as np
SU = pd.read_csv("E:\\DATA_SCIENCE_ASS\\MULTI_LINEAR_REGRESSION\\50_Startups.csv")
SU
SU.shape
list(SU)
SU.dtypes
SU.head()
SU.describe()
SU.info()
SU. isnull().sum()

SU.corr()


# scatter plot
SU.plot.scatter(x='R&D Spend',y='Profit')
SU.plot.scatter(x='Marketing Spend',y='Profit')
SU.plot.scatter(x='Administration',y='Profit')


------# FIRST HIGHEST CORR POINT


X = SU['R&D Spend']
X
X.ndim
# CHANGING DIMENSIONS

X =X[:,np.newaxis]
X.ndim

Y = SU['Profit']
Y

 # Import Linear Regression
 from sklearn.linear_model import LinearRegression
 model = LinearRegression().fit(X, Y)
 model.intercept_  ## To check the Bo values
 model.coef_       ## To check the coefficients (B1)

Y_Pred = model.predict(X)
Y_Pred


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred)*100
print("R square: ", r2.round(3))


-----------# SECOND HIGHEST CORR

X1 = SU[['R&D Spend','Marketing Spend']]
X1
X1.ndim


Y = df['Profit']
Y

 # Import Linear Regression
 from sklearn.linear_model import LinearRegression
 model = LinearRegression().fit(X1, Y)
 model.intercept_
 model.coef_

Y_Pred2 = model.predict(X1)
Y_Pred2


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred2)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred2)*100
print("R square: ", r2.round(3))

------------# THIRD HIGHEST CORR

X3 = SU[['R&D Spend','Marketing Spend','Administration']]
X3
X3.ndim


Y = df['Profit']
Y

# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X2, Y)
model.intercept_
model.coef_

Y_Pred3 = model.predict(X2)
Y_Pred3


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred3)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred3)*100
print("R square: ", r2.round(3))


------------------------

X4 = SU[['R&D Spend','Administration']]
X4
X4.ndim


Y = df['Profit']
Y

 # Import Linear Regression
 from sklearn.linear_model import LinearRegression
 model = LinearRegression().fit(X4, Y)
 model.intercept_
 model.coef_

Y_Pred4 = model.predict(X4)
Y_Pred4


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred4)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred4)*100
print("R square: ", r2.round(3))

