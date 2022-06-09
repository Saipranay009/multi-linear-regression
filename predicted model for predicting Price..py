# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:05:27 2022

@author: Sai pranay
"""

#-----------------------IMPORTING_THE_DATA_SET---------------------------------


import pandas as pd

tc = pd.read_csv("E:\\DATA_SCIENCE_ASS\\MULTI_LINEAR_REGRESSION\\ToyotaCorolla.csv",encoding='latin1')
print(tc)
tc.shape
list(tc)
tc.dtypes
tc.head()
tc.describe()
tc.info()
tc. isnull().sum()


#--------------------checking_correlation--------------------------------------
tc.corr().Price

#------------------------------scatter plot------------------------------------
tc.plot.scatter(x= 'Age_08_04',y='Price')
tc.plot.scatter(x= 'KM',y='Price')
tc.plot.scatter(x= 'HP',y='Price')
tc.plot.scatter(x= 'cc',y='Price')
tc.plot.scatter(x= 'Doors',y='Price')
tc.plot.scatter(x= 'Gears',y='Price')
tc.plot.scatter(x= 'Quarterly_Tax',y='Price')
tc.plot.scatter(x= 'Weight',y='Price')



#-----------------------FIRST_HIGHEST_CORR-------------------------------------

# ================split the variables as X and Y===============================
Y = tc["Price"]
Y
Y.shape
Y.ndim


X = tc['Age_08_04']
X
X.ndim
X.shape

#-------------------------changing_the_dimension-------------------------------
import numpy as np
X = X[:,np.newaxis]
X.ndim



#---------------Import_Linear_Regression---------------------------------------

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

#---------Make_predictions_using_independent_variable_values-------------------
Y_Pred = model.predict(X)

#-------------------------importing_the_mean_squared_error_--------------------


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred)
print("Mean square error: ", (mse).round(3))

#-------------------------importing_the_r__squared_error_--------------------


from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred)*100
print("R square: ", r2.round(3))

'''
X
X.shape
X.shape[0]
X.shape[1]
'''

n = X.shape[0]
k = X.shape[1] + 1
ssres = np.sum((Y - Y_Pred)**2  )
sstot = np.sum((Y - np.mean(Y))**2  )

num = ssres/(n-k)
den = sstot/(n-1)

r2_adj =  1  - (num/den)
print("Adjusted Rsquare: ", (r2_adj*100).round(3))




#---------------------------------SECOND_HIGHEST_CORR--------------------------

Y = tc["Price"]
Y
Y.shape
Y.ndim


X1 = tc[['Age_08_04','Weight']]
X1
X1.ndim
X1.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X1, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred1 = model.predict(X1)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred1)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred1)*100
print("R square: ", r2.round(3))


#------------------------------THIRD_HIGH_CORR---------------------------------


Y = tc["Price"]
Y
Y.shape
Y.ndim


X2 = tc[['Age_08_04','Weight','KM']]
X2
X2.ndim
X2.shape



# Import Linear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X2, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values

Y_Pred2 = model.predict(X2)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred2)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred2)*100
print("R square: ", r2.round(3))


#-----------------------------forth highest corr------------------------------

Y = tc["Price"]
Y
Y.shape
Y.ndim


X3 = tc[['Age_08_04','Weight','KM','HP']]
X3
X3.ndim
X3.shape



# Import Linear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X3, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values

Y_Pred3 = model.predict(X3)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred3)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred3)*100
print("R square: ", r2.round(3))


#-------------------------fifth highest corr-----------------------------------


Y = tc["Price"]
Y
Y.shape
Y.ndim


X4 = tc[['Age_08_04','Weight','KM','HP','Quarterly_Tax']]
X4
X4.ndim
X4.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X4, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred4 = model.predict(X4)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred4)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred4)*100
print("R square: ", r2.round(3))


#-------------------------sixth highest corr-----------------------------------


Y = tc["Price"]
Y
Y.shape
Y.ndim


X5 = tc[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors']]
X5
X5.ndim
X5.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X5, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred5 = model.predict(X5)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred5)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred5)*100
print("R square: ", r2.round(3))

#-------------------------seventh highest corr---------------------------------


Y = tc["Price"]
Y
Y.shape
Y.ndim


X6 = tc[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc']]
X6
X6.ndim
X6.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X6, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred6 = model.predict(X6)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred6)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred6)*100
print("R square: ", r2.round(3))




#---------------------eigth highest corr-------------------------------------



Y = tc["Price"]
Y
Y.shape
Y.ndim


X7 = tc[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc','Gears']]
X7
X7.ndim
X7.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X7, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred7 = model.predict(X7)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred7)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred7)*100
print("R square: ", r2.round(3))


#--------------------best model------------------------------------------------



Y = tc["Price"]
Y
Y.shape
Y.ndim


X8 = tc[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Gears']]
X8
X8.ndim
X8.shape



# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X8, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred8 = model.predict(X8)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred8)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred8)*100
print("R square: ", r2.round(3))
