import numpy as np 
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import pandas as pd 
import matplotlib.pyplot as plt 

train_data = pd.read_csv('train_data.csv', header=0)
test_data = pd.read_csv('test_data.csv', header=0)

dataX = train_data.drop(['url',' shares'], axis = 1)
dataY = train_data[' shares']

testdataX = test_data.drop(['url'], axis = 1)

# Create linear regression object
regr = LinearRegression()

regr.fit(dataX,dataY)

regr.fit(dataX,dataY)
regrTestPredictions = regr.predict(testdataX).astype(int)

regrDF = pd.DataFrame(data = regrTestPredictions[:])
regrDF.columns = ['shares']
regrDF.to_csv('outputLinear.csv')