import numpy as np 
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import pandas as pd 
import matplotlib.pyplot as plt 

train_data = pd.read_csv('train_data.csv', header=0)
test_data = pd.read_csv('test_data.csv', header=0)
# print train_data.dtypes

# dataX = train_data.ix[:, train_data.columns != 'shares']
# dataY = train_data.ix[:, train_data.columns == 'shares']

# dataX = pd.DataFrame(train_data,columns = ['shares'])
# dataX = train_data.drop('shares',axis = 1)

dataX = train_data.drop(['url',' shares'], axis = 1)
dataY = train_data[' shares']

testdataX = test_data.drop(['url'], axis = 1)

# print dataX.head()
# print dataY.head()

# Create linear regression object
regr = LinearRegression()

regr.fit(dataX,dataY)
# print dataY[:10]
# print regr.predict(dataX[:10])

# trainPredictions = regr.predict(dataX).astype(int)
# # print trainPredictions[:10]
# err = abs(trainPredictions - dataY)		#to err is human
# # print err[:10]
# total_error = np.dot(err,err)				#to err,err is pirate
# rmse_train = np.sqrt(total_error/ len(err))	#RMSE on training data
# print "lin reg error : ", rmse_train

# print regr.coef_
# linTestPredictions = regr.predict(testdataX).astype(int)

# linDF = pd.DataFrame(data = linTestPredictions[:])
# linDF.columns = ['shares']
# linDF.to_csv('outputLin.csv')
# print linTestPredictions[:10]
# linTestPredictions.to_csv('outputLin.csv')

# # Create ridge regression object
ridge = Ridge(fit_intercept=True, alpha=0.418)

ridge.fit(dataX,dataY)
ridgeTestPredictions = ridge.predict(testdataX).astype(int)

ridgeDF = pd.DataFrame(data = ridgeTestPredictions[:])
ridgeDF.columns = ['shares']
ridgeDF.to_csv('outputridge0_418.csv')


# print ridge.coef_
# # print dataY[:10]
# # print ridge.predict(dataX[:10])

# trainPredictions = ridge.predict(dataX).astype(int)
# # print trainPredictions[:10]
# err = abs(trainPredictions - dataY)		#to err is human
# # print err[:10]
# total_error = np.dot(err,err)				#to err,err is pirate
# rmse_train = np.sqrt(total_error/ len(err))	#RMSE on training data
# print "ridge reg error : ", rmse_train

# alpha = np.linspace(.01,20,50)
# t_rmse = np.array([])
# for a in alpha:
#     ridge = Ridge(fit_intercept=True, alpha=a)
#     ridge.fit(dataX,dataY)
#     p = ridge.predict(dataX)
#     err = p-dataY
#     total_error = np.dot(err,err)
#     rmse_train = np.sqrt(total_error/len(p))
#     t_rmse = np.append(t_rmse, [rmse_train])
#     print('ridge reg: alpha: {:.3f}\t error : {:.4f}'.format(a,rmse_train))

# Now let's compute RMSE using 10-fold x-validation
# kf = KFold(len(dataX), n_folds=5)
# xval_err = 0
# # print len(dataX)
# for train,test in kf:
#     # print train, test
#     ridge.fit(dataX[train],dataY[train])
#     # p = np.array([regr.predict(xi) for xi in x[test]])
#     p = ridge.predict(dataX[test])
#     e = p-dataY[test]

#     xval_err += np.dot(e,e)
    
# rmse_10cv = np.sqrt(xval_err/len(dataX))
# print rmse_10cv

