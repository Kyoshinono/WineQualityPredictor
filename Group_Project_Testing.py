# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from sklearn.model_selection import train_test_split, cross_val_score
import sys
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor




# switch to seaborn default stylistic parameters
"""
sns.set()
sns.set_context('notebook') 
"""
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")




df.isnull().sum()
df.isna().sum()
#Our data does not contain any null or NAN values, checking for outliers is next



df[df == 0]



df.describe()



zwine = df.apply(zscore)
zwine.describe()


def unit_scaling(x):
    return (x - x.min())/(x.max() - x.min())

############# NEW ############## 2/22 7pm
def display_names(xlabl, ylabl, title):
    plt.xlabel(xlabl)
    plt.ylabel(ylabl)
    plt.title(title)

#######################


wine = unit_scaling(df)


wine.describe()


wine.corr()


sns.heatmap(wine.corr())




predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target = 'quality'
X = df[predictors].values
y = df[target].values




scaler = StandardScaler()
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)



print(X_train.shape)
print(X_train[:3])



regr = KNeighborsRegressor(n_neighbors=7, algorithm='brute')
wregr = KNeighborsRegressor(n_neighbors=7, algorithm='brute', weights='distance')
regr.fit(X_train,y_train)
wregr.fit(X_train,y_train)



y_predict = regr.predict(X_test)
y_w_predict = wregr.predict(X_test)


def rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())



rmse(y_predict, y_test)


rmse(y_w_predict, y_test)



def g_train_test_rmse(reg, X_train, X_test, y_train, y_test):
    reg.fit(X_train,y_train)
    train_predict = reg.predict(X_train)
    test_predict = reg.predict(X_test)
    return rmse(train_predict, y_train), rmse(test_predict, y_test)



n = 30
test_rmse = []
train_rmse = []
wtest_rmse = []
wtrain_rmse = []
ks = np.arange(1, n+1, 2)
for k in ks:
    print(k, ' ', end='')
    regr = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    wregr = KNeighborsRegressor(n_neighbors=k, algorithm='brute', weights='distance')
    rmse_tr, rmse_te = g_train_test_rmse(regr, X_train, X_test, y_train, y_test)
    wrmse_tr, wrmse_te = g_train_test_rmse(wregr, X_train, X_test, y_train, y_test)
    train_rmse.append(rmse_tr)
    test_rmse.append(rmse_te)
    wtrain_rmse.append(wrmse_tr)
    wtest_rmse.append(wrmse_te)
print('done')


def get_best(ks, rmse):
    bes_rm = 100000000
    bes_k = 0
    y = 0
    for k in ks:
        if (rmse[y] < bes_rm):
            bes_rm = rmse[y]
            bes_k = k
            y = y + 1
    return bes_k, bes_rm

best_k, best_rmse = get_best(ks, test_rmse)
wbest_k, wbest_rmse = get_best(ks, wtest_rmse)



plt.plot(test_rmse)
plt.plot(wtest_rmse)
plt.xlabel('K')
plt.ylabel('RMSE')




########### NEW  ########### 2/22 7pm

compared_rmse = []
finalized = X_train
final_test = X_test
picked_reg = regr



for i in range(0, finalized.shape[1]):
    if i == 0:
        dropped_train = finalized[:,1:]
        dropped_test = final_test[:,1:]
    else:
        dropped_train = np.delete(finalized, i, 1)
        dropped_test = np.delete(final_test, i, 1)
    ## get rmse and add to compared_rmse
    compared_rmse.append(g_train_test_rmse(picked_reg, dropped_train, dropped_test, y_train, y_test))
#############FINISHED 7/22 8pm###################
################################





########### NEW  ########### 2/22 815pm
dropped_compared_rmse = []
selected_compared_rmse = []
finalized = X_train
final_test = X_test
picked_reg = regr
sel_reg = LinearRegression()



for i in range(0, finalized.shape[1]):
    if i == 0:
        dropped_train = finalized[:,1:]
        dropped_test = final_test[:,1:]

    else:
        dropped_train = np.delete(finalized, i, 1)
        dropped_test = np.delete(final_test, i, 1)
    selected_train = finalized[:,i].reshape((-1,1))
    selected_test = final_test[:,i].reshape((-1,1))
    ## get rmse and add to compared_rmse
    dropped_compared_rmse.append(g_train_test_rmse(picked_reg, dropped_train, dropped_test, y_train, y_test))
    selected_compared_rmse.append(g_train_test_rmse(sel_reg, selected_train, selected_test, y_train, y_test))
dropped_compared_rmse
selected_compared_rmse
################################

# Plotting the selected_compared_rmse
plt.plot(selected_compared_rmse)
display_names('Features', 'RMSE', 'Feature versus RMSE')








