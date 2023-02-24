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


df = pd.DataFrame()
df['test'] = pd.Series(test_rmse)
df['wtest'] = pd.Series(wtest_rmse)
df = df.set_index(ks)
df[['test','wtest']].plot()
plt.xlabel('K')
plt.ylabel('RMSE')








########### NEW  ########### 2/23 945pm
dropped_linear_rmse = []
selected_linear_rmse = []
dropped_knn_rmse = []
selected_knn_rmse = []
wdropped_knn_rmse = []
wselected_knn_rmse = []
dropped__rmse = []
selected__rmse = []
finalized = X_train
final_test = X_test
dropped_linear = LinearRegression()
sel_linear = LinearRegression()
dropped_knn = KNeighborsRegressor(n_neighbors=7, algorithm='brute')
sel_knn = KNeighborsRegressor(n_neighbors=7, algorithm='brute')
wdropped_knn = KNeighborsRegressor(n_neighbors=13, algorithm='brute', weights='distance')
wsel_knn = KNeighborsRegressor(n_neighbors=13, algorithm='brute', weights='distance')


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
    dropped_linear_rmse.append(g_train_test_rmse(dropped_linear, dropped_train, dropped_test, y_train, y_test))
    selected_linear_rmse.append(g_train_test_rmse(sel_linear, selected_train, selected_test, y_train, y_test))
    #### add any regression down here below
    #### and use selected / dropped on it
    dropped_knn_rmse.append(g_train_test_rmse(dropped_knn, dropped_train, dropped_test, y_train, y_test))
    selected_knn_rmse.append(g_train_test_rmse(sel_knn, selected_train, selected_test, y_train, y_test))
    wdropped_knn_rmse.append(g_train_test_rmse(wdropped_knn, dropped_train, dropped_test, y_train, y_test))
    wselected_knn_rmse.append(g_train_test_rmse(wsel_knn, selected_train, selected_test, y_train, y_test))
dropped_linear_rmse
selected_linear_rmse
dropped_knn_rmse
selected_knn_rmse
wdropped_knn_rmse
wselected_knn_rmse
################################

# Plotting the selected_compared_rmse
plt.plot(selected_linear_rmse)
display_names('Features', 'RMSE', 'Feature versus RMSE')

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz

#TODO 
#Might need a scatterplot to show the splits and the mean between the splits
clf = DecisionTreeRegressor(max_depth=(3), random_state = 0)
clf.fit(X_train, y_train)
dot_data = export_graphviz(clf, precision = 2, feature_names = predictors, proportion = True, class_names = 'quality', filled = True, rounded = True, special_characters = True)
graph = graphviz.Source(dot_data)
display(graph)


########## NEW ########### 2/24 250PM
def selected_features(model, Xdata = X, ydata = y):
    remaining = list(range(Xdata.shape[1]))
    selected = []
    n = 10
    #####
    rmse_previous = 1
    counter = 0
    model.fit(Xdata,ydata)
    ######
    while len(selected) <= n:
        # find the single features that works best in conjunction
        # with the already selected features
        rmse_min = 1e7
        for i in remaining:
            # make a version of the training data with just feature x
            selected.append(i)
            X_curr = Xdata[:,selected]
            selected.remove(i)
            
            # compute negated mean square error scores using 5-fold cross validation
            scores = cross_val_score(model, X_curr, ydata, scoring='neg_mean_squared_error', cv=5)
            
            # work out the average root mean squared error.  We need to
            # first negate the scores, because they are negative MSE, not MSE.
            rmse_curr = np.sqrt(-scores.mean())
            if rmse_curr < rmse_min:
                rmse_min = rmse_curr
                i_min = i
                #print(i_min)
        remaining.remove(i_min)
        selected.append(i_min)
        #############
        rmse_compared = np.sqrt(-scores.mean())
        if (rmse_previous - rmse_compared) > .005:
            counter = counter + 1
        rmse_previous = rmse_compared
        #print('num features: {}; rmse: {:.9f}'.format(len(selected), rmse_min))
    return selected[:counter]
    #return selected ############
###############

########### NEW ############ 2/24 4pm
def get_features(selected_list):
    returned_predictors = []
    for i in selected_list:
        returned_predictors.append(predictors[i])
    return returned_predictors
#################



