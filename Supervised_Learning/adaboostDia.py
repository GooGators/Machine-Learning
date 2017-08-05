import pdb
import numpy as np
from sklearn.externals.six.moves import zip
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
import pylab as pl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import zero_one_loss
import time


def timeit(func):
    """ Decorator function used to time execution.
    """
    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print '%s execution time: %f secs' % (func.__name__, end - start)
        return output
    return timed_function

def split(data):

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    return x_train, x_test, y_train, y_test, X, y

column_names = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]
with open('data/pima-indians-diabetes copy.csv') as f:
    data = pandas.read_csv(f, sep=',', names=column_names)
    X_train, X_test, y_train, y_test, X, y = split(data)
start_time = time.time()


n_features = X_train.shape[1]


# Adaboost classifier using SAMME.R
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(criterion="entropy", min_samples_split=11, max_depth=19,
                                      min_samples_leaf=14, max_leaf_nodes = 18, random_state=1),
    algorithm="SAMME.R",
    learning_rate=1)


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(criterion="entropy", min_samples_split=11, max_depth=19,
                                      min_samples_leaf=14, max_leaf_nodes = 18, random_state=1),
    learning_rate=1,
    algorithm="SAMME")



param_grid = {"n_estimators" : [10], "learning_rate" : [.1, .2, .3, .4]}
grid_search = GridSearchCV(bdt_real, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print 'Best parameters of Adaboost SAMME.R:' , grid_search.best_params_
print 'Best scrore of Adaboost SAMME.R:', grid_search.best_score_


grid_search = GridSearchCV(bdt_discrete, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print 'Best parameters of Adaboost SAMME:' , grid_search.best_params_
print 'Best scrore of Adaboost SAMME:', grid_search.best_score_



num_estimators = X_train.shape[0]

bdt_real.set_params(n_estimators=num_estimators)
bdt_discrete.set_params(n_estimators=num_estimators)

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []


ypred_r = bdt_real.predict(X_test)
ypred_e = bdt_discrete.predict(X_test)
print 'Accuracy of SAMME.R: {} '.format(bdt_real.score(X_test, ypred_r))
print 'Accuracy of SAMME: {}'.format(bdt_discrete.score(X_test, ypred_e))
print("--- %s seconds ---" % (time.time() - start_time))

for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))


n_trees = xrange(1, num_estimators + 1)

pl.figure(figsize=(15, 5))

pl.subplot(131)
pl.plot(xrange(1, len(discrete_test_errors)+1), discrete_test_errors, c='black', label='SAMME')
pl.plot(xrange(1, len(real_test_errors)+1), real_test_errors, c='black',\
        linestyle='dashed', label='SAMME.R')
pl.legend()
pl.ylim(0.18, 0.62)
pl.ylabel('Test Error')
pl.xlabel('Number of Trees')


pl.subplot(132)
pl.plot(n_trees, bdt_discrete.estimator_errors_, "b", label='SAMME', alpha=.5)
pl.plot(n_trees, bdt_real.estimator_errors_, "r", label='SAMME.R', alpha=.5)
pl.legend()
pl.ylabel('Error')
pl.xlabel('Number of Trees')
pl.ylim((.2,
        max(bdt_real.estimator_errors_.max(),
            bdt_discrete.estimator_errors_.max()) * 1.2))
pl.xlim((-20, len(bdt_discrete) + 20))



pl.subplot(133)
pl.plot(n_trees, bdt_discrete.estimator_weights_, "b", label='SAMME')
pl.legend()
pl.ylabel('Weight')
pl.xlabel('Number of Trees')
pl.ylim((0, bdt_discrete.estimator_weights_.max() * 1.2))
pl.xlim((-20, len(bdt_discrete) + 20))


pl.subplots_adjust(wspace=0.25)
pl.show()

fig = pl.figure()
ax = fig.add_subplot(111)

