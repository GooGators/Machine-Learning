from sklearn.externals.six.moves import zip
from sklearn.ensemble import AdaBoostClassifier
import pylab as pl
import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties
from sklearn import *
import time
from functools import wraps
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit


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

def load_data():
    """ Load Ford alertness data set. Return variables are:
    trainx - features of the training set
    trainy - labels of the training set
    testx - features of the test set
    testy - labels of the test set
    """
    column_names = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                    "order", "mail", "receive", "will", "people", "report", "addresses",
                    "free", "business", "email", "you", "credit", "your", "font", "000",
                    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                    "meeting", "original", "project", "re", "edu", "table", "conference"]


    with open('/data/spambase.csv') as f:
        data = pandas.read_csv(f, sep=',', names=column_names)

        return data
def err_plot(ts, te, ve, title=''):
    """ Make a plot of the training and test error given the data
    resulting from get_learning_curve() or ann_learning_curve().
    Input variables are:
    data - a list of the three outputs from get_learning_curve()
           or ann_learning_curve()
    title - the title of the plot
    """

    te = [1-x for x in te]  # convert to error
    ve = [1-x for x in ve]  # error
    plt.plot(ts, te, 'r-', ts, ve, 'b-')
    plt.ylabel('Error (out of 1)')
    plt.xlabel('Fractional training set size')
    plt.title(title)
    plt.legend(['Training error', 'Test error'], loc='best',
               fancybox=True, shadow=True,
               prop=FontProperties().set_size('small'))
    plt.show()

def get_learning_curve(estimator, trainx, trainy, testx, testy, cv=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """ Returns the learning curve for scikit lassifiers (no neural nets!), i.e.
    training and test accuracies (not error!).
    The input variables are:
    estimator - the scikit classifier to be used (parameters should alread by set)
    trainx - features of the training data
    trainy - labels of the training data
    testx - features of the test data
    testy - labels of the test data
    cv - the number of trainings to be average for more accurate estimates
    train_sizes - list of training size proportions, from (0.0, 1.0]
                   corresponding to 0% to 100% of the full training set set size
    The return variables are:
    train_sizes - list of the training set (proportional) sizes, i.e. x axis
    average_train_scores - the average training accuracy at each training set size
    average_test_scores - the average test accuracy at each training set size
    """
    cv_train_scores = [[0] * len(train_sizes)]
    cv_test_scores = [[0] * len(train_sizes)]
    for c in range(cv):
        train_scores = []
        test_scores = []
        for ts in train_sizes:
            n_examples = int(round(len(trainx) * ts))
            rows = random.sample(range(len(trainx)), n_examples)
            subx = trainx.iloc[rows,]
            suby = trainy.iloc[rows,]
            estimator.fit(subx, suby)
            current_train_score = estimator.score(subx, suby)
            current_test_score = estimator.score(testx, testy)
            train_scores.append(current_train_score)
            test_scores.append(current_test_score)
        cv_train_scores.append(train_scores)
        cv_test_scores.append(test_scores)
    average_train_scores = [sum(i) / cv for i in zip(*cv_train_scores)]
    average_test_scores = [sum(i) / cv for i in zip(*cv_test_scores)]
    return train_sizes, average_train_scores, average_test_scores



@timeit
def test_decision_tree(trainx, trainy, testx, testy):
    """ Train and test a decision tree. max_depth is the max depth
    of the tree that can be grown."""
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    dt = dt.fit(trainx, trainy)
    print 'train accuracy: {}'.format(dt.score(trainx, trainy))
    print 'test accuracy: {}'.format(dt.score(testx, testy))
    return dt



def split(data):
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)
    return x_train, x_test, y_train, y_test, X, y









def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred



def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X.iloc[:, 0].min() - .1, X.iloc[:, 0].max() + .1
    y_min, y_max = X.iloc[:, 1].min() - .1, X.iloc[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    pl.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold)
    pl.show()

def plot_learning_curve(estimator,fitted_dt, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):

    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    return train_sizes, train_scores_mean, test_scores_mean






def main():
    data_percentage_array = np.linspace(0.1, 1, 10)
    full_data = load_data()
    x_train, x_test, y_train, y_test, X, y = split(full_data)
    start_time = time.time()
    cv = ShuffleSplit(x_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

    dt = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=2, max_depth=10,
                                     min_samples_leaf=1, random_state=0)

    fitted_dt = dt.fit(x_train, y_train)
    data_percentage_array = np.linspace(0.1, 1, 10)
    a, average_train_scores, average_test_scores = plot_learning_curve(fitted_dt, x_train, y_train, x_test, y_test,
                                                                       cv=10, train_sizes=data_percentage_array)

    dt = test_decision_tree(x_train, y_train, x_test, y_test)
    dt = dt.fit(x_train,y_train)
    y_pred = dt.predict(x_test)
    y_train_pred = dt.predict(x_train)
    print metrics.classification_report(y_test, y_pred)
    print 'train accuracy: {}'.format(dt.score(x_train, y_train))
    print 'test accuracy: {}'.format(dt.score(x_test, y_test))
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])
    decision_Tree_matrix = metrics.confusion_matrix(y_test, y_pred)
    print decision_Tree_matrix
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()


    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(criterion="entropy", min_samples_split=11, max_depth=19,
                                      min_samples_leaf=14, max_leaf_nodes = 18, random_state=1),
        learning_rate=.3,
        algorithm="SAMME.R", n_estimators=100)
    bdt_discrete.fit(x_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    a, average_train_scores, average_test_scores = plot_learning_curve(bdt_discrete, x_train, y_train, x_test, y_test, cv=10, train_sizes=data_percentage_array)
    plot = err_plot(np.linspace(0.1, 1, 10), average_train_scores, average_test_scores,"Decision Tree AdaBoostClassifier SAMME.R  - Learning Curve")

    print 'train accuracy: {}'.format(bdt_discrete.score(x_train, y_train))
    print 'test accuracy: {}'.format(bdt_discrete.score(x_test, y_test))
    y_pred = bdt_discrete.predict(x_test)
    print metrics.classification_report(y_test, y_pred)

    y_train_pred = bdt_discrete.predict(x_train)  # Let's get the score summary print

    print"test"
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print "train"
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])

    decision_Tree_matrix = metrics.confusion_matrix(y_test, y_pred)
    print decision_Tree_matrix



    column_names = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference"]
    with open('data/spambase.csv') as f:
        data = pandas.read_csv(f, sep=',', names=column_names)

#knn has sensitvity to irrelevent features, after seeing theses results I deceided to look at the feature importance to see if this had a factor
    gbc = ensemble.GradientBoostingClassifier()
    gbc.fit(X, y)
    feature_importance = gbc.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(16, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
    plt.yticks(pos, np.asanyarray(data.columns.tolist())[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == '__main__':
    main()