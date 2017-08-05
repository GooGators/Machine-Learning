import matplotlib.pylab as plt
from sklearn.cross_validation import ShuffleSplit
import pandas
from matplotlib.font_manager import FontProperties
from sklearn import *
import time
from functools import wraps
from sklearn.learning_curve import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np


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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return train_sizes, train_scores_mean, test_scores_mean




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


def main():
    with open('data/spambase.csv') as f:
        data = pandas.read_csv(f, sep=',')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)
    start_time = time.time()


    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    print "Results with 15 Neighbors"


    estimator = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)

    start_time = time.time()
    print "Results with 1 Neighbors"


    estimator = KNeighborsClassifier(n_neighbors=1)

    estimator.fit(X_train, y_train)

    # Let's look at the best estimator that was found by GridSearchCV

    title = "Learning Curves (kNN, K=1)"
    y_pred = estimator.predict(X_test)
    y_train_pred = estimator.predict(X_train)

    print("--- %s seconds ---" % (time.time() - start_time))
    print "Final Classification Report"
    print metrics.classification_report(y_test, y_pred)
    print 'train accuracy: {}'.format(estimator.score(X_train, y_train))
    print 'test accuracy: {}'.format(estimator.score(X_test, y_test))
    print metrics.classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['Not Spam', 'Spam'])
    print metrics.confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    main()