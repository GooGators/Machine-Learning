import matplotlib.pylab as plt
from sklearn.cross_validation import ShuffleSplit
import pandas
from sklearn.cross_validation import cross_val_score
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


def err_plot(train_sizes, train_scores_mean, test_scores_mean, title=''):
    train_scores_mean = [1-x for x in train_scores_mean]
    test_scores_mean = [1-x for x in test_scores_mean]
    plt.plot(train_sizes, train_scores_mean, 'r-', train_sizes, test_scores_mean, 'b-')
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
    start_time = time.time()
    column_names = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]
    with open('/Users/tyler/machine/data/pima-indians-diabetes copy.csv') as f:
        data = pandas.read_csv(f, sep=',', names=column_names)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    print "Results with 15 Neighbors"


    estimator = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_train_pred = estimator.predict(X_train)

    print("--- %s seconds ---" % (time.time() - start_time))

    title = "Learning Curves (kNN, K=15)"
    plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
    train_sizes, average_train_scores, average_test_scores = plot_learning_curve(estimator, title, X_train, y_train,
                                                                                 cv=cv)
    plot = err_plot(train_sizes, average_train_scores, average_test_scores)
    print 'train accuracy: {}'.format(estimator.score(X_train, y_train))
    print 'test accuracy: {}'.format(estimator.score(X_test, y_test))
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.confusion_matrix(y_test, y_pred)


    start_time = time.time()
    print "Results with 9 Neighbors"
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])

    estimator = KNeighborsClassifier(n_neighbors=9)

    estimator.fit(X_train, y_train)

    title = "Learning Curves (kNN, K=9)"
    plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
    train_sizes, average_train_scores, average_test_scores = plot_learning_curve(estimator, title, X_train, y_train,
                                                                                 cv=cv)
    plot = err_plot(train_sizes, average_train_scores, average_test_scores)

    y_pred = estimator.predict(X_test)
    y_train_pred = estimator.predict(X_train)

    print("--- %s seconds ---" % (time.time() - start_time))
    print "Final Classification Report"
    print metrics.classification_report(y_test, y_pred)
    print 'train accuracy: {}'.format(estimator.score(X_train, y_train))
    print 'test accuracy: {}'.format(estimator.score(X_test, y_test))
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.confusion_matrix(y_test, y_pred)

    knn = KNeighborsClassifier()

    n_neighbors = np.arange(1, 141, 2)

    train_scores = list()
    test_scores = list()
    cv_scores = list()
    for n in n_neighbors:
        knn.n_neighbors = n
        knn.fit(X_train, y_train)
        train_scores.append(
            1 - metrics.accuracy_score(y_train, knn.predict(X_train)))
        test_scores.append(1 - metrics.accuracy_score(y_test, knn.predict(X_test)))
        cv_scores.append(1 - cross_val_score(knn, X_train, y_train, cv=cv).mean())
    print(
        'The best values of k are:\n' \
        '{} according to the Training Set\n' \
        '{} according to the Test Set and\n' \
        '{} according to Cross-Validation'.format(
            min(n_neighbors[train_scores == min(train_scores)]),
            min(n_neighbors[test_scores == min(test_scores)]),
            min(n_neighbors[cv_scores == min(cv_scores)])
        ))

    plt.figure(figsize=(10, 7.5))
    plt.plot(n_neighbors, train_scores, c="black", label="Training Set")
    plt.plot(n_neighbors, test_scores, c="black", linestyle="--", label="Test Set")
    plt.plot(n_neighbors, cv_scores, c="green", label="Cross-Validation")
    plt.xlabel('Number of K Nearest Neighbors')
    plt.ylabel('Classification Error')
    plt.gca().invert_xaxis()
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    main()