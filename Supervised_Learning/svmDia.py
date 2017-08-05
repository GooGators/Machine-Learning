import matplotlib.pylab as plt
from sklearn.cross_validation import ShuffleSplit
import seaborn
import pandas
from sklearn.cross_validation import train_test_split
from matplotlib.font_manager import FontProperties
from sklearn import *
import time
from functools import wraps
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve

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

    column_names = ["preg","plas","pres","skin","insu","mass","pedi","age","class"]
    with open('data/pima-indians-diabetes copy.csv') as f:
        data = pandas.read_csv(f, sep=',', names=column_names)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    start_time = time.time()

    title = "Learning Curves rbf(SVM, C=%.6f, gamma=%.6f)" % (2, .0001)
    estimator = SVC(kernel='rbf', C=2, gamma=.0001)
    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    train_sizes, average_train_scores, average_test_scores = plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
    plot = err_plot(train_sizes, average_train_scores, average_test_scores)

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print metrics.classification_report(y_test, y_pred)
    SVM_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("--- %s seconds ---" % (time.time() - start_time))

    conf_matrix = {
        1: {
            'matrix': SVM_matrix ,
            'title': 'SVM-Linear',
        },

    }
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('Confusion Matrix of SVM-Linear')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(3, 3, ii) # starts from 1
        plt.title(title)
        seaborn.heatmap(matrix, annot=True,  fmt='')
    plt.show()
    y_train_pred = estimator.predict(X_train)
    print 'train accuracy: {}'.format(estimator.score(X_train, y_train))
    print 'test accuracy: {}'.format(estimator.score(X_test, y_test))
    print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])
    seaborn.heatmap(metrics.confusion_matrix(y_test, y_pred))
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.show()
if __name__ == '__main__':
    main()