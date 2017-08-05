from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.cross_validation import train_test_split, ShuffleSplit
import pandas
import keras
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import Callback
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
def main():
    start_time = time.time()
    seed = 7
    numpy.random.seed(seed)

    column_names = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]
    with open('data/pima-indians-diabetes copy.csv') as f:
        data = pandas.read_csv(f, sep=',', names=column_names)


    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()
    y_train = y_train.as_matrix()
    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

    model = Sequential()
    model.add(Dense(15, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(6, init='uniform', activation='linear'))
    model.add(Dense(4, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10000, batch_size=20, verbose=0)

    print("--- %s seconds ---" % (time.time() - start_time))

    print(history.history.keys())

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    print("Train acc", history.history["acc"])



if __name__ == '__main__':
    main()
