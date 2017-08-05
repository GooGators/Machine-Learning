import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import pdb
from time import time
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import datasets, metrics

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cross_validation import train_test_split

import urllib2




data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
    sep=",", header=None)
colnames =  ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
data.columns = colnames
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X.columns = colnames[:len(colnames)-1]

gm = mixture.GMM(n_components=2, covariance_type='diag')
gm.fit(X)

y_pred = gm.predict(X)

both = pd.concat([pd.DataFrame(y_pred), pd.DataFrame(y)],1)
both.columns = ['pred', 'class']


from sklearn.metrics import accuracy_score
print "Accuracy"
print accuracy_score(both['class'], both['pred'])


#print the
#the number of positive diagnosis when comparred against the labels

for k in range(1,8):
    model = mixture.GMM(n_components=k, covariance_type='diag')
    labels = model.fit_predict(X)

    if k == 2:
        all = np.concatenate((X, np.expand_dims(labels, axis=1), np.expand_dims(y, axis=1)), axis=1)
        all= pd.DataFrame(all)

        for l in range(0, 2):
            print "Clus {}".format(l)
            clus = all.loc[all.iloc[:, -2] == l].iloc[:, -2:]
            print clus.shape[0]
            print float(clus.loc[clus.iloc[:, -1] == 0].shape[0]) / clus.shape[0]
            print float(clus.loc[clus.iloc[:, -1] == 1].shape[0]) / clus.shape[0]

from sklearn.metrics import accuracy_score
print accuracy_score(both['class'], both['pred'])
print both['pred']
print both['class']

homo = []
compl = []
v_m = []
sil = []
man = []
numPoints = 9
for i in range(2, numPoints):
    gm = mixture.GMM(n_components=i, covariance_type='diag')
    gm.fit(X)
    y_pred = gm.predict(X)
    homo.append(metrics.homogeneity_score(y, y_pred))
    compl.append(metrics.completeness_score(y, y_pred))
    v_m.append(metrics.v_measure_score(y, y_pred))
    sil.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
    man.append(metrics.silhouette_score(X, y_pred, metric='manhattan'))

x = xrange(2, numPoints)
fig = plt.figure()
plt.plot(x, homo, label='homogeneity score')
plt.plot(x, compl, label='completeness score')
plt.plot(x, v_m, label='v measure score')
plt.plot(x, sil, label='Silhouette Score euclidean')
plt.plot(x, man, label='Silhouette Score manhattan')
plt.legend(loc='upper right', shadow=True)
plt.show()
