import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
import scipy
import plotly.tools as tls

tls.set_credentials_file(username='tbobik1', api_key='2thtg5g46r')

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
    sep=",", header=None)
colnames =  ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
data.columns = colnames
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X.columns = colnames[:len(colnames)-1]


reducer2 = FastICA(n_components=8)
ica_sources = reducer2.fit_transform(X)
print "kurt"
print scipy.stats.kurtosis(ica_sources)
print "end"

from sklearn import  datasets, metrics, decomposition
kurts = []
for i in range(1, 9):
    ica = decomposition.FastICA(n_components=i, whiten=True)
    output = ica.fit_transform(X)
    kurt = np.average(kurtosis(output))
    kurts.append(kurt)
kurts = pd.Series(kurts, index=pd.Series(range(1, 9)))
"""
Plot Kurtosis for ICA
"""
kurts.plot()
plt.xlabel('Dimension')
plt.ylabel('Average Kurtosis')
plt.title('Average Kurtosis vs Dimension')
plt.show()
print "KURT Final"
print kurts
