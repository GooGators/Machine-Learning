from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy.random as npr
import numpy as np
import pandas as pd

import urllib2
def split_data(data, train_split=0.7):
    data = np.array(data)
    num_train = data.shape[0] * train_split
    npr.shuffle(data)

    return (data[:num_train], data[num_train:])

# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file

raw_data = urllib2.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,:8]

y = dataset[:,8]


homo = []
compl = []
v_m = []
euc = []
man = []
numPoints = 7
for i in range(2, numPoints):
    pca = PCA(n_components = 6)
    fitted = pca.fit_transform(X)
    pca.fit(X)
    km = KMeans(n_clusters=i)
    x_fit = km.fit(fitted_data_pca)
    homo.append(metrics.homogeneity_score(y, km.labels_))
    compl.append(metrics.completeness_score(y, km.labels_))
    v_m.append(metrics.v_measure_score(y, km.labels_))
    euc.append(metrics.silhouette_score(fitted, km.labels_, metric='euclidean'))
    man.append(metrics.silhouette_score(fitted, km.labels_, metric='manhattan'))

x = xrange(2, numPoints)
fig = plt.figure()
plt.plot(x, homo, label='Homogeneity Score')
plt.plot(x, compl, label='Completeness Score')
plt.plot(x, v_m, label='V-measure Score')
plt.plot(x, euc, label='Silhouette Score euclidean')
plt.plot(x, man, label='Silhouette Score manhattan')
plt.legend(loc='upper center', shadow=True)
plt.show()


data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
    sep=",", header=None)
colnames =  ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'class']

X, y = data.iloc[:, :6], data.iloc[:, -1]

X.columns = colnames[:len(colnames)-1]


pca = PCA(n_components = 6)
fittedpca = pca.fit_transform(X)
pca.fit(X)
km = KMeans(n_clusters=2, random_state=1)
km_train = km.fit(fittedpca)
clus = km_train.predict(fittedpca)
pd.set_option('display.float_format', lambda l: '%.3f' % l)
columns = {str(l): km_train.cluster_centers_[l] for l in range(0,len(km_train.cluster_centers_))}
clusters = pd.DataFrame(columns, index=X.columns)
clusters.index = X.columns
print clusters
#print y_train

df = pd.DataFrame(columns, index=X.columns).copy()
df.columns=['1','2']
labelss = pd.DataFrame(list(zip(clus, y)), columns=['clus','class'])
labelss['clus'] = labelss.clus.map({i:col for i, col in enumerate(df.columns)})
the = (labelss.groupby(['clus']).mean()*100)
print "true vs predicted"
print the

y_pred = clus
print(metrics.classification_report(y, clus))
from sklearn.metrics import mean_squared_error

print(metrics.confusion_matrix(y, clus))


test_results = pd.concat([pd.DataFrame(clus), pd.DataFrame(y)],1)
#print test_results
test_results.columns = ['pred', 'class']

print "mse"
print mean_squared_error(test_results['pred'], test_results['class'])
print accuracy_score(test_results['class'], test_results['pred'])

