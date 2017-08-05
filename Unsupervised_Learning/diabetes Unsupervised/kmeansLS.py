from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import urllib2

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file

raw_data = urllib2.urlopen(url)

dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,:4]

y = dataset[:,8]

homo = []
compl = []
v_m = []
euc = []
man = []
numPoints = 8
for i in range(2, numPoints):
    km = KMeans(n_clusters=i)
    km.fit(X)
    homo.append(metrics.homogeneity_score(y, km.labels_))
    compl.append(metrics.completeness_score(y, km.labels_))
    v_m.append(metrics.v_measure_score(y, km.labels_))
    euc.append(metrics.silhouette_score(X , km.labels_, metric='euclidean'))
    man.append(metrics.silhouette_score(X , km.labels_, metric='manhattan'))

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
colnames =  ['preg', 'plas', 'pres', 'skin', 'class']

X, y = data.iloc[:, :4], data.iloc[:, -1]
X.columns = colnames[:len(colnames)-1]


km = KMeans(n_clusters=2, random_state=1)
km_train = km.fit(X)
labels2 = km_train.predict(X)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
columns = {str(x): km_train.cluster_centers_[x] for x in range(0,len(km_train.cluster_centers_))}

print columns
#print y_train

dfclus = pd.DataFrame(columns).copy()
dfclus.columns=['1','2']
clussdf = pd.DataFrame(list(zip(labels2, y)), columns=['clus','class'])
clussdf['clus'] = dfcl.clus.map({i:col for i, col in enumerate(dfclus.columns)})
the = (clussdf.groupby(['clus']).mean()*100)
print "acccc"
print the

ypred = labels2
print(metrics.classification_report(y, ypred))
from sklearn.metrics import mean_squared_error

print(metrics.confusion_matrix(y, ypred))


test_results = pd.concat([pd.DataFrame(ypred), pd.DataFrame(y)],1)
#print test_results
test_results.columns = ['pred', 'class']

print "mse"
print mean_squared_error(test_results['pred'], test_results['class'])



