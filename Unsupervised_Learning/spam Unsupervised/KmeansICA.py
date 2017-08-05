from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import  datasets, metrics, decomposition
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import urllib2


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
# download the file

raw_data = urllib2.urlopen(url)

dataset = np.loadtxt(raw_data, delimiter=",")

X = dataset[:,:57]
y = dataset[:,-1]


compl = []
v_m = []
euc = []
man = []
numPoints = 8
for i in range(2, numPoints):
    ica = decomposition.FastICA(n_components=45)
    output = ica.fit_transform(X)
    km = KMeans(n_clusters=i)
    km.fit(output)
    homo.append(metrics.homogeneity_score(y, km.labels_))
    compl.append(metrics.completeness_score(y, km.labels_))
    v_m.append(metrics.v_measure_score(y, km.labels_))
    euc.append(metrics.silhouette_score(X, km.labels_, metric='euclidean'))
    man.append(metrics.silhouette_score(X, km.labels_, metric='manhattan'))

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
    'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    sep=",", header=None)

colnames =  ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                    "order", "mail", "receive", "will", "people", "report", "addresses",
                    "free", "business", "email", "you", "credit", "your", "font", "000",
                    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                    "meeting", "original", "project", "re", "class" ]
X, y = data.iloc[:, :45], data.iloc[:, -1]
X.columns = colnames[:len(colnames)-1]


ica = decomposition.FastICA(n_components=45, whiten=True)
output = ica.fit_transform(X)
km = KMeans(n_clusters=2, random_state=1)
km_train = km.fit(output)
clus = km_train.predict(X)
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


