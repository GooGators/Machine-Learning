from sklearn import  datasets, metrics, decomposition, mixture
import matplotlib.pyplot as plt
from sklearn import mixture
import numpy as np
import pandas as pd

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    sep=",", header=None)

colnames =  ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                    "order", "mail", "receive", "will", "people", "report", "addresses",
                    "free", "business", "email", "you", "credit", "your", "font", "000",
                    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                    "meeting", "original", "project", "re", "edu", "table", "conference", ";",
             "(", "[", "!", "$", "#", "average", "longest","total", "class" ]

X, y = data.iloc[:, :-1], data.iloc[:, -1]

selected_features1 = X.iloc[:, 3:12]
selected_features2 = X.iloc[:, 45:50]
X = np.concatenate((selected_features1, selected_features2), axis=1)

gm = mixture.GMM(n_components=2, covariance_type='tied')
gm.fit(X)

ypred = gm.predict(X)

both = pd.concat([pd.DataFrame(ypred), pd.DataFrame(y)],1)
both.columns = ['pred', 'class']


from sklearn.metrics import accuracy_score
print "Accuracy"
print accuracy_score(both['class'], both['pred'])
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
comp = []
v_mea = []
sil = []
man = []
numPoints = 14
for i in range(2, numPoints):
    gm = mixture.GMM(n_components=i, covariance_type='diag')
    gm.fit(X)
    y_pred = gm.predict(X)
    homo.append(metrics.homogeneity_score(y, y_pred))
    comp.append(metrics.completeness_score(y, y_pred))
    v_mea.append(metrics.v_measure_score(y, y_pred))
    sil.append(metrics.silhouette_score(X, y_pred,metric='euclidean'))
    man.append(metrics.silhouette_score(X, y_pred, metric='manhattan'))

x = xrange(2, numPoints)
fig = plt.figure()
plt.plot(x, homo, label='homogeneity score')
plt.plot(x, comp, label='completeness score')
plt.plot(x, v_mea, label='v measure score')
plt.plot(x, sil, label='Silhouette Score euclidean')
plt.plot(x, man, label='Silhouette Score manhattan')
plt.legend(loc='middle right', shadow=True)
plt.show()