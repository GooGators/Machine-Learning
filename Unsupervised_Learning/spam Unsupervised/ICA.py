import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import numpy as np
import scipy
import pandas as pd
import plotly.tools as tls
tls.set_credentials_file(username='tbobik1', api_key='2thtg5g46r')

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

data.columns = colnames
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X.columns = colnames[:len(colnames)-1]
#scale = StandardScaler()
#X_std = pd.DataFrame(scale.fit_transform(X))


reducer2 = FastICA(n_components=8)
#reducer2 = reducer2.fit(X)
ica_sources = reducer2.fit_transform(X)
print "kurt"
print scipy.stats.kurtosis(ica_sources)
print "end"

from sklearn import  datasets, metrics, decomposition
kurts = []
for i in range(1, 35):
    ica = decomposition.FastICA(n_components=i, whiten=True)
    output = ica.fit_transform(X)
    kurt = np.average(kurtosis(output))
    kurts.append(kurt)
kurts = pd.Series(kurts, index=pd.Series(range(1, 35)))
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
