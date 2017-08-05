from skfeature.utility import construct_W
import matplotlib.pyplot as plt
import numpy as np
import urllib2

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
# download the file

raw_data = urllib2.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,:57]
#scale = StandardScaler()
#X_std = pd.DataFrame(scale.fit_transform(X))
y = dataset[:,57]


kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
W = construct_W.construct_W(X, **kwargs_W)
from skfeature.function.similarity_based import lap_score
score = lap_score.lap_score(X, W=W)

idx = lap_score.feature_ranking(score)

fig = plt.figure()
plt.plot(score, label='Laplacian Score')

plt.legend(loc='upper middle', shadow=True)
plt.show()



#selected_features1 = X[:, idx[41:49]]
#selected_features2 = X[:, idx[0:14]]

#selected_features = np.concatenate((selected_features2, selected_features1), axis=1)

selected_features1 = X[:, 3:12]
selected_features2 = X[:, 45:50]
selected_features = np.concatenate((selected_features1, selected_features2), axis=1)
#print selected_features
from skfeature.utility import unsupervised_evaluation
import numpy as np
num_cluster = len(np.unique(y))
print num_cluster

nmi,acc=unsupervised_evaluation.evaluation(X_selected=selected_features,n_clusters=num_cluster,y=y)
print "mutual info score"
print nmi
print "accuracy"
print acc


