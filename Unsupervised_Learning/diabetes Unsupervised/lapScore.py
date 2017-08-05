from skfeature.utility import construct_W
import matplotlib.pyplot as plt
import numpy as np
import urllib2


# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file

raw_data = urllib2.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,:9]
y = dataset[:,8]


kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
W = construct_W.construct_W(X, **kwargs_W)
from skfeature.function.similarity_based import lap_score
score = lap_score.lap_score(X, W=W)
print score
idx = lap_score.feature_ranking(score)

fig = plt.figure()
plt.plot(score, label='Laplacian Score')

plt.legend(loc='upper middle', shadow=True)
plt.show()
print idx
num_fea = 3

#selected_features = X[:, idx[0:num_fea]]
#print selected_features
#print selected_features
selected_features1 = X[:, 0:1]
selected_features2 = X[:, 5:9]
selected_features = np.concatenate((selected_features1, selected_features2), axis=1)



from skfeature.utility import unsupervised_evaluation
import numpy as np
num_cluster = len(np.unique(y))
print num_cluster

nmi,acc=unsupervised_evaluation.evaluation(X_selected=selected_features,n_clusters=num_cluster,y=y)
print "mutual info score"
print nmi
print "accuracy"
print acc


