from sklearn.random_projection import johnson_lindenstrauss_min_dim
import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
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

print johnson_lindenstrauss_min_dim(4601,eps=0.1)


split = train_test_split(X, y, test_size = 0.3,
    random_state = 42)
(trainData, testData, trainTarget, testTarget) = split
accuracies = []
components = np.int32(np.linspace(2, 56, 14))
model = LinearSVC()
model.fit(trainData, trainTarget)
baseline = metrics.accuracy_score(model.predict(testData), testTarget)
# loop over the projection sizes
for comp in components:
    # create the random projection
    sp = SparseRandomProjection(n_components=comp)
    X = sp.fit_transform(trainData)

    # train a classifier on the sparse random projection
    model = LinearSVC()
    model.fit(X, trainTarget)

    # evaluate the model and update the list of accuracies
    test = sp.transform(testData)
    accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

# create the figure
plt.figure()
plt.suptitle("Accuracy of Sparse Projection on Spam")
plt.xlabel("# of Components")
plt.ylabel("Accuracy")
plt.xlim([2, 56])
plt.ylim([0, 1.0])

# plot the baseline and random projection accuracies
plt.plot(components, [baseline] * len(accuracies), color="r")
plt.plot(components, accuracies)

plt.show()