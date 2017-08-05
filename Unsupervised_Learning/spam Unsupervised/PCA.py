from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import plotly.plotly as py
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

pca = PCA(n_components=57, whiten=True)
pca.fit(X)
plt.figure(figsize=(10,6))
plt.plot(range(1,58), pca.explained_variance_ratio_, 'o-',
        label='Individual variance explained')
plt.plot(range(1,58), np.cumsum(pca.explained_variance_ratio_),'o-',
        label='Cumulative variance explained')
for i in np.linspace(.2,.8,4):
    plt.plot(range(1,58), [i]*57, color='k', alpha=0.2)
plt.title('Variance explained by principal components')
plt.ylabel('Explained variance')
plt.xlabel('# of Principal Components')
plt.legend(loc="best")
plt.show()

X_std = StandardScaler().fit_transform(X)
import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)
print u
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,58)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,58)],
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.plot(fig)