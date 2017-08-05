# Unsupervised Learning and Dimensionality Reduction

# Analysis
  * Unsupervised Learning and Dimensionality Reduction analysis.pdf
  
# Purpose
  * Explore unsupervised learning algorithms, how the algorithms are the same, different from, and interact with my earlier work.

# Implement six algorithms. The first two are clustering algorithms:
  * K-means clustering
  * Expectation Maximization

The last four algorithms are dimensionality reduction algorithms:

  * PCA
  * ICA
  * Randomized Projections
  * Any other feature selection algorithm you desire

# System Requirements
#### Python 2.7
#### Sklearn
#### matplotlib
#### pylab
#### Kereas
## Data
#### The two files I used to generate my results are named "diabetes Unsupervised" for the diabetes dataset and "spam Unsupervised" for the spambase dataset. For this assignment I decided to get the data straight through the UCI ML database from python except for the neural network tests I did. The neural network tests were only done on the diabetes dataset and I have included the csv for that, you will have to specify where the data set is located in your system and putting that path in the code wherever it was set to be located on my file directory for to run these tests.

## Runing Each Test
#### I decided to use the same naming scheme for each  test in the two python project files I have included, they are the following:
### Clustering
#### •	k-means clustering: kmeans.py

#### •	PCA: PCA.py
#### •	Randomized Projections: rp.py
#### •	Lapacian Score : lapscore.py

#### •	PCA: kmeansPCA.py
#### •	Randomized Projections: kmeansrp.py
#### •	Lapacian Score : kmeansls.py

#### •	PCA: emPCA.py
#### •	Randomized Projections: emrp.py
#### •	Lapacian Score : emls.py
### Neural Network only with Diabetes Dataset (located in "diabetes Unsupervised"" project file)
#### •	Neural Network with clustering: nnClustering.py