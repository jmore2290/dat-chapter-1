ttribution: This code was obtained from
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
# All I have done is separate the code
# and add some clarifying commentary

import numpy as np
from sklearn.datasets import load_digits

# Set a random seed
np.random.seed(42)

# load the data
digits = load_digits()

# now let's use this scale utility to scale the features, 
# since that's really important in clustering 
from sklearn.preprocessing import scale
feature_data = scale(digits.data)

# let's see how many unique targets we have
n_digits = len(np.unique(digits.target))
target_data = digits.target


--

n_digits
digits.data

--

from sklearn import metrics
from time import time

# Since we have the labels for this dataset, we can actually
# benchmark how well the clustering separates the data
def bench_k_means(estimator, name, data):
    t0 = time()
    # run a certain configuration k means 
    # e.g. random_state, number of clusters)
    # on the data
    estimator.fit(data)
    # print out how it matched up against the actual class labels
    print('Name: % 9s \nDuration: %.2fs \nEstimator inertia: %i \nHomogeneity: %.3f \nCompleteness: %.3f \nV-measure %.3f \nAdjusted Rand Score: %.3f \nMutual Info: %.3f \nSilhouette:  %.3f'
            
            # how long it took run k means
          % (name, (time() - t0), estimator.inertia_,
            # how homogeneous the clusters are
            # a score of 1 means that all of the points in each cluster
            # belong to the same class, while a score of 0 means
            # that none of the points in each cluster belong to the same
            # class
            # more documentation here: 
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
             metrics.homogeneity_score(target_data, estimator.labels_),
            # completeness is 1 if all members of class A are in the same
            # cluster, documentation here:
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
             metrics.completeness_score(target_data, estimator.labels_),
            # the harmonic mean of homogeneity and completeness
             metrics.v_measure_score(target_data, estimator.labels_),
            # documentation here: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
             metrics.adjusted_rand_score(target_data, estimator.labels_),
            # documentation here: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
             metrics.adjusted_mutual_info_score(target_data,  estimator.labels_),
            # documentation here: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))
#             All this code is to evalutate how good the clustering is.  
# Adjusted Rand score goes between -1 and 1 

--

# Now let's run some different configurations of K-means
from sklearn.cluster import KMeans

# let's try k means where we initialize the centers randomly
k_means_random = KMeans(init='random', n_clusters=14, n_init=10)
bench_k_means(k_means_random, name="k-means random", data=feature_data)
# Estimator  - is W
# Homogenerity - how homogeneous are are clusters, how much of cluster contains the same class
# Completeness - all things in the class are now inthe same cluster, kinda opposite of homogeneity.  Though there is no direct correlation 
# between the two. 

--

# now let's try k means where we smartly initialize the centers
k_means_smart = KMeans(init='k-means++', n_clusters=14, n_init=10)
bench_k_means(k_means_smart, name="k-means smart", data=feature_data)

--

# Now we're going to plot the data, but first we'll
# need to reduce the dimensionality from 700+ to 2
from sklearn.decomposition import PCA

reduced_data = PCA(n_components=2).fit_transform(feature_data)

--

#  then we'll fit K-Means to this 2-dimensional transformed data
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

--

import numpy as np
import matplotlib.pyplot as plt

# Now some magic visualization code, which you'll remember
# from Sonia's lecture.

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

--


