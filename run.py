import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]

y = [2, 8, 1.8, 8, 0.6, 11]

features = []
for i, j in zip(x,y):
    features.append([i, j])

print(features)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

figure = plt.figure(figsize=(8,6))
plt.scatter(x, y, c = ['r', 'g'])
plt.scatter(centroids[:,0], centroids[:,1], c = ['r', 'g'], marker='X')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('showing two clusters')
plt.show()

print(centroids)
print(labels)


# import ipdb;ipdb.set_trace()
