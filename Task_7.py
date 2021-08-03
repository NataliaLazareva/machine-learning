#Реализация алгоритма кластеризации k-means.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from random import choice
import numpy as np

def init_data():
 centers = [[-1, -1], [0, 1], [1, -1]]
 X, y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.5)
 return X, y

def init_centroid(x, k):

 centroids = {
     i + 1: [choice(x[:,0]), choice(x[:,1])] for i in range(k)}

 return centroids

def GetClusters(centroids, K, X):

 Clusters = {}
 FunctionE = 0

 for ind in range(1000):
        ClusterNum = 0
        maximum = 1000.0
        vector = []
        elem = X[ind]

        for i in range(K):
             norma = np.linalg.norm(elem - np.array(centroids[i+1]))
             if maximum > norma:
                ClusterNum = i+1
                maximum = norma
                vector = elem

        try:
            Clusters[ClusterNum].append(vector)
        except KeyError:
            Clusters[ClusterNum] = [vector]

        oldCentroid = centroids

        for cluster in range(1, K+1):
            if (cluster in Clusters):
                centroids[ClusterNum] = CentroidsRecalculation(Clusters[ClusterNum])
            else:
                 centroids[cluster] = X[np.random.randint(3000, size=1)]


        FunctionE += maximum

        if(oldCentroid == centroids):
            return Clusters, FunctionE

 return Clusters, FunctionE

def CentroidsRecalculation(vectors):
       vectors = np.array(vectors)
       NewCentroid = np.sum(vectors, axis=0)/len(vectors)
       return (NewCentroid)


def visualization(DList, TargetList, Clusters):
    a = [i for i in range(1, len(TargetList) + 1)]
    b = [i for i in range(1, len(DList) + 1)]
    fig, ax = plt.subplots()
    for key in Clusters:
      x = np.array(Clusters[key])
      print(x)
      ax.scatter(x[:,0], x[:,1], c='r')

    fig.set_figwidth(5)  # ширина и
    fig.set_figheight(5)  # высота "Figure"

    plt.figure()
    plt.plot(TargetList, a, 'b.-')
    plt.plot(DList, b, 'r.-')

    plt.show()

def main():
 #количество кластеров
 X, y = init_data()
 TargetList = []
 DList = []

 for i in range(1, 11):
   centroids = init_centroid(X, i)
   Clusters, TargetValue = GetClusters(centroids, i, X)
   TargetList.append(TargetValue)


 for i in range(1,9):
     D = abs((TargetList[i] - TargetList[i+1])/(TargetList[i-1] - TargetList[i]))
     DList.append(D)

 centroids = init_centroid(X, np.argmin(DList)+2)
 Clusters, TargetValue = GetClusters(centroids, np.argmin(DList)+2, X)

 visualization(DList, TargetList, Clusters)

if __name__ == '__main__':
    main()

