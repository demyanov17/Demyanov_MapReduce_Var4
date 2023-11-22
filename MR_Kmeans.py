from mrjob.job import MRJob
import numpy as np
from typing import List, Tuple
from distances import euclidean_distance, cosine_distance, manhattan_distance


class MRKMeans(MRJob):


    def configure_args(self):

        super().configure_args()
        self.add_file_arg('--c')


    def get_distances(self, X: np.array, Y: np.array) -> np.array:
        return (euclidean_distance(X,Y) + cosine_distance(X, Y) + manhattan_distance(X,Y)) /3


    def get_centroids(self) -> List:

        centroids=[]
        with open(self.options.c,'r') as f:
            for line in f:
                x, y, z = line.split('\t')
                centroids.append([float(x), float(y), float(z)])
        return centroids


    def mapper(self, _: None, X_line: str) -> Tuple[int, List]:
        """ Calculate the appropriate class for the current object
        Args:
            X_line(str): string with features
        Returns:
            key(int): cluster_number
            value - X(List): features of the object
        """
        centroids = self.get_centroids()
        x, y, z = X_line.split('\t')
        X = np.asarray([float(x),float(y),float(z)])
        centroids = np.asarray(self.get_centroids())
        cluster_number = np.argmin(self.get_distances(X[np.newaxis, :], centroids))

        yield int(cluster_number), X.tolist()


    def reducer(self, cluster: int, X_val: List) -> Tuple[int, List]:
        """ Recalculation of centroids for the current class
        Args:
            cluster(int): current cluster number
            X_val(List): string with features
        Returns:
            key - cluster(int): current cluster number
            value(List): new centroid for current cluster
        """

        X = np.asarray(list(X_val))
        yield cluster, (X.sum(axis=0) / X.shape[0]).tolist()
    