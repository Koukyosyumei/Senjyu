import numpy as np
from mpi4py import MPI


class Kmeans:
    def __init__(self, k=3, num_iterations=100, seed=42):
        self.k = k
        self.num_iterations = num_iterations
        self.centorids = None
        self.dim = None
        self.n = None

        np.random.seed(seed)

    def train(self, X, parallel=False):
        if parallel:
            pass
        else:
            return self._train_standalone(X)

    def _init_distiution(self, args=None):
        self.args = args
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def _em_standalone(self, X):
        # E-step
        distance = np.zeros((self.k, self.n))
        for cluster_id in range(self.k):
            distance[cluster_id, :] = np.linalg.norm(
                X - self.centorids[cluster_id, :], axis=1
            )
        pred = np.argmin(distance, axis=0)

        # M-step
        for cluster_id in range(self.k):
            self.centorids[cluster_id, :] = np.mean(X[pred == cluster_id, :], axis=0)

        return pred

    def _train_standalone(self, X):
        self.n = X.shape[0]
        self.dim = X.shape[1]
        self.centorids = np.random.normal(0, 1, (self.k, self.dim))

        for _ in range(self.num_iterations):
            pred = self._em_standalone(X)

        return pred
