import scipy.stats as stats
from mpi4py import MPI
from numpy.random import choice

from decisiontree import DecisionTree, Vertex


class RandomForest(DecisionTree):
    def __init__(
        self,
        criterion,
        alpha=0,
        n_min=10,
        feature_fraction=1.0,
        num_trees=3,
    ):
        super().__init__(criterion, alpha=0, n_min=10, feature_fraction=1.0)
        self.criterion_name = criterion
        self.num_trees = num_trees
        self.forest = []

    def clear(
        self,
        criterion=None,
        alpha=None,
        n_min=None,
        feature_fraction=None,
        num_trees=None,
    ):
        criterion = self.criterion if criterion is None else criterion
        alpha = self.alpha if alpha is None else alpha
        n_min = self.n_min if n_min is None else n_min
        feature_fraction = (
            self.feature_fraction if feature_fraction is None else feature_fraction
        )
        num_trees = self.num_trees if num_trees is None else num_trees

        super().__init__(
            criterion, alpha=alpha, n_min=n_min, feature_fraction=feature_fraction
        )
        self.criterion_name = criterion
        self.num_trees = num_trees
        self.forest = []

    def _init_distiution(self, args=None):
        self.args = args
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def train(self, X, y, parallel=True, process_id=None):
        if parallel:
            self._train_parallel(X, y, process_id=process_id)
        else:
            self._train_standalone(X, y)

    def _train_standalone(self, X, y):
        n = len(y)
        for _ in range(self.num_trees):
            index = choice(list(range(n)), n)
            vertexs = self._train_single_tree(X[index], y[index])
            self.forest.append(vertexs)

    def predict(self, u):
        predicts = []
        for tree in self.forest:
            predicts.append(self._predict_single_tree(u, tree))
        return stats.mode(predicts)[0][0]

    def _train_server(self):
        received_trees = []
        final_num_trees = int(self.num_trees / (self.size - 1)) * (self.size - 1)

        while len(received_trees) < final_num_trees:
            local_forest = self.comm.recv(tag=11)
            for tree in local_forest:
                local_tree = []
                for v in tree:
                    vertex = Vertex()
                    vertex.from_dict(v)
                    local_tree.append(vertex)
                received_trees.append(local_tree)

        self.forest = [tree for tree in received_trees]

    def _train_client(self, X, y, local_num_trees=1):
        local_forest = []
        for _ in range(local_num_trees):
            decision_tree = DecisionTree(
                self.criterion_name,
                alpha=self.alpha,
                n_min=self.n_min,
                feature_fraction=self.feature_fraction,
            )
            tree = decision_tree._train_single_tree(X, y)
            tree_dict = [t.to_dict() for t in tree]
            local_forest.append(tree_dict)

        self.comm.send(local_forest, dest=0, tag=11)

    def _train_parallel(self, X, y, process_id=None):
        self._init_distiution()
        if process_id is None:
            process_id = self.rank

        if process_id == 0:
            self._train_server()
        else:
            self._train_client(
                X, y, local_num_trees=int(self.num_trees / (self.size - 1))
            )
