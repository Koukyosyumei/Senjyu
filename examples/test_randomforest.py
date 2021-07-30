import numpy as np
from mpi4py import MPI
from senjyu.ml.trees import RandomForest
from sklearn import datasets


def test_classifer():

    rank = MPI.COMM_WORLD.Get_rank()

    data = datasets.load_iris()

    x = data.data
    y = data.target
    n = x.shape[0]
    s = int(n / 10)

    for f in ["mis_math", "gini", "entropy"]:
        cv_scores = []
        for h in range(1, 11):

            test = list(range(h * s - s + 1, h * s))
            train = list(set(list(range(n))) - set(test))

            rf = RandomForest(f, n_min=10, feature_fraction=0.7, num_trees=6)

            rf.train(x[train], y[train])

            if rank == 0:
                SS = 0
                for t in test:
                    if y[t] == rf.predict(x[t]):
                        SS += 1
                cv_scores.append(SS / len(test))

        if rank == 0:
            print(f, "accuracy is ", np.mean(cv_scores))


if __name__ == "__main__":
    test_classifer()
