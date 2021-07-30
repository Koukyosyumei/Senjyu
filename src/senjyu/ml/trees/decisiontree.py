import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.random import choice

from loss import entropy, gini, mis_math, sq_loss

NAME2CRITERION = {
    "sq_loss": sq_loss,
    "gini": gini,
    "entropy": entropy,
    "mis_math": mis_math,
}


class Vertex:
    def __init__(
        self,
        parent=None,
        sets=None,
        score=None,
        th=None,
        j=None,
        center=None,
        right=None,
        left=None,
    ):
        self.parent = parent
        self.sets = sets
        self.score = score
        self.th = th
        self.j = j  # j == 0 if vertex is 端点
        self.right = right
        self.left = left
        self.center = center

    def to_dict(self):
        # parent_dict = self.parent.to_dict() if self.parent is not None else None
        return {
            "parent": self.parent,
            "sets": self.sets,
            "score": self.score,
            "th": self.th,
            "j": self.j,
            "right": self.right,
            "left": self.left,
            "center": self.center,
        }

    def from_dict(self, dict):
        self.parent = dict["parent"]
        self.sets = dict["sets"]
        self.score = dict["score"]
        self.th = dict["th"]
        self.j = dict["j"]
        self.right = dict["right"]
        self.left = dict["left"]
        self.center = dict["center"]


class DecisionTree:
    def __init__(self, criterion, alpha=0, n_min=10, feature_fraction=None):
        self.criterion = NAME2CRITERION[criterion]
        self.alpha = alpha
        self.n_min = n_min

        self.num_features = 0
        self.g = np.mean if criterion == "sq_loss" else lambda x: stats.mode(x)[0][0]
        self.vertexs = []
        self.feature_fraction = feature_fraction

    def train(self, X, y):
        self.vertexs = self._train_single_tree(X, y)

    def _train_single_tree(self, X, y):
        self.num_features = X.shape[1]
        n = len(y)
        vertexs = []
        stack = []
        stack.append(Vertex(parent=-1, sets=list(range(n)), score=self.criterion(y)))
        k = 0

        while len(stack) > 0:
            node = stack.pop()
            best_branch = self.branch(X, y, node.sets)

            if (
                ((node.score - best_branch["score"]) < self.alpha)
                or len(node.sets) < self.n_min
                or len(best_branch["left"]) == 0
                or len(best_branch["right"]) == 0
            ):
                vertexs.append(Vertex(parent=node.parent, j=-1, sets=node.sets))
            else:
                vertexs.append(
                    Vertex(
                        parent=node.parent,
                        sets=node.sets,
                        th=X[best_branch["i"], best_branch["j"]],
                        j=best_branch["j"],
                    )
                )
                stack.append(
                    Vertex(
                        parent=k,
                        sets=best_branch["right"],
                        score=best_branch["right_score"],
                    )
                )
                stack.append(
                    Vertex(
                        parent=k,
                        sets=best_branch["left"],
                        score=best_branch["left_score"],
                    )
                )
            k += 1

        r = len(vertexs)

        for h in range(r):
            vertexs[h].left = None
            vertexs[h].right = None

        for h in range(r - 1, 0, -1):  # rから2まで
            if vertexs[vertexs[h].parent].right is None:
                vertexs[vertexs[h].parent].right = h
            else:
                vertexs[vertexs[h].parent].left = h

        # 端点に、その値(center)を設定
        for h in range(r):
            temp = vertexs[h]
            if temp.j == -1:
                temp.center = self.g(y[temp.sets])
                vertexs[h] = temp

        return vertexs

    def branch(self, X, y, idxs):
        num_idxs = len(idxs)
        best_score = float("inf")
        result = None

        if num_idxs == 0:
            return result

        if self.feature_fraction is None or self.feature_fraction >= self.num_features:
            feature_idxs = range(self.num_features)
        else:
            feature_idxs = choice(
                list(range(self.num_features)),
                int(self.num_features * self.feature_fraction),
            )

        for j in feature_idxs:
            for i in idxs:
                left, right = [], []
                for k in idxs:
                    if X[k, j] < X[i, j]:
                        left.append(k)
                    else:
                        right.append(k)

                L = self.criterion(y[left])
                R = self.criterion(y[right])
                score = L + R
                if score < best_score:
                    best_score = score
                    result = {
                        "i": i,
                        "j": j,
                        "left": left,
                        "right": right,
                        "score": best_score,
                        "left_score": L,
                        "right_score": R,
                    }

        return result

    def _predict_single_tree(self, u, vertexs):
        r = 0
        while vertexs[r].j != -1:
            if u[vertexs[r].j] < vertexs[r].th:
                r = vertexs[r].left
            else:
                r = vertexs[r].right
        return vertexs[r].center

    def predict(self, u):
        return self._predict_single_tree(u, self.vertexs)

    def get_threshold(self):
        r = len(self.vertexs)
        VAR = []
        TH = []
        for h in range(r):
            if self.vertexs[h].j != 0:
                j = self.vertexs[h].j
                th = self.vertexs[h].th
                VAR.append(j)
                TH.append(th)

        return pd.DataFrame(TH, VAR, columns=["threshold"])
