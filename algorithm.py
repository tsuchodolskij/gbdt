from tree import build_decision_tree
import math


class GBDT:

    def __init__(self, max_iter=100, learning_rate=0.05, tree_depth=6, loss_function='MSE'):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tree_depth = tree_depth
        self.loss = SquareError()
        self.trees = []
        self.F0 = 0

    def fit(self, dataset):
        self.F0 = self.loss.compute_F0(dataset)
        F = {index: self.F0 for index in dataset.indexes()}
        for i in range(0, self.max_iter):
            residual = self.loss.compute_residual(dataset, F)
            tree = [None]
            build_decision_tree(dataset, residual, 0, self.tree_depth, tree)
            self.trees.append(tree[0])
            self.loss.update_F(F, tree[0], dataset, self.learning_rate)

    def predict(self, x):
        F = self.F0
        for tree in self.trees:
            F += self.learning_rate * tree.predict(x)
        return F


class SquareError:
    def compute_F0(self, dataset):
        return self.compute_fit(dataset, dataset.indexes(), None)

    def update_F(self, F, tree, dataset, learning_rate):
        for leaf in tree.leaves:
            indexes = leaf.subset.indexes()  # get the indexes of one region
            gamma = self.compute_fit(dataset, indexes, F)
            leaf.set_gamma(gamma)
            for index in indexes:
                F[index] += gamma * learning_rate

    def compute_residual(self, dataset, F):
        residual = {}
        for index in F:
            residual[index] = dataset[index][dataset.y] - F[index]

        return residual

    def compute_fit(self, dataset, indexes, F):
        ys = [dataset.dataset[index][dataset.y] for index in indexes]
        pos = neg = 0
        for val in ys:
            if val == 1.0:
                pos = pos + 1
            else:
                neg = neg + 1
        if F is None:
            log_odds = math.log(pos/neg, math.e)
            prob = pow(math.e, log_odds) / (1 + pow(math.e, log_odds))
            return prob
        else:
            Fs = [F[index] for index in indexes]
            return (pos - Fs.count(1.0)) / len(ys)
