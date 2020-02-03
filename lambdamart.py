from collections import defaultdict

import numpy as np

from tqdm import tqdm
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor


class LambdaMART:

    def __init__(self, n_trees, **kwargs):
        self.n_trees = n_trees
        self.params = kwargs
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.sigma = kwargs.get('sigma', 1)
        self.top_k = kwargs.get('top_k', None)
        self.trees = []
        self.train_loss = []
        self.val_loss = []

    @staticmethod
    def _compute_dcg(relevance, k=None):
        k = len(relevance) if k is None else k
        dcg = np.sum(((np.power(2, relevance) - 1) / np.log2(2 + np.arange(len(relevance))))[:k])
        return dcg if dcg > 0 else 1

    def _compute_val_score(self, y_true, y_pred):
        loss = 0
        idcg = 0
        for query_id, query_indexes in self.val_query_groups.items():
            query_y_true = y_true[query_indexes]
            query_y_pred = y_pred[query_indexes]

            loss += self._compute_dcg(query_y_true[np.argsort(-query_y_pred)], self.top_k)
            idcg += self._compute_dcg(sorted(query_y_true, reverse=True), self.top_k)

        self.val_loss.append(loss / idcg)

    def _preprocess(self, y, query_ids, is_val=False):
        query_groups = defaultdict(list)
        for query_index, query_id in enumerate(query_ids):
            query_groups[query_id].append(query_index)

        query_idcg = dict()
        query_permutations = dict()
        for query_id, query_indexes in query_groups.items():
            idcg = self._compute_dcg(sorted(y[query_indexes], reverse=True))
            query_idcg[query_id] = idcg
            query_permutations[query_id] = np.tile(np.arange(len(query_indexes)), (len(query_indexes), 1))

        if not is_val:
            self.query_groups = query_groups
            self.query_idcg = query_idcg
            self.query_permutations = query_permutations
        else:
            self.val_query_groups = query_groups
            self.val_query_idcg = query_idcg
            self.val_query_permutations = query_permutations

    def _compute_lambdas(self, y_true, y_pred):
        lambdas = np.empty_like(y_true)

        loss = 0
        idcg = 0
        hess = np.empty(len(y_true))
        for query_id, query_indexes in self.query_groups.items():
            query_y_true = y_true[query_indexes]
            query_y_pred = y_pred[query_indexes]
            i_j = self.query_permutations[query_id]
            i_j_preds = query_y_pred[i_j]
            i_j_true = query_y_true[i_j]

            document_positions = np.empty_like(query_indexes)
            document_positions[np.argsort(-query_y_pred, kind='mergesort')] = np.arange(1, len(query_indexes) + 1)
            doc_pos_matrix = np.tile(document_positions, (len(query_indexes), 1))

            delta_ndcg = (((np.power(2, query_y_true.reshape(-1, 1)) - np.power(2, i_j_true))
                           * (1 / np.log2(1 + document_positions.reshape(-1, 1)) - 1 / np.log2(1 + doc_pos_matrix)))
                          / self.query_idcg[query_id])

            delta_preds = query_y_pred.reshape(-1, 1) - i_j_preds
            perm_mask = query_y_true.reshape(-1, 1) - i_j_true

            p_ij = np.zeros_like(delta_preds)
            p_ij += expit(-self.sigma * delta_preds) * (perm_mask > 0)
            p_ij += expit(self.sigma * delta_preds) * (perm_mask < 0)

            lambda_ij = - self.sigma * np.abs(delta_ndcg) * p_ij
            query_lambdas = np.sum(lambda_ij * (perm_mask > 0) - lambda_ij * (perm_mask < 0), axis=1)
            lambdas[query_indexes] = query_lambdas

            loss += self._compute_dcg(query_y_true[np.argsort(-query_y_pred)], self.top_k)
            idcg += self._compute_dcg(sorted(query_y_true, reverse=True), self.top_k)

            hess[query_indexes] = self.sigma * np.sum(
                np.abs(delta_ndcg) * p_ij * (1 - p_ij) * (perm_mask != 0), axis=1)

        self.train_loss.append(loss / idcg)

        return lambdas, hess

    def _reweight_tree_by_newton_step(self, X, tree, lambdas, hess):
        leaf_index_dct = defaultdict(list)
        for sample_index, leaf_index in enumerate(tree.tree_.apply(X)):
            leaf_index_dct[leaf_index].append(sample_index)

        for leaf_index, sample_indexes in leaf_index_dct.items():
            nom = - lambdas[sample_indexes].sum()
            denom = hess[sample_indexes].sum()
            if nom == 0 or denom == 0:
                tree.tree_.value[leaf_index] = 0.
            else:
                tree.tree_.value[leaf_index] = nom / denom

        return tree

    def fit(self, X, y, query_ids, X_val=None, y_val=None, q_val=None):
        self._preprocess(y, query_ids)
        if y_val is not None and q_val is not None:
            self._preprocess(y_val, q_val, is_val=True)
            val_predictions = np.zeros_like(y_val)

        predictions = np.zeros_like(y)
        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTreeRegressor(max_depth=self.params['max_depth'], max_features=self.params['max_features'])
            lambdas, hess = self._compute_lambdas(y, predictions)
            tree.fit(X, - lambdas)
            tree = self._reweight_tree_by_newton_step(X, tree, lambdas, hess)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)

            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * tree.predict(X_val)
                self.compute_val_score(y_val, val_predictions)
                print('Iter {}, train_loss={:.6f}'.format(len(self.trees), self.train_loss[-1]))
                print('Iter {}, val_loss={:.6f}'.format(len(self.trees), self.val_loss[-1]))
            else:
                print('Iter {}, train_loss={:.6f}'.format(len(self.trees), self.train_loss[-1]))

        return self

    def predict(self, X):
        preds = np.sum([self.learning_rate * tree.predict(X) for tree in self.trees], axis=0)
        return preds
