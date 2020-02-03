import numpy as np
import pandas as pd

from lambdamart import LambdaMART
from sklearn.datasets import load_svmlight_file


def get_sumbmission():
    X, y, query_ids = load_svmlight_file('train.txt', query_id=True)

    X = np.asarray(X.todense())
    non_zero_columns_mask = X.sum(0) != 0
    X = X[:, non_zero_columns_mask].astype(np.float32)

    params = {
        'n_estimators': 1,
        'max_depth': 10,
        # 'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'learning_rate': 0.05,
        'sigma': 1,
        'top_k': 5
    }

    gbm = LambdaMART(n_trees=3000, **params)

    gbm.fit(X, y, query_ids)

    X_test, y_test, query_ids_test = load_svmlight_file('test.txt', query_id=True)

    X_test = np.asarray(X_test.todense())
    X_test = X_test[:, non_zero_columns_mask].astype(np.float32)

    pred_test = gbm.predict(X_test)

    subm = pd.read_csv('sample.made.fall.2019')

    subm['QueryId'] = query_ids_test
    subm['DocumentId'] = np.arange(1, subm.shape[0] + 1)
    subm['pred'] = pred_test

    subm = (subm[['QueryId']]
            .drop_duplicates()
            .merge(subm.sort_values(by=['QueryId', 'pred'], ascending=False)))[['QueryId', 'DocumentId']]

    subm.to_csv('subm.csv', index=False)


if __name__ == '__main__':
    get_sumbmission()
