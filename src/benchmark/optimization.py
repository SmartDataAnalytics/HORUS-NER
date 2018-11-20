import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV

from src.config import HorusConfig

config = HorusConfig()
trees_param_bootstrap = {"max_features": ['auto', 'sqrt'],
                     "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                     "min_samples_split": [2, 5, 10],
                     "min_samples_leaf": [1, 2, 4],
                     "n_estimators": [10, 25, 50, 100, 200, 400, 600, 800],
                     "bootstrap": [True, False]
}

config_classification = [(RandomForestClassifier(), trees_param_bootstrap, 'random')]

X_train = None
X_test = None
y_train = None

for estimator, hyper, grid_method in config_classification:
    clf = RandomizedSearchCV(estimator, hyper, cv=5, scoring=['precision', 'recall', 'f1'],
                             n_jobs=-1, refit='f1')
    config.logger.info('training')
    clf.fit(X_train, y_train)
    config.logger.info(clf.best_params_)
    config.logger.info(clf.best_score_)
    predicted = clf.best_estimator_.predict(X_test)
