import numbers
import numpy as np
import itertools

from sklearn.base import BaseEstimator
from sklearn.ensemble.bagging import (BaseEnsemble, _parallel_build_estimators, 
                                      _partition_estimators,_parallel_predict_regression)
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed

MAX_INT = np.iinfo(np.int32).max

class MultiOutputBaggingRegressor(BaseEnsemble):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        self.estimators_ = []
        self.estimators_features_ = []
        
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        n_samples, self.n_features_ = X.shape
        
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)
        
        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:
            max_samples = int(self.max_samples * X.shape[0])
            
        self._max_features = max_features
        self._max_samples = max_samples
        self.base_estimator_ = self.base_estimator
        
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds
        
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                self.n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))
        
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))
            
        return self

    def predict(self, X):
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))
        
        return sum(all_y_hat) / self.n_estimators