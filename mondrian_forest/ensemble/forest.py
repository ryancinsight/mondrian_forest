import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod
from sklearn.ensemble._base import BaseEnsemble
from sklearn.base import RegressorMixin
import six
from sklearn.exceptions import NotFittedError
from joblib import delayed, Parallel
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y

from ..tree import MondrianTreeRegressor
MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None):
    """Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(BaseForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def apply(self, X):
        """Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           backend="threading")(
            delayed(parallel_helper)(tree, 'apply', X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """Return the decision path in the forest

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        n_nodes_ptr : array of size (n_estimators + 1, )
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              backend="threading")(
            delayed(parallel_helper)(tree, 'decision_path', X,
                                      check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            for i in range(n_more_estimators):
                tree = self._make_estimator(append=False,
                                            random_state=random_state)
                trees.append(tree)

            # Parallel loop: we use the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading always more efficient than multiprocessing in
            # that case.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        check_is_fitted(self, 'estimators_')

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   backend="threading")(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_)

        return sum(all_importances) / len(self.estimators_)


# This is a utility function for joblib's Parallel. It can't go locally in
# ForestClassifier or ForestRegressor, because joblib complains that it cannot
# pickle it when placed there.

def accumulate_prediction(predict, X, out):
    prediction = predict(X, check_input=False)
    if len(out) == 1:
        out[0] += prediction
    else:
        for i in range(len(out)):
            out[i] += prediction[i]

class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):
    """Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ForestRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(accumulate_prediction)(e.predict, X, [y_hat])
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_

def _single_tree_pfit(tree, X, y, classes=None):
    if classes is not None:
        tree.partial_fit(X, y, classes)
    else:
        tree.partial_fit(X, y)
    return tree

class BaseMondrian(object):
    def weighted_decision_path(self, X):
        """
        Returns the weighted decision path in the forest.

        Each non-zero value in the decision path determines the
        weight of that particular node while making predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input.

        Returns
        -------
        decision_path : sparse csr matrix, shape = (n_samples, n_total_nodes)
            Return a node indicator matrix where non zero elements
            indicate the weight of that particular node in making predictions.

        est_inds : array-like, shape = (n_estimators + 1,)
            weighted_decision_path[:, est_inds[i]: est_inds[i + 1]]
            provides the weighted_decision_path of estimator i
        """
        X = self._validate_X_predict(X)
        est_inds = np.cumsum(
            [0] + [est.tree_.node_count for est in self.estimators_])
        paths = sparse.hstack(
            [est.weighted_decision_path(X) for est in self.estimators_]).tocsr()
        return paths, est_inds

    # XXX: This is mainly a stripped version of BaseForest.fit
    # from sklearn.forest
    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Forests.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianForest
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        random_state = check_random_state(self.random_state)

        # Wipe out estimators if partial_fit is called after fit.
        first_call = not hasattr(self, "first_")
        if first_call:
            self.first_ = True

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        self.n_outputs_ = 1

        # Initialize estimators at first call to partial_fit.
        if first_call:
            # Check estimators
            self._validate_estimator()
            self.estimators_ = []

            for _ in range(self.n_estimators):
                tree = self._make_estimator(append=False, random_state=random_state)
                self.estimators_.append(tree)

        # XXX: Switch to threading backend when GIL is released.
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_single_tree_pfit)(t, X, y) for t in self.estimators_)

        return self


class MondrianForestRegressor(ForestRegressor, BaseMondrian):
    """
    A MondrianForestRegressor is an ensemble of MondrianTreeRegressors.

    The variance in predictions is reduced by averaging the predictions
    from all trees.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_depth : integer, optional (default=None)
        The depth to which each tree is grown. If None, the tree is either
        grown to full depth or is constrained by `min_samples_split`.

    min_samples_split : integer, optional (default=2)
        Stop growing the tree if all the nodes have lesser than
        `min_samples_split` number of samples.

    bootstrap : boolean, optional (default=False)
        If bootstrap is set to False, then all trees are trained on the
        entire training dataset. Else, each tree is fit on n_samples
        drawn with replacement from the training dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(MondrianForestRegressor, self).__init__(
            base_estimator=MondrianTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("max_depth", "min_samples_split",
                              "random_state"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """Builds a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        return super(MondrianForestRegressor, self).fit(X, y)

    def predict(self, X, return_std=False):
        """
        Returns the predicted mean and std.

        The prediction is a GMM drawn from
        \(\sum_{i=1}^T w_i N(m_i, \sigma_i)\) where \(w_i = {1 \over T}\).

        The mean \(E[Y | X]\) reduces to \({\sum_{i=1}^T m_i \over T}\)

        The variance \(Var[Y | X]\) is given by $$Var[Y | X] = E[Y^2 | X] - E[Y | X]^2$$
        $$=\\frac{\sum_{i=1}^T E[Y^2_i| X]}{T} - E[Y | X]^2$$
        $$= \\frac{\sum_{i=1}^T (Var[Y_i | X] + E[Y_i | X]^2)}{T} - E[Y| X]^2$$

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input samples.

        return_std : boolean, default (False)
            Whether or not to return the standard deviation.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            Predictions at X.

        std : array-like, shape = (n_samples,)
            Standard deviation at X.
        """
        X = check_array(X)
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model has to be fit before prediction.")
        ensemble_mean = np.zeros(X.shape[0])
        exp_y_sq = np.zeros_like(ensemble_mean)

        for est in self.estimators_:
            if return_std:
                mean, std = est.predict(X, return_std=True)
                exp_y_sq += (std**2 + mean**2)
            else:
                mean = est.predict(X, return_std=False)
            ensemble_mean += mean

        ensemble_mean /= len(self.estimators_)
        exp_y_sq /= len(self.estimators_)

        if not return_std:
            return ensemble_mean
        std = exp_y_sq - ensemble_mean**2
        std[std <= 0.0] = 0.0
        std **= 0.5
        return ensemble_mean, std

    def partial_fit(self, X, y):
        """
        Incremental building of Mondrian Forest Regressors.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianForestClassifier
        """
        return super(MondrianForestRegressor, self).partial_fit(X, y)