import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_less

from mondrian_forest import MondrianTreeRegressor


def test_tree_predict():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = [-1, -1, -1, 1, 1, 1]
    T = [[-1, -1], [2, 2], [3, 2]]

    # This test is dependent on the random-state since the feature
    # and the threshold selected at every split is independent of the
    # label.
    mtr = MondrianTreeRegressor(max_depth=1, random_state=0)
    mtr.fit(X, y)
    mtr_tree = mtr.tree_
    cand_feature = mtr_tree.feature[0]
    cand_thresh = mtr_tree.threshold[0]
    assert_almost_equal(cand_thresh, -0.38669141)
    assert_almost_equal(cand_feature, 0.0)

    # Close to (1.0 / np.sum(np.max(X, axis=0) - np.min(X, axis=0)))
    assert_almost_equal(mtr_tree.tau[0], 0.07112633)

    # For [-1, -1]:
    # P_not_separated = 1.0
    # Root:
    # eta_root = 0.0 (inside the bounding boc of the root)
    # P_root = 1 - exp(0.0) = 0.0
    # weight_root = P_root
    # mean_root = 0.0
    # Leaf:
    # P_not_separated = 1.0 * (1 - 0.0) = 1.0
    # weight_leaf = P_not_separated = 1.0
    # mean_leaf = -1.0
    # prediction = 0.0 - 1.0 = -1.0

    # variance = (weight_root * (var_root + mean_root**2) +
    #             weight_leaf * (var_leaf + mean_leaf**2)) - mean**2
    # This reduces to weight_leaf * mean_leaf**2 - mean**2 = 1.0 * (1.0 - 1.0)
    # = 0.0

    # Similarly for [2, 2]:
    # prediction = 0.0 + 1.0
    # Variance reduces to zero

    # For [3, 2]
    # P_not_separated = 1.0
    # Root:
    # Delta_root = 0.07112633
    # eta_root = 1.0
    # weight_root = 1 - exp(-0.07112633) = 0.0686
    # Leaf:
    # weight_leaf = P_not_separated = (1 - 0.0686) = 0.93134421
    # prediction = weight_leaf

    # variance = (weight_root * (var_root + mean_root**2) +
    #             weight_leaf * (var_leaf + mean_leaf**2)) - mean**2
    # = 0.0686 * (1 + 0) + 0.93134 * (0 + 1) - 0.93134421**2 = 0.132597

    T_predict, T_std = mtr.predict(T, return_std=True)
    assert_array_almost_equal(T_predict, [-1.0, 1.0, 0.93134421])
    assert_array_almost_equal(T_std, np.sqrt([0.0, 0.0, 0.132597]))


def test_boston():
    """Consistency on boston house prices"""
    mtr = MondrianTreeRegressor(random_state=0)
    boston = load_boston()
    X, y = boston.data, boston.target
    mtr.fit(X, y)
    score = mean_squared_error(mtr.predict(X), y)
    assert_less(score, 1, "Failed with score = {0}".format(score))


def test_array_repr():
    X = np.arange(10000)[:, np.newaxis]
    y = np.arange(10000)
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.fit(X, y)


def test_pure_set():
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.fit(X, y)
    assert_array_almost_equal(mtr.predict(X), y)


def test_numerical_stability():
    X = np.array([
        [152.08097839, 140.40744019, 129.75102234, 159.90493774],
        [142.50700378, 135.81935120, 117.82884979, 162.75781250],
        [127.28772736, 140.40744019, 129.75102234, 159.90493774],
        [132.37025452, 143.71923828, 138.35694885, 157.84558105],
        [103.10237122, 143.71928406, 138.35696411, 157.84559631],
        [127.71276855, 143.71923828, 138.35694885, 157.84558105],
        [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

    y = np.array(
        [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916, 0.49622521])

    with np.errstate(all="raise"):
        mtr = MondrianTreeRegressor(random_state=0)
        mtr.fit(X, y)
        mtr.fit(X, -y)
        mtr.fit(-X, y)
        mtr.fit(-X, -y)


def test_min_samples_split():
    iris = load_iris()
    X, y = iris.data, iris.target
    mtr = MondrianTreeRegressor(min_samples_split=10, random_state=0)
    mtr.fit(X, y)
    n_node_samples = mtr.tree_.n_node_samples[mtr.tree_.children_left != -1]
    assert_less(np.min(n_node_samples), 11)


def test_tau():
    """
    Test time of split for the root.
    """
    X, y = make_regression(random_state=0, n_features=10)
    rate = np.sum(np.max(X, axis=0) - np.min(X, axis=0))
    mtr = MondrianTreeRegressor(random_state=0, max_depth=1)

    taus = []
    for random_state in np.arange(100):
        mtr.set_params(random_state=random_state).fit(X, y)
        taus.append(mtr.tree_.tau[0])
    assert_almost_equal(np.mean(taus), 1.0 / rate, 2)


def test_dimension_location():
    """
    Test dimension and location of split.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(100, 2)
    X[:, 1] *= 100
    y = rng.randn(100)

    mtr = MondrianTreeRegressor(random_state=0, max_depth=1)
    n = 200
    diff = np.max(X, axis=0) - np.min(X, axis=0)
    expected = diff / np.sum(diff) * n

    features = []
    thresholds = []
    for random_state in np.arange(200):
        mtr.set_params(random_state=random_state).fit(X, y)
        features.append(mtr.tree_.feature[0])
        thresholds.append(mtr.tree_.threshold[0])

    features = np.array(features)
    thresholds = np.array(thresholds)
    counts = np.bincount(features)
    calc_thresh = np.mean(thresholds[features == 1])
    exp_thresh = np.mean(X, axis=0)[-1]

    assert_less(expected[-1] - 5, counts[-1])
    assert_less(calc_thresh, exp_thresh + 3.0)
    assert_less(exp_thresh - 3.0, calc_thresh)


def test_node_weights():
    """
    Test the implementation of node_weights.
    """
    rng = np.random.RandomState(0)
    boston = load_boston()
    X, y = boston.data, boston.target
    n_train = 100
    n_test = 100
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[-n_test:], y[-n_test:]

    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.transform(X_test)

    # Test that when all samples are in the training data all weights
    # should be concentrated at the leaf.
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.fit(X_train, y_train)
    leaf_nodes = mtr.apply(X_train)
    weights_sparse = mtr.weighted_decision_path(X_train)
    assert_array_equal(weights_sparse.data, np.ones(X_train.shape[0]))
    assert_array_equal(weights_sparse.indices, leaf_nodes)
    assert_array_equal(weights_sparse.indptr, np.arange(n_train + 1))

    # Test prediction using the node_weights function gives similar results
    # to that using the prediction method.
    weights = mtr.weighted_decision_path(X_test)
    node_means = mtr.tree_.mean
    node_variances = mtr.tree_.variance
    variances1 = []
    means1 = []

    for startptr, endptr in zip(weights.indptr[:-1], weights.indptr[1:]):
        curr_nodes = weights.indices[startptr: endptr]
        curr_weights = weights.data[startptr: endptr]
        curr_means = node_means[curr_nodes]
        curr_var = node_variances[curr_nodes]

        means1.append(np.sum(curr_weights * curr_means))
        variances1.append(np.sum(curr_weights * (curr_var + curr_means**2)))

    means1 = np.array(means1)
    variances1 = np.array(variances1)
    variances1 -= means1**2
    means2, std2 = mtr.predict(X_test, return_std=True)
    assert_array_almost_equal(means1, means2, 5)
    assert_array_almost_equal(variances1, std2**2, 3)


def test_min_samples_leaf():
    X, y = make_regression(random_state=0)
    mtr = MondrianTreeRegressor(random_state=0, min_samples_leaf=5)
    mtr.fit(X, y)
    assert_array_less(4.0*np.ones(mtr.tree_.node_count), mtr.tree_.n_node_samples)
