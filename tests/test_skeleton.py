import numpy as np
import pytest

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseDecisionTree()


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_hyperparameters(cls):
    model = cls(max_depth=3, min_samples_split=4)
    assert model.get_params()["max_depth"] == 3
    model.set_params(criterion="entropy")
    assert model.criterion == "entropy"
    with pytest.raises(ValueError):
        model.set_params(foo=1)


# --- Datos sencillos y linealmente separables para el clasificador ---
def _toy_data():
    X = np.array(
        [
            [2.7, 2.5],
            [1.4, 2.3],
            [3.4, 4.4],
            [1.4, 1.8],
            [3.1, 3.0],
            [7.6, 2.7],
            [5.3, 2.1],
            [6.9, 1.7],
            [8.6, -0.2],
            [7.7, 3.5],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


# ----------------------------- Tests del classifier -----------------------------
def test_classifier_fit_returns_self_and_sets_root():
    X, y = _toy_data()
    model = DecisionTreeClassifier(max_depth=3)
    result = model.fit(X, y)
    assert result is model
    assert model.root is not None


def test_classifier_predict_perfect_on_separable_data():
    X, y = _toy_data()
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.array_equal(preds, y)


def test_classifier_eval_accuracy_is_one_on_training_set():
    X, y = _toy_data()
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert model.eval(X, y) == pytest.approx(1.0)


def test_classifier_pure_node_makes_leaf():
    # Todas las muestras de la misma clase -> la raiz debe ser hoja.
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5, 5, 5])
    model = DecisionTreeClassifier().fit(X, y)
    assert model.root.is_leaf
    assert np.array_equal(model.predict([[10.0]]), np.array([5]))


def test_classifier_leaf_value_is_majority_class():
    model = DecisionTreeClassifier()
    assert model._leaf_value(np.array([0, 0, 1, 0, 1])) == 0


def test_classifier_gini_impurity_values():
    model = DecisionTreeClassifier(criterion="gini")
    # Nodo puro -> impureza 0.
    assert model._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)
    # Mezcla 50/50 -> Gini = 0.5.
    assert model._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_classifier_entropy_impurity_values():
    model = DecisionTreeClassifier(criterion="entropy")
    assert model._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)
    # Mezcla 50/50 -> entropia = 1 bit.
    assert model._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_classifier_both_criteria_fit_separable_data():
    X, y = _toy_data()
    for criterion in ("gini", "entropy"):
        model = DecisionTreeClassifier(criterion=criterion, max_depth=3).fit(X, y)
        assert model.eval(X, y) == pytest.approx(1.0)


def test_classifier_invalid_criterion_raises():
    X, y = _toy_data()
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="no_existe").fit(X, y)


def test_classifier_invalid_min_samples_split_raises():
    X, y = _toy_data()
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1).fit(X, y)


def test_classifier_predict_before_fit_raises():
    model = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        model.predict([[0.0, 0.0]])


# ----------------------------- Tests del regressor -----------------------------
def _toy_reg_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    return X, y


def test_regressor_leaf_value_is_mean():
    model = DecisionTreeRegressor()
    assert model._leaf_value(np.array([2.0, 4.0, 6.0])) == pytest.approx(4.0)


def test_regressor_mse_impurity_values():
    model = DecisionTreeRegressor(criterion="mse")
    assert model._impurity(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    # var de [0, 2] = 1.0
    assert model._impurity(np.array([0.0, 2.0])) == pytest.approx(1.0)


def test_regressor_mae_impurity_values():
    model = DecisionTreeRegressor(criterion="mae")
    assert model._impurity(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    # mediana de [1,2,3,4] = 2.5 -> mean(|y-2.5|) = 1.0
    assert model._impurity(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)


def test_regressor_fit_returns_self_and_sets_root():
    X, y = _toy_reg_data()
    model = DecisionTreeRegressor(max_depth=3)
    result = model.fit(X, y)
    assert result is model
    assert model.root is not None


@pytest.mark.parametrize("criterion", ["mse", "mae"])
def test_regressor_fits_and_scores_perfectly(criterion):
    X, y = _toy_reg_data()
    model = DecisionTreeRegressor(max_depth=3, criterion=criterion).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.allclose(preds, y)
    assert model.eval(X, y) == pytest.approx(1.0)


def test_regressor_r2_of_mean_predictor_is_zero():
    # Predecir siempre la media -> R^2 = 0.
    model = DecisionTreeRegressor()
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert model._score(y_true, y_pred) == pytest.approx(0.0)


def test_regressor_invalid_criterion_raises():
    X, y = _toy_reg_data()
    with pytest.raises(ValueError):
        DecisionTreeRegressor(criterion="no_existe").fit(X, y)


def test_regressor_predict_before_fit_raises():
    model = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        model.predict([[0.0]])
