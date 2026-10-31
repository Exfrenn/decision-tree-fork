"""Regression tree skeleton."""
from .base import BaseDecisionTree
import numpy as np
from .node import Node


class DecisionTreeRegressor(BaseDecisionTree):
    """Tree variant for regression.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'mse', 'mae'.

    Hooks:
        _build_tree(X, y, depth): Recursively grow the tree.
        _impurity(y): Impurity of ``y`` per ``self.criterion``: mean squared error or mean absolute error of y around its center (mean for 'mse', median for 'mae').
        _best_split(X, y): Search all features/thresholds for the largest impurity decrease.
        _leaf_value(y): Return the mean of y.
        _score(y_true, y_pred): Metric for ``eval``: R^2 = 1 - SS_res / SS_tot.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion="mse"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.

        Basicamente aqui se construye el nodo del arbol, que se arma por el tamaño de y
        """
        n_samples = len(y)

        # --- Condiciones de parada (criterios para hacer una hoja) ---
        # 1) Si ya llegamos a la profundidad maxima permitida.
        # 2) Si quedan muy pocas muestras para dividir.
        # 3) Si el nodo ya es puro (todas las muestras de la misma clase).
        stop_by_depth = self.max_depth is not None and depth >= self.max_depth
        stop_by_samples = n_samples < self.min_samples_split
        is_pure = len(np.unique(y)) == 1

        if stop_by_depth or stop_by_samples or is_pure:
            return Node(value=self._leaf_value(y))

        # Buscamos la mejor division posible para este nodo.
        split = self._best_split(X, y)

        # Si no hay ninguna division que ayude, tambien hacemos hoja.
        if split is None:
            return Node(value=self._leaf_value(y))

        feature_index, threshold = split

        # Separamos las muestras: las que cumplen x <= umbral van a la izquierda.
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Construimos recursivamente los dos subarboles (bajamos un nivel).
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: mean squared error or mean absolute error of y around its center (mean for 'mse', median for 'mae').

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        n = len(y)
        # Si el nodo esta vacio, no hay impureza que calcular.
        if n == 0:
            return 0.0

        if self.criterion == "mse":
            # MSE 
            return np.mean((y - y.mean()) ** 2)
        elif self.criterion == "mae":
            # MAE
            return np.mean(np.abs(y - np.median(y)))
        else:
            raise ValueError(f"criterion invalido: {self.criterion}")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        n_samples, n_features = X.shape
        # Impureza del nodo actual, es el punto de referencia para medir ganancia.
        parent_impurity = self._impurity(y)

        best_gain = 0.0          # solo nos quedamos con splits que mejoren algo
        best_split = None        # (feature_index, threshold) del mejor split

        # Probamos columna por columna (cada feature).
        for feature_index in range(n_features):
            values = X[:, feature_index]
            # Candidatos a umbral: puntos medios entre valores unicos ordenados.
            unique_values = np.unique(values)
            if len(unique_values) == 1:
                # Si todos los valores son iguales no se puede dividir por aqui.
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            # Para cada umbral candidato medimos que tan buena es la division.
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                # Si una rama queda vacia este split no sirve.
                if n_left == 0 or n_right == 0:
                    continue

                # Impureza ponderada por el tamano de cada rama.
                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[right_mask])
                weighted = (n_left * imp_left + n_right * imp_right) / n_samples

                # La ganancia es cuanto bajamos la impureza respecto al padre.
                gain = parent_impurity - weighted

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)
        return best_split

    def _leaf_value(self, y):
        """Return the mean of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """

        # Clase mayoritaria: la que mas se repite entre las muestras de la hoja.
        return y.mean()

    def _valid_criteria(self):
        """Criterios de impureza que soporta el clasificador."""
        return ("mse", "mae")

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: R^2 = 1 - SS_res / SS_tot.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # R^2 = 1 - SS_res / SS_tot
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


