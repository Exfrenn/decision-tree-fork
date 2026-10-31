"""Ejemplo rapido para probar el DecisionTreeClassifier.

Se corre desde la raiz del proyecto con:  python main.py
"""
import numpy as np

from decision_tree import DecisionTreeClassifier


def main():
    # Datos test
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [9.0, 1.0],
            [9.0, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    # Probamos con el criterio Gini (el que viene por defecto).
    clf = DecisionTreeClassifier(criterion="gini").fit(X, y)
    print("criterio gini")
    print("  predicciones:", clf.predict(X))
    print("  accuracy:", clf.eval(X, y))

    # Y ahora con entropia, limitando la profundidad para ver el hiperparametro.
    clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(X, y)
    print("criterio entropy (max_depth=3)")
    print("  predicciones:", clf2.predict(X))
    print("  accuracy:", clf2.eval(X, y))


if __name__ == "__main__":
    main()
