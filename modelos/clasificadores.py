# -*- coding: utf-8 -*-
"""Clasificadores del experimento, equivalentes a los de Weka.

| Algoritmo Weka | Implementación |
|----------------|----------------|
| NaiveBayes     | GaussianNB     |
| IBk (k=5)      | KNeighborsClassifier |
| J48 (C4.5)     | DecisionTreeClassifier (criterio entropía, poda vía min_samples_leaf) |
| SMO            | CalibratedClassifierCV(SVC RBF, C=1) |
| Logistic       | LogisticRegression (C=1, regularizado) |
| RandomForest   | RandomForestClassifier (100 árboles) |
| MultilayerPerceptron | MLPClassifier (capa oculta 32, early stopping) |

Modelos regularizados y acotados por defecto para robustez (overfitting).
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg

S = cfg.SEED


def obten_clasificadores(seed=cfg.SEED):
    """Devuelve un dict {nombre weka: factory()} de clasificadores."""
    return {
        "NaiveBayes": lambda: GaussianNB(),
        "IBk": lambda: KNeighborsClassifier(n_neighbors=5),
        "J48": lambda: DecisionTreeClassifier(criterion="entropy",
                                              min_samples_leaf=5,
                                              random_state=seed),
        "SMO": lambda: CalibratedClassifierCV(
            SVC(C=1.0, gamma="scale", random_state=seed), ensemble=False, cv=3),
        "Logistic": lambda: LogisticRegression(C=1.0, max_iter=3000,
                                               random_state=seed),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1),
        "MultilayerPerceptron": lambda: MLPClassifier(
            hidden_layer_sizes=(32,), alpha=1e-3, max_iter=1200,
            early_stopping=True, n_iter_no_change=20, random_state=seed),
    }