# -*- coding: utf-8 -*-
"""Regresores del experimento, equivalentes a los de Weka.

| Algoritmo Weka | Implementación |
|----------------|----------------|
| LinearRegression (ridge) | Ridge (α=1.0) |
| SMOreg          | SVR (RBF, C=1) |
| IBk             | KNeighborsRegressor (k=5) |
| M5P             | DecisionTreeRegressor (hojas >= 5) |
| RandomForest    | RandomForestRegressor (100 árboles) |
| MultilayerPerceptron | MLPRegressor (32 ocultas, early stopping) |
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import config as cfg


def obten_regresores(seed=cfg.SEED):
    """Devuelve un dict {nombre weka: factory()} de regresores."""
    return {
        "LinearRegression": lambda: Ridge(alpha=1.0),
        "SMOreg": lambda: SVR(C=1.0, gamma="scale"),
        "IBk": lambda: KNeighborsRegressor(n_neighbors=5),
        "M5P": lambda: DecisionTreeRegressor(min_samples_leaf=5,
                                             random_state=seed),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=100, random_state=seed, n_jobs=-1),
        "MultilayerPerceptron": lambda: MLPRegressor(
            hidden_layer_sizes=(32,), alpha=1e-3, max_iter=1200,
            early_stopping=True, n_iter_no_change=20, random_state=seed),
    }