# -*- coding: utf-8 -*-
"""Algoritmos adicionales para el cribado del Módulo Nacional.

`clasificadores.py` y `regresores.py` mantienen el conjunto base (7 y 6
algoritmos) que reproduce el proceso del Experimenter de Weka sobre los cinco
modelos ya entrenados. Este módulo **no los modifica**: vive aparte para que el
cribado amplio del Módulo Nacional compare muchas más familias sin que cambie
la rejilla de esos modelos.

Se separan en dos niveles por coste, porque una rejilla de 4 selectores × N
algoritmos × 5 folds se paga en cada ajuste:

- **Nivel A** entra en la rejilla completa con los cuatro selectores de Weka:
  modelos de ajuste rápido (lineales, bayesianos, discriminantes, vecindarios,
  árboles y ensembles de bagging o boosting histogramado).
- **Nivel B** entra solo en el cribado y en el afinado: familias caras (márgenes
  con núcleo, perceptrón multicapa, boosting clásico). En la rejilla × 4
  selectores multiplicarían el tiempo por cuatro o más sin cambiar la
  conclusión.

No están `ComplementNB` ni `MultinomialNB`: exigen variables no negativas y
el preprocesador estandariza (igual que hace Weka con Normalize), así que
fallarían con `Negative values in data`. `BernoulliNB` sí aguanta el escalado
porque binariza en `> 0`, que con media cero equivale a "por encima de la
media".

Los nombres siguen el criterio de `clasificadores.py`: se conserva el nombre de
Weka cuando el algoritmo es su equivalente (SMO, SMOreg, M5P, J48, IBk) y se
usa el de la familia cuando no hay equivalente (ExtraTrees, HistGB, ElasticNet,
KRR, Voting, Stacking). Varios nombres coinciden con los del registro base a
propósito: para el Módulo Nacional el registro extendido **sustituye** al base,
no se suman.

| Algoritmo | Implementación |
|-----------|----------------|
| Logistic | LogisticRegression (C=1) |
| LDA | LinearDiscriminantAnalysis |
| QDA | QuadraticDiscriminantAnalysis (reg_param=0,1) |
| NaiveBayes | GaussianNB |
| BernoulliNB | BernoulliNB (binariza en >0: con variables estandarizadas, "por encima de la media") |
| J48 | DecisionTreeClassifier (entropía, min_samples_leaf=5) |
| M5P | DecisionTreeRegressor (min_samples_leaf=5) |
| IBk | KNeighbors(k=5) |
| RandomForest / ExtraTrees | 100 árboles cada uno |
| HistGB | HistGradientBoosting (200 etapas, lr 0,1) |
| LinearRegression | Ridge (α=1) |
| ElasticNet | ElasticNet (α=0,1; l1_ratio=0,5) |
| SMO | CalibratedClassifierCV(SVC RBF, C=1) |
| SMOreg | SVR (RBF, C=1) |
| LinearSVC / LinearSVR | margen lineal (C=1, dual automático) |
| GradientBoosting | boosting clásico (100 etapas, lr 0,1) |
| KRR | KernelRidge (núcleo RBF) — solo regresión |
| MultilayerPerceptron | MLP (32 ocultas, early stopping) |
| Voting | votación blanda sobre tres familias baratas |
| Stacking | los mismos tres modelos base con ridge como meta |
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (ElasticNet, LogisticRegression, Ridge,
                                  RidgeClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config as cfg

VOTING = "Voting"
STACKING = "Stacking"


# --------------------------------------------------------------- factorías
def _arbol(seed, tipo):
    if tipo == "clasificacion":
        return lambda: DecisionTreeClassifier(criterion="entropy",
                                              min_samples_leaf=5,
                                              random_state=seed)
    return lambda: DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)


def _bosque(seed, clase):
    if clase == "clasificacion":
        return lambda: RandomForestClassifier(n_estimators=100, random_state=seed,
                                              n_jobs=-1)
    return lambda: RandomForestRegressor(n_estimators=100, random_state=seed,
                                         n_jobs=-1)


def _extra(seed, clase):
    if clase == "clasificacion":
        return lambda: ExtraTreesClassifier(n_estimators=100, random_state=seed,
                                            n_jobs=-1)
    return lambda: ExtraTreesRegressor(n_estimators=100, random_state=seed,
                                       n_jobs=-1)


def _mlp(seed, tipo):
    if tipo == "clasificacion":
        return lambda: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3,
                                     max_iter=1200, early_stopping=True,
                                     n_iter_no_change=20, random_state=seed)
    return lambda: MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3,
                                max_iter=1200, early_stopping=True,
                                n_iter_no_change=20, random_state=seed)


def nivel_a_clasificadores(seed=cfg.SEED):
    """Familias baratas: completas para la rejilla con los cuatro selectores."""
    return {
        "Logistic": lambda: LogisticRegression(C=1.0, max_iter=3000,
                                               random_state=seed),
        "LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "QDA": lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
        "NaiveBayes": lambda: GaussianNB(),
        "BernoulliNB": lambda: BernoulliNB(),
        "J48": _arbol(seed, "clasificacion"),
        "IBk": lambda: KNeighborsClassifier(n_neighbors=5),
        "RandomForest": _bosque(seed, "clasificacion"),
        "ExtraTrees": _extra(seed, "clasificacion"),
        "HistGB": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=seed),
    }


def nivel_a_regresores(seed=cfg.SEED):
    """Familias baratas de regresión (ver `nivel_a_clasificadores`)."""
    return {
        "LinearRegression": lambda: Ridge(alpha=1.0),
        "ElasticNet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5,
                                         max_iter=5000, random_state=seed),
        "M5P": _arbol(seed, "regresion"),
        "IBk": lambda: KNeighborsRegressor(n_neighbors=5),
        "RandomForest": _bosque(seed, "regresion"),
        "ExtraTrees": _extra(seed, "regresion"),
        "HistGB": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=seed),
    }


def nivel_b_clasificadores(seed=cfg.SEED):
    """Familias caras: cribado y afinado, no rejilla × 4 selectores."""
    return {
        "SMO": lambda: CalibratedClassifierCV(
            SVC(C=1.0, gamma="scale", random_state=seed), ensemble=False, cv=3),
        "LinearSVC": lambda: LinearSVC(C=1.0, dual="auto", max_iter=5000,
                                       random_state=seed),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=seed),
        "MultilayerPerceptron": _mlp(seed, "clasificacion"),
    }


def nivel_b_regresores(seed=cfg.SEED):
    """Familias caras de regresión (ver `nivel_b_clasificadores`)."""
    return {
        "SMOreg": lambda: SVR(C=1.0, gamma="scale"),
        "LinearSVR": lambda: LinearSVR(C=1.0, dual="auto", max_iter=5000,
                                       random_state=seed),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=seed),
        "KRR": lambda: KernelRidge(alpha=1.0, kernel="rbf"),
        "MultilayerPerceptron": _mlp(seed, "regresion"),
    }


# --------------------------------------------------------------- ensembles
def voting_clasificador(seed=cfg.SEED):
    """Votación blanda sobre tres familias distintas y baratas."""
    return VotingClassifier(
        estimators=[("log", LogisticRegression(C=1.0, max_iter=3000,
                                               random_state=seed)),
                    ("bosque", RandomForestClassifier(n_estimators=200,
                                                      random_state=seed,
                                                      n_jobs=-1)),
                    ("gb", HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.1, random_state=seed))],
        voting="soft")


def voting_regresor(seed=cfg.SEED):
    """Votación (promedio) sobre las mismas tres familias."""
    return VotingRegressor(
        estimators=[("ridge", Ridge(alpha=1.0)),
                    ("bosque", RandomForestRegressor(n_estimators=200,
                                                     random_state=seed,
                                                     n_jobs=-1)),
                    ("gb", HistGradientBoostingRegressor(
                        max_iter=200, learning_rate=0.1, random_state=seed))])


def stacking_clasificador(seed=cfg.SEED):
    """Stacking con los mismos tres modelos base y ridge como meta."""
    return StackingClassifier(
        estimators=[("log", LogisticRegression(C=1.0, max_iter=3000,
                                               random_state=seed)),
                    ("arbol", DecisionTreeClassifier(max_depth=6,
                                                     random_state=seed)),
                    ("gb", HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.1, random_state=seed))],
        final_estimator=RidgeClassifier(), cv=3, n_jobs=-1)


def stacking_regresor(seed=cfg.SEED):
    """Stacking con ridge como meta-regresor."""
    return StackingRegressor(
        estimators=[("ridge", Ridge(alpha=1.0)),
                    ("arbol", DecisionTreeRegressor(max_depth=6,
                                                     random_state=seed)),
                    ("gb", HistGradientBoostingRegressor(
                        max_iter=200, learning_rate=0.1, random_state=seed))],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)


ENSEMBLES = {
    "clasificacion": {VOTING: voting_clasificador,
                      STACKING: stacking_clasificador},
    "regresion": {VOTING: voting_regresor,
                  STACKING: stacking_regresor},
}


# --------------------------------------------------------------- registro
def obten_clasificadores_extendidos(seed=cfg.SEED, con_ensembles=True):
    """Nivel A + nivel B + ensembles, con los nombres del proceso Weka.

    El nombre deja claro que no reemplaza a `obten_clasificadores`: para el
    Módulo Nacional este registro es el que se usa en la rejilla.
    """
    d = nivel_a_clasificadores(seed)
    d.update(nivel_b_clasificadores(seed))
    if con_ensembles:
        d.update(ENSEMBLES["clasificacion"])
    return d


def obten_regresores_extendidos(seed=cfg.SEED, con_ensembles=True):
    """Nivel A + nivel B + ensembles de regresión."""
    d = nivel_a_regresores(seed)
    d.update(nivel_b_regresores(seed))
    if con_ensembles:
        d.update(ENSEMBLES["regresion"])
    return d


def nombres_cribado(tipo, con_ensembles=False):
    """Nombres del cribado, en el orden en que aparecen en el informe."""
    if tipo == "clasificacion":
        d = obten_clasificadores_extendidos(con_ensembles=con_ensembles)
    else:
        d = obten_regresores_extendidos(con_ensembles=con_ensembles)
    return list(d)


def grupos(tipo):
    """Reparte los nombres en los dos barridos del cribado.

    Devuelve `(nivel_a, nivel_b)` con los ensembles dentro del segundo grupo:
    el nivel A entra en la rejilla completa con los cuatro selectores de Weka y
    el nivel B solo con `CfsSubsetEval`, que es el selector por defecto de Weka.
    Así el cribado cubre todas las familias sin pagar cuatro veces el coste de
    las más caras.
    """
    if tipo == "clasificacion":
        a = list(nivel_a_clasificadores())
        b = list(nivel_b_clasificadores())
    else:
        a = list(nivel_a_regresores())
        b = list(nivel_b_regresores())
    return a, b + list(ENSEMBLES[tipo])


SIN_WRAPPER = (VOTING, STACKING)
"""Modelos que no se cruzan con `WrapperSubsetEval`.

Cada subconjunto que el wrapper evalúa ya es un `cross_val_score` del modelo
completo; con un ensemble de tres modelos base eso multiplica el coste del
cribado por más de diez sin aportar información nueva sobre el subconjunto.
"""


# --------------------------------------------------------------- afinado
# Distribuciones para `RandomizedSearchCV`, acotadas a rangos con sentido para
# ~4 000 filas y 28 variables. Los ensembles quedan fuera a propósito: con
# esta muestra, afinar un ensemble sin haber afinado sus componentes solo
# produce sobreajuste.
ESPACIOS_AJUSTE = {
    "clasificacion": {
        "Logistic": {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
                     "class_weight": [None, "balanced"]},
        "LDA": {"shrinkage": ["auto", 0.1, 0.5, 0.9]},
        "QDA": {"reg_param": [0.0, 0.05, 0.1, 0.3, 0.6]},
        "NaiveBayes": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        "BernoulliNB": {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0]},
        "J48": {"max_depth": [3, 5, 8, 12, None],
                "min_samples_leaf": [1, 5, 10, 20, 40],
                "criterion": ["entropy", "gini"]},
        "IBk": {"n_neighbors": [3, 5, 9, 15, 25, 40],
                "weights": ["uniform", "distance"], "p": [1, 2]},
        "RandomForest": {"n_estimators": [200, 400],
                         "max_depth": [None, 6, 10, 14],
                         "min_samples_leaf": [1, 5, 10, 20],
                         "max_features": ["sqrt", 0.3, 0.5]},
        "ExtraTrees": {"n_estimators": [200, 400],
                       "max_depth": [None, 6, 10, 14],
                       "min_samples_leaf": [1, 5, 10, 20],
                       "max_features": ["sqrt", 0.3, 0.5]},
        "HistGB": {"learning_rate": [0.03, 0.06, 0.1, 0.2],
                   "max_iter": [100, 200, 400],
                   "max_leaf_nodes": [15, 31, 63],
                   "min_samples_leaf": [10, 20, 40, 80],
                   "l2_regularization": [0.0, 0.1, 1.0]},
        "SMO": {"estimator__C": [0.1, 1.0, 10.0, 100.0],
                "estimator__gamma": ["scale", 0.001, 0.01, 0.1]},
        "LinearSVC": {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
                      "class_weight": [None, "balanced"]},
        "GradientBoosting": {"n_estimators": [100, 200, 400],
                             "learning_rate": [0.03, 0.1, 0.2],
                             "max_depth": [2, 3, 5],
                             "min_samples_leaf": [5, 10, 20]},
        "MultilayerPerceptron": {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                                  "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                                  "learning_rate_init": [3e-4, 1e-3, 3e-3]},
    },
    "regresion": {
        "LinearRegression": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "ElasticNet": {"alpha": [0.01, 0.1, 1.0, 10.0],
                       "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]},
        "M5P": {"max_depth": [3, 5, 8, 12, None],
                "min_samples_leaf": [5, 10, 20, 40]},
        "IBk": {"n_neighbors": [3, 5, 9, 15, 25, 40],
                "weights": ["uniform", "distance"], "p": [1, 2]},
        "RandomForest": {"n_estimators": [200, 400],
                         "max_depth": [None, 6, 10, 14],
                         "min_samples_leaf": [1, 5, 10, 20],
                         "max_features": ["sqrt", 0.3, 0.5]},
        "ExtraTrees": {"n_estimators": [200, 400],
                       "max_depth": [None, 6, 10, 14],
                       "min_samples_leaf": [1, 5, 10, 20],
                       "max_features": ["sqrt", 0.3, 0.5]},
        "HistGB": {"learning_rate": [0.03, 0.06, 0.1, 0.2],
                   "max_iter": [100, 200, 400],
                   "max_leaf_nodes": [15, 31, 63],
                   "min_samples_leaf": [10, 20, 40, 80],
                   "l2_regularization": [0.0, 0.1, 1.0]},
        "SMOreg": {"C": [0.1, 1.0, 10.0, 100.0],
                   "gamma": ["scale", 0.001, 0.01, 0.1],
                   "epsilon": [0.01, 0.1, 0.2]},
        "LinearSVR": {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
                      "epsilon": [0.0, 0.1, 0.2]},
        "GradientBoosting": {"n_estimators": [100, 200, 400],
                             "learning_rate": [0.03, 0.1, 0.2],
                             "max_depth": [2, 3, 5],
                             "min_samples_leaf": [5, 10, 20]},
        "KRR": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "gamma": [0.001, 0.01, 0.1, "scale"]},
        "MultilayerPerceptron": {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                                  "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                                  "learning_rate_init": [3e-4, 1e-3, 3e-3]},
    },
}

N_AFINADO = 20
CV_AFINADO = 5


def espacio_ajuste(nombre, tipo):
    """Distribuciones del algoritmo, o `None` si no tiene nada que buscar."""
    return ESPACIOS_AJUSTE.get(tipo, {}).get(nombre)


def afinables(tipo, nombres, n=N_AFINADO):
    """Filtra `nombres` a los que tienen espacio de búsqueda."""
    return [x for x in nombres if espacio_ajuste(x, tipo) is not None]