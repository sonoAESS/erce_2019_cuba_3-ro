# -*- coding: utf-8 -*-
"""Evaluadores de selección de atributos fieles al proceso de Weka.

Cuatro AttributeEvaluators + búsqueda, tal como los usa `AttributeSelection`
dentro de un `FilteredClassifier` de Weka:

1. **CfsSubsetEval + GreedyStepwise** — subconjunto con mejor mérito
   (correlación con la clase y entre atributos).
2. **ChiSquaredAttributeEval + Ranker** — chi-cuadrado tras discretizar
   (en regresión se usa el F univariado, equivalente funcional).
3. **InfoGainAttributeEval + Ranker** — ganancia de información
   (`mutual_info_*`), top-K.
4. **WrapperSubsetEval + GreedyStepwise** — búsqueda con la precisión del
   propio clasificador/regresor base (forward selection, CV interna).

Todos implementan la interfaz `fit/transform` de scikit-learn para poder
vivir como paso intermedio de un `Pipeline` (y, al predecir, se aplican en
transform, igual que un filtro de Weka).
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.metrics import make_scorer, f1_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge

from . import config as cfg


def _codificar_y(y):
    """Codifica etiquetas de clase a enteros (0..k-1) p.ej. 'I','II','III'."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        return y.astype(float)
    return LabelEncoder().fit_transform(y).astype(float)


def _preseleccionar_candidatos(X, y, task_type, seed=cfg.SEED, top=None):
    """Rankea atributos con un criterio univariado y acota los candidatos.

    Equivalente a aplicar un `Ranker` (ChiSquared/InfoGain o F univariado)
    antes de la búsqueda por subconjunto (CFS/Wrapper). Es una heurística
    práctica de Weka cuando el número de atributos es grande: filtra a los
    `top` primeros y deja que el evaluador de subconjunto profundice solo en
    ellos. Devuelve la lista de índices de los `top` mejores.
    """
    y = _codificar_y(y)
    top = min(int(top or cfg.PRESELECCION_MAX), X.shape[1])
    if X.shape[1] <= top:
        return list(range(X.shape[1]))
    if task_type == "clasificacion":
        import warnings
        disc = KBinsDiscretizer(n_bins=5, strategy="quantile", encode="ordinal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xd = disc.fit_transform(X)
        scores = SelectKBest(chi2, k=min(top, Xd.shape[1])).fit(Xd, y).scores_
    else:
        scores = SelectKBest(f_regression, k=min(top, X.shape[1])).fit(X, y).scores_
    scores = np.nan_to_num(scores, nan=0.0, neginf=-np.inf, posinf=np.inf)
    return np.argsort(scores)[::-1].tolist()[:top]


class _BaseSelector(BaseEstimator, TransformerMixin):
    """Base común: expone `support_` (máscara booleana) y `transform`."""

    def __init__(self, task_type="clasificacion", seed=cfg.SEED, etiqueta=""):
        self.task_type = task_type
        self.seed = seed
        self.etiqueta = etiqueta

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def resumen(self):
        return {
            "selector": self.etiqueta or self.__class__.__name__,
            "n_seleccionados": int(self.support_.sum()),
            "seleccionados": [i for i, s in enumerate(self.support_) if s],
        }


# ------------------------------------------------------------------ CFS
class CfsSubsetGreedy(_BaseSelector):
    """CfsSubsetEval + GreedyStepwise (correlación con clase y entre atributos).

    Mérito (Weka): merit = k*media(r_cf) / sqrt(k + k*(k−1)*media(r_ff)).

    Búsqueda greedily hacia adelante: añade el atributo que más mejora el
    mérito hasta que ninguna adición lo incrementa.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = _codificar_y(y)
        d = X.shape[1]

        def merit(sub):
            k = len(sub)
            if k == 0:
                return 0.0
            Xs = X[:, sub]
            rcf = np.empty(k)
            for j in range(k):
                c = Xs[:, j]
                if np.std(c) == 0 or np.std(y) == 0:
                    rcf[j] = 0.0
                else:
                    rcf[j] = abs(pearsonr(c, y).statistic)
            if k == 1:
                return rcf[0]
            R = np.corrcoef(Xs.T)
            R = np.nan_to_num(R, nan=0.0)
            rff = (R.sum() - k) / (k * (k - 1))
            denom = np.sqrt(k + k * (k - 1) * rff)
            m = (k * rcf.mean()) / denom if denom > 0 else 0.0
            # pequeña penalización de tamaño: empates se resuelven con el más corto
            return m - 1e-9 * k

        subset = set()
        # Con muchos atributos, prefiltra por ranking univariado (equivalent
        # Ranker de Weka) y deja que el greedy profundice solo en los mejores.
        disponibles = _preseleccionar_candidatos(X, y, self.task_type,
                                                 seed=self.seed,
                                                 top=cfg.PRESELECCION_MAX)
        mejor_merit = 0.0
        while disponibles:
            mejor = None
            for f in disponibles:
                m = merit(sorted(subset | {f}))
                if mejor is None or m > mejor[1] + 1e-6:
                    mejor = (f, m)
            if mejor is None or mejor[1] <= mejor_merit + 1e-6:
                break
            subset.add(mejor[0])
            disponibles.remove(mejor[0])
            mejor_merit = mejor[1]
        self.support_ = np.zeros(d, dtype=bool)
        self.support_[list(subset)] = True
        self._merit_ = merit(sorted(subset))
        return self


# --------------------------------------------------------------- Ranker
def _rank_top(scores, k):
    scores = np.asarray(scores, dtype=float)
    k = min(int(k or cfg.K_SELECTOR), len(scores))
    if k == len(scores):
        return np.ones(len(scores), dtype=bool)
    top = np.argsort(scores)[-k:]
    mask = np.zeros(len(scores), dtype=bool)
    mask[top] = True
    return mask


class ChiCuadradoRanker(_BaseSelector):
    """ChiSquaredAttributeEval + Ranker (clasificación).

    Discretiza las features (equivalente a discretizar en Weka) y aplica
    chi-cuadrado; en regresión usa el estadístico F univariado.
    """

    def __init__(self, task_type="clasificacion", k=cfg.K_SELECTOR,
                 seed=cfg.SEED, etiqueta=""):
        super().__init__(task_type=task_type, seed=seed, etiqueta=etiqueta)
        self.k = k

    def fit(self, X, y):
        import warnings
        X = np.asarray(X, dtype=float)
        y = _codificar_y(y)
        if self.task_type == "clasificacion":
            disc = KBinsDiscretizer(n_bins=5, strategy="quantile",
                                    encode="ordinal")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                Xd = disc.fit_transform(X)
            self.scores_ = SelectKBest(chi2, k=min(self.k, X.shape[1])).fit(Xd, y).scores_
        else:
            self.scores_ = SelectKBest(f_regression,
                                       k=min(self.k, X.shape[1])).fit(X, y).scores_
        self.support_ = _rank_top(self.scores_, self.k)
        return self


class InfoGainRanker(_BaseSelector):
    """InfoGainAttributeEval + Ranker (ganancia de información, top-K)."""

    def __init__(self, task_type="clasificacion", k=cfg.K_SELECTOR,
                 seed=cfg.SEED, etiqueta=""):
        super().__init__(task_type=task_type, seed=seed, etiqueta=etiqueta)
        self.k = k

    def fit(self, X, y):
        from sklearn.feature_selection import (mutual_info_classif,
                                               mutual_info_regression)
        X = np.asarray(X, dtype=float)
        y = _codificar_y(y)
        mi = (mutual_info_classif if self.task_type == "clasificacion"
              else mutual_info_regression)
        self.scores_ = np.asarray(mi(X, y, random_state=self.seed), dtype=float)
        self.support_ = _rank_top(self.scores_, self.k)
        return self


# --------------------------------------------------------------- Wrapper
class WrapperSubset(_BaseSelector):
    """WrapperSubsetEval + GreedyStepwise con el clasificador/regresor base.

    Evalúa cada subconjunto por el rendimiento (F1 macro / RMSE) del propio
    estimador con CV interna. Búsqueda hacia adelante con tope de atributos
    para limitar el costo (fiel al espíritu de Weka, no a su implementación).
    """

    def __init__(self, task_type="clasificacion", seed=cfg.SEED, etiqueta="",
                 estimador=None, cv=2, max_features=cfg.MAX_WRAPPER_FEATURES,
                 n_jobs=2):
        super().__init__(task_type=task_type, seed=seed, etiqueta=etiqueta)
        self.estimador = estimador
        self.cv = cv
        self.max_features = max_features
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        est = self.estimador
        if est is None:
            est = (LogisticRegression(max_iter=2000, random_state=self.seed)
                   if self.task_type == "clasificacion" else Ridge())
        if self.task_type == "clasificacion":
            scorer = make_scorer(f1_score, average="macro")
            folds = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                    random_state=self.seed)
        else:
            scorer = make_scorer(root_mean_squared_error,
                                 greater_is_better=False)
            folds = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        subset = set()
        # Prefiltro por ranking univariado: el wrapper del subconjunto con el
        # estimador base es caro; acota la búsqueda forward a los mejores
        # candidatos según F/chi-cuadrado (Ranker previo a la subbúsqueda).
        disponibles = _preseleccionar_candidatos(X, y, self.task_type,
                                                 seed=self.seed,
                                                 top=cfg.PRESELECCION_WRAPPER)
        mejor_actual = -np.inf
        while disponibles and len(subset) < self.max_features:
            mejores = None
            mejor_s = -np.inf
            for f in disponibles:
                cand = sorted(subset | {f})
                s = cross_val_score(est, X[:, cand], y, cv=folds,
                                    scoring=scorer, n_jobs=self.n_jobs).mean()
                if s > mejor_s + 1e-9:
                    mejor_s = s
                    mejores = f
            if mejores is None:
                break
            if mejor_s <= mejor_actual + 1e-9:
                break
            subset.add(mejores)
            disponibles.remove(mejores)
            mejor_actual = mejor_s
        self.support_ = np.zeros(d, dtype=bool)
        self.support_[list(subset)] = True
        self.mejor_ = mejor_actual
        return self


SELECTORES = {
    "cfs": CfsSubsetGreedy,
    "chi2": ChiCuadradoRanker,
    "infogain": InfoGainRanker,
    "wrapper": WrapperSubset,
}
ETIQUETAS_WEEKA = {
    "cfs": "CfsSubsetEval + GreedyStepwise",
    "chi2": "ChiSquaredAttributeEval + Ranker",
    "infogain": "InfoGainAttributeEval + Ranker",
    "wrapper": "WrapperSubsetEval + GreedyStepwise",
}


def crear_selector(nombre, task_type, estimador=None, seed=cfg.SEED):
    """Construye un evaluador por nombre ('cfs'|'chi2'|'infogain'|'wrapper')."""
    cls = SELECTORES[nombre]
    if nombre == "wrapper":
        return cls(task_type=task_type, seed=seed, estimador=estimador,
                   etiqueta=nombre)
    return cls(task_type=task_type, seed=seed, etiqueta=nombre)