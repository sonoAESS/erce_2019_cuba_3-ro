# -*- coding: utf-8 -*-
"""Prueba de integración ligera del orquestador (sin datos reales).

Los helpers internos entrenan con el preprocesador por defecto (columnas de
`config.FEATURES`), así que la fixture sintética respeta esa forma; el flujo
completo con los datos ERCE reales se valida con
`scripts/verificar_artefactos.py` tras el entrenamiento.
"""
import numpy as np
import pandas as pd
import pytest

from modelos import config as cfg, experimento as ex
from modelos.clasificadores import obten_clasificadores
from modelos.preprocesamiento import construir_preprocesador


def _preparar(n=240, seed=0):
    """DataFrame con la forma de FEATURES ERCE y una clase binaria sintética."""
    rng = np.random.RandomState(seed)
    datos = {c: rng.randn(n) for c in cfg.FEATURES_NUMERICOS}
    datos.update({c: rng.choice(["a", "b", "c"], n)
                  for c in cfg.FEATURES_CATEGORICOS})
    df = pd.DataFrame(datos)
    df.loc[::11, "ISECF"] = np.nan  # un NaN que imputará el pipeline
    X = df[cfg.FEATURES]
    y = pd.Series(rng.randint(0, 2, n))
    return X, y


def test_pipeline_cv_y_entrenar_final():
    X, y = _preparar()
    modelado_s = obten_clasificadores(seed=42)
    config = {"selector": "infogain", "algoritmo": "NaiveBayes", "balanceo": "none"}

    pipe, nombres = ex._entrenar_final(X, y, "clasificacion", config,
                                       modelado_s, 42)
    assert len(nombres) == pipe.named_steps["sel"].support_.sum()
    yp = pipe.named_steps["mod"].predict(
        pipe.named_steps["sel"].transform(
            pipe.named_steps["pre"].transform(X)))
    assert len(yp) == len(y)

    pipe_cv = ex._pipeline_cv(config, "clasificacion", modelado_s, 42)
    pipe_cv.fit(X, y)
    yp2 = pipe_cv.predict(X)
    assert len(yp2) == len(y)


def test_probas_clasif():
    from sklearn.naive_bayes import GaussianNB
    X = np.nan_to_num(np.asarray(pd.DataFrame(
        {c: np.random.RandomState(1).randn(120)
         for c in cfg.FEATURES_NUMERICOS})[cfg.FEATURES_NUMERICOS]))
    y = np.random.RandomState(1).randint(0, 2, 120)
    m = GaussianNB().fit(X, y)
    p = ex._probas(m, X)
    assert p.shape == (120, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_nombre_original_categoricas():
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        **{c: rng.randn(n) for c in cfg.FEATURES_NUMERICOS},
        **{c: rng.choice(["x", "y"], n) for c in cfg.FEATURES_CATEGORICOS},
    })
    pre = construir_preprocesador()
    pre.fit(df[cfg.FEATURES])
    assert ex._nombre_original("provincia_x", pre.numericos) == "provincia"
    assert ex._nombre_original("ISECF", pre.numericos) == "ISECF"


def test_importancias_df():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    n = 100
    nombres = list(cfg.FEATURES_NUMERICOS)[:3]
    X = np.nan_to_num(np.asarray(pd.DataFrame(
        {c: rng.randn(n) for c in cfg.FEATURES_NUMERICOS})[nombres]))
    y = rng.randint(0, 2, n)
    m = LogisticRegression(max_iter=500).fit(X, y)
    imp = ex._importancias(m, X[:60], y[:60], nombres, "clasificacion")
    assert list(imp.columns) == ["variable", "importancia"]
    assert len(imp) == 3