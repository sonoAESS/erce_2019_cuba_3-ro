# -*- coding: utf-8 -*-
"""Tests del bloque de construcción de modelos (Weka-fiel).

Fixture base: datos sintéticos pequeños con features numéricas y categóricas
para no depender de los datasets reales en los tests unitarios.
"""
import numpy as np
import pytest

from modelos import config as cfg
from modelos.preprocesamiento import construir_preprocesador


@pytest.fixture
def datos_clasif():
    """DataFrame pequeño y determinista (binario) con NaN y categoría extra."""
    import pandas as pd
    n = 200
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "num1": rng.randn(n) + (np.arange(n) % 2),
        "num2": rng.randn(n) * 0.5,
        "cat1": np.random.RandomState(1).choice(["A", "B", "C"], n),
        "y": (np.arange(n) % 2),
    })
    df.loc[::7, "num1"] = np.nan
    return df


@pytest.fixture
def datos_regresion():
    import pandas as pd
    n = 200
    rng = np.random.RandomState(0)
    x = rng.randn(n)
    return pd.DataFrame({
        "num1": x,
        "num2": rng.randn(n),
        "cat1": np.random.RandomState(1).choice(["A", "B"], n),
        "y": 3.0 * x + rng.randn(n) * 0.1,
    })


@pytest.fixture
def preprocessor():
    return construir_preprocesador(numericos=["num1", "num2"],
                                   categoricos=["cat1"])


def test_preprocesador_campos_ok(preprocessor, datos_clasif):
    X = datos_clasif.drop(columns=["y"])
    preprocessor.fit(X)
    out = preprocessor.transform(X)
    assert out.shape[0] == len(X)
    # mediana imputó el NaN
    assert np.isfinite(out).all()