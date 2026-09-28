# -*- coding: utf-8 -*-
import numpy as np
import pytest

from modelos import balanceo
from modelos.seleccion_atributos import (ChiCuadradoRanker, InfoGainRanker,
                                         CfsSubsetGreedy, WrapperSubset)
from modelos.preprocesamiento import construir_preprocesador


def _X_sintetica(d=6, n=240, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    # dos features informativas correlacionadas con la clase
    y = (rng.randn(n) + X[:, 0] + 2.0 * X[:, 1] > 0).astype(float)
    return X, y


def test_calcular_ir():
    y = np.array([0, 0, 0, 0, 1, 1])
    assert balanceo.calcular_ir(y) == pytest.approx(4.0 / 2.0)
    assert balanceo.calcular_ir(np.ones(5)) is None
    # etiquetas string (multiclase)
    import pandas as pd
    s = pd.Series(["I", "I", "I", "I", "II", "II", "III"])
    assert balanceo.calcular_ir(s) == pytest.approx(4.0 / 1.0)


@pytest.mark.parametrize("ir,esperado", [
    (1.0, None), (1.19, None), (1.21, "smote"), (4.9, "smote"),
    (6.0, "smote_rus"), (12.0, "smoteenn")])
def test_elegir_balanceo(ir, esperado):
    assert balanceo.elegir_balanceo(ir) == esperado


def test_balanceo_aplicar_none_vs_smote(datos_clasif):
    X = np.nan_to_num(datos_clasif[["num1", "num2"]].to_numpy(float))
    y = np.array([1] * 130 + [0] * 70)  # desbalanceada a propósito
    Xn, yn = balanceo.aplicar_balanceo(X, y, "none")
    assert Xn is X and yn is y
    Xs, ys = balanceo.aplicar_balanceo(X, y, "smote", seed=42)
    assert len(ys) > len(y)
    assert abs(np.bincount(ys.astype(int))[0] - np.bincount(ys.astype(int))[1]) <= 1


def test_cfs_merit_selecciona_informativas():
    X, y = _X_sintetica()
    sel = CfsSubsetGreedy()
    sel.fit(X, y)
    assert sel.support_.sum() >= 1
    # las features 0 y 1 son las informativas y deberían estar presentes
    assert sel.support_[0] and sel.support_[1]


def test_chi2_ranker_respeta_k():
    X, y = _X_sintetica()
    sel = ChiCuadradoRanker(k=4)
    sel.fit(X, y)
    assert sel.support_.sum() == 4


def test_infogain_ranker_con_strings():
    X, y = _X_sintetica()
    y_str = np.where(y > 0.5, "alt", "baj")  # etiquetas de clase string
    sel = InfoGainRanker(k=3)
    sel.fit(X, y_str)
    assert sel.support_.sum() == 3


def test_wrapper_con_regresion(datos_regresion):
    from sklearn.linear_model import Ridge
    X = datos_regresion[["num1", "num2"]].to_numpy(float)
    y = datos_regresion["y"].to_numpy()
    sel = WrapperSubset(task_type="regresion", estimador=Ridge(),
                        cv=2, max_features=2)
    sel.fit(X, y)
    assert sel.support_.sum() >= 1


def test_wrapper_con_clasificacion(datos_clasif):
    from sklearn.naive_bayes import GaussianNB
    X = np.nan_to_num(datos_clasif[["num1", "num2"]].to_numpy(float))
    y = datos_clasif["y"].to_numpy()
    sel = WrapperSubset(task_type="clasificacion", estimador=GaussianNB(),
                        cv=2, max_features=2)
    sel.fit(X, y)
    assert sel.support_.sum() >= 1