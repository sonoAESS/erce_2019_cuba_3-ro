# -*- coding: utf-8 -*-
import numpy as np
import pytest

from modelos import metricas, sobreajuste


def test_metricas_regresion_diagonal(datos_regresion):
    y = datos_regresion["y"].to_numpy()
    m = metricas.calcular_metricas_regresion(y, y)
    assert m["r"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_metricas_clasificacion_exactitud(datos_clasif):
    y = datos_clasif["y"].to_numpy()
    m = metricas.calcular_metricas_clasificacion(
        y, y, np.column_stack([1 - y, y]), [0, 1])
    assert m["exactitud"] == pytest.approx(1.0)
    assert m["kappa"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["matriz"].shape == (2, 2)
    assert m["por_clase"]["TP_Rate"].tolist() == pytest.approx([1.0, 1.0])


def test_metricas_clasificacion_multiclase():
    y = np.array(["I", "II", "III", "I", "II", "III", "III", "III"])
    yp = np.array(["I", "II", "III", "I", "II", "III", "III", "III"])
    prob = np.eye(3)[np.array([0, 1, 2, 0, 1, 2, 2, 2])]
    m = metricas.calcular_metricas_clasificacion(y, yp, prob, ["I", "II", "III"])
    assert m["clases"] == ["I", "II", "III"]
    assert m["auc_macro"] == pytest.approx(1.0)


def test_fold_clasif_con_probabilidades_nulas():
    y = np.array([0, 1, 0, 1, 0, 1])
    m = metricas.metricas_fold_clasif(y, y, np.column_stack([1 - y, y]), [0, 1])
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["auc_macro"] == pytest.approx(1.0)


def test_dividir_holdout_estratificado(datos_clasif):
    X = datos_clasif[["num1", "num2"]]
    y = datos_clasif["y"]
    Xtr, Xte, ytr, yte = sobreajuste.dividir_holdout(X, y, "clasificacion")
    prop_tr = ytr.mean()
    prop_te = yte.mean()
    assert abs(prop_tr - prop_te) < 0.06
    assert len(ytr) + len(yte) == len(y)


def test_detectar_sobreajuste():
    s1 = sobreajuste.detectar_sobreajuste(0.95, 0.70)
    assert s1["estado"] == "sobreajuste"
    s2 = sobreajuste.detectar_sobreajuste(0.45, 0.40)
    assert s2["estado"] == "underfitting"
    s3 = sobreajuste.detectar_sobreajuste(0.72, 0.70)
    assert s3["estado"] == "ok"


def test_curva_aprendizaje_regresion(datos_regresion):
    from sklearn.linear_model import Ridge
    X = datos_regresion[["num1", "num2"]]
    y = datos_regresion["y"]
    curva = sobreajuste.curva_aprendizaje(Ridge(), X, y, "regresion",
                                          "neg_root_mean_squared_error",
                                          cv=2, sizes=[0.3, 0.8], n_jobs=1)
    assert len(curva) == 2
    assert curva[-1]["n_entrenamiento"] > curva[0]["n_entrenamiento"]
    assert np.isfinite(curva[0]["train_media"])