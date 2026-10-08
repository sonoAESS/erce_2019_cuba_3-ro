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


def test_metricas_clasificacion_clases_str_con_objetivo_numerico():
    """`clases` en str + y numérico es el caso real de las tareas derivadas."""
    y = np.array([0.0] * 30 + [1.0] * 10)
    yp = np.array([0.0] * 28 + [1.0] * 2 + [1.0] * 10)
    m = metricas.calcular_metricas_clasificacion(
        y, yp, np.column_stack([1 - yp, yp]), ["0.0", "1.0"])
    assert m["matriz"].sum() == len(y)
    assert m["matriz"][1, 1] == 10
    assert m["por_clase"]["TP_Rate"].tolist() == pytest.approx(
        [28 / 30, 1.0], abs=1e-4)
    assert m["por_clase"]["MCC"].iloc[1] > 0


def test_linea_base_clasificacion_es_el_suelo():
    """Adivinar siempre lo más frecuente: kappa 0, AUC 0.5, f1>0 engañoso."""
    y = np.array([0.0] * 76 + [1.0] * 24)
    b = metricas.linea_base_clasificacion(y)
    assert b["exactitud"] == pytest.approx(0.76, abs=1e-4)
    assert b["kappa"] == pytest.approx(0.0)
    assert b["auc_macro"] == pytest.approx(0.5)
    assert b["f1_macro"] > 0.3  # <- el valor que se confunde con desempeño
    assert b["matriz"].sum() == len(y)


def test_linea_base_regresion_es_el_suelo():
    y = np.array([700.0, 750.0, 800.0, 820.0])
    b = metricas.linea_base_regresion(y)
    assert b["r2"] == pytest.approx(0.0)
    assert b["rmse"] == pytest.approx(y.std(ddof=0), abs=1e-3)


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


def test_detectar_sobreajuste_colapsado_a_la_base():
    """El caso que se leía como "ok": train y test iguales y ambos en la base.

    clf_repitencia daba brecha +0.005 y validación 0.487 > piso, así que pasaba
    por "brecha controlada" cuando en realidad colapsaba a la clase mayoritaria.
    """
    s = sobreajuste.detectar_sobreajuste(0.492, 0.487, linea_base=0.487)
    assert s["estado"] == "nulo"
    assert s["linea_base"] == pytest.approx(0.487)


def test_detectar_sobreajuste_supera_la_base_levemente():
    s = sobreajuste.detectar_sobreajuste(0.50, 0.494, linea_base=0.432)
    assert s["estado"] == "ok"


def test_detectar_sobreajuste_regresion_piso_no_aplica():
    """En regresión la métrica va negada: el piso de 0.45 no debe disparar."""
    s = sobreajuste.detectar_sobreajuste(-60.0, -50.0, piso=None, linea_base=-75.0)
    assert s["estado"] == "ok"
    # Por debajo de la media del objetivo: RMSE mayor que el de la base.
    s2 = sobreajuste.detectar_sobreajuste(-50.0, -80.0, piso=None, linea_base=-75.0)
    assert s2["estado"] == "nulo"


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

def test_auc_binaria_con_clases_en_orden_de_aparicion():
    """`clases` como ['1','0']: la positiva no es siempre la columna 1.

    `y_prob` llega ALINEADA a `clases` (reordena `_probas`/`probas`), así que
    con el orden invertido la columna de la positiva es la 0. Antes se usaba
    siempre `y_prob[:, 1]` y el AUC salía 0,0 en lugar de 1,0.
    """
    y = np.array([1] * 30 + [0] * 30)  # str(y) = '1'/'0', como en la tarea real
    scores = np.r_[np.linspace(0.6, 1.0, 30), np.linspace(0.0, 0.4, 30)]
    P = np.column_stack([scores, 1 - scores])  # columnas en orden ["1", "0"]
    m = metricas.calcular_metricas_clasificacion(y, y, P, ["1", "0"])
    assert m["auc_macro"] == pytest.approx(1.0)
    assert m["por_clase"].set_index("Clase").loc["1", "ROC_Area"] == pytest.approx(1.0)


def test_auc_multiclase_reordena_a_orden_alfabetico_de_y_true():
    """`multi_class="ovr"` exige las columnas en orden alfabético de `y_true`,
    pero `clases` guarda el orden de aparición (['bajo','medio','alto'])."""
    y = np.array(["bajo"] * 20 + ["medio"] * 20 + ["alto"] * 20)
    clases = ["bajo", "medio", "alto"]  # orden de aparición, no alfabético
    P = np.full((60, 3), 0.1)
    bloques = {"bajo": slice(0, 20), "medio": slice(20, 40), "alto": slice(40, 60)}
    for i, c in enumerate(clases):
        P[bloques[c], i] = 0.8
    m = metricas.calcular_metricas_clasificacion(y, y, P, clases)
    assert m["auc_macro"] == pytest.approx(1.0)
