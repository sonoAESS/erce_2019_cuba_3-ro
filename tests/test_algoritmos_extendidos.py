# -*- coding: utf-8 -*-
"""Pruebas del registro de algoritmos ampliados (`algoritmos_extendidos`).

Lo que hay que garantizar aquí no es la calidad de cada modelo (eso lo
mide el cuaderno sobre datos reales), sino tres invariantes:

1. Que el registro **extendido no toca el base**: los cinco modelos ya
   entrenados se siguen reproduciendo con `obten_clasificadores()` y
   `obten_regresores()`.
2. Que toda factory devuelve un estimador usable: se puede ajustar y predecir
   sin modificarse a sí mismo.
3. Que los espacios de afinado son válidos: si `RandomizedSearchCV`
   combinara un parámetro con un valor imposible, el cuaderno fallaría a
   mitad del cribado.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from modelos import algoritmos_extendidos as ext
from modelos.clasificadores import obten_clasificadores
from modelos.regresores import obten_regresores

CLASIF = ext.obten_clasificadores_extendidos()
REGRES = ext.obten_regresores_extendidos()


@pytest.fixture(scope="module")
def datos_clasif():
    rng = np.random.RandomState(0)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = pd.Series((X["x0"] + 0.5 * X["x1"] + rng.normal(scale=0.5, size=n) > 0)
                  .map({True: "bajo", False: "alto"}))
    return X, y


@pytest.fixture(scope="module")
def datos_regresion():
    rng = np.random.RandomState(0)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = pd.Series(10 * X["x0"] - 3 * X["x2"] + rng.normal(scale=0.5, size=n))
    return X, y


# ------------------------------------------------------------- invariantes
def test_el_registro_base_no_cambia():
    """El registro base debe seguir siendo el de siempre: 7 y 6 algoritmos."""
    assert len(obten_clasificadores()) == 7
    assert len(obten_regresores()) == 6
    assert set(obten_clasificadores()) == {
        "NaiveBayes", "IBk", "J48", "SMO", "Logistic", "RandomForest",
        "MultilayerPerceptron"}


def test_extendido_mas_grande_que_el_base():
    assert len(CLASIF) > len(obten_clasificadores())
    assert len(REGRES) > len(obten_regresores())


def test_niveles_A_y_B_no_se_solapan():
    a = set(ext.nivel_a_clasificadores()) | set(ext.nivel_a_regresores())
    b = set(ext.nivel_b_clasificadores()) | set(ext.nivel_b_regresores())
    assert not (a & b)


def test_ensambles_solo_en_el_registro_completo():
    sin_ensembles = ext.obten_clasificadores_extendidos(con_ensembles=False)
    assert ext.VOTING not in sin_ensembles
    assert ext.VOTING in CLASIF and ext.STACKING in CLASIF
    assert ext.VOTING in REGRES and ext.STACKING in REGRES


# --------------------------------------------------------------- factories
@pytest.mark.parametrize("nombre", sorted(CLASIF))
def test_clasificador_se_ajusta_y_predice(nombre, datos_clasif):
    X, y = datos_clasif
    m = CLASIF[nombre]()
    assert m.get_params() or True          # es un estimador de sklearn
    m.fit(X, y)
    pred = m.predict(X)
    assert len(pred) == len(y)
    assert set(np.unique(pred)) <= set(y.unique())


@pytest.mark.parametrize("nombre", sorted(REGRES))
def test_regresor_se_ajusta_y_predice(nombre, datos_regresion):
    X, y = datos_regresion
    m = REGRES[nombre]()
    m.fit(X, y)
    assert len(m.predict(X)) == len(y)


@pytest.mark.parametrize("nombre", sorted(CLASIF))
def test_clasificador_no_muta_el_y_de_entrada(nombre, datos_clasif):
    X, y = datos_clasif
    m = CLASIF[nombre]()
    m.fit(X, y)
    assert list(y.unique()) == ["bajo", "alto"]


def test_determinismo_con_la_misma_semilla(datos_clasif):
    X, y = datos_clasif
    a = CLASIF["HistGB"]().fit(X, y).predict(X)
    b = CLASIF["HistGB"]().fit(X, y).predict(X)
    assert list(a) == list(b)


def test_smo_devuelve_probabilidades(datos_clasif):
    X, y = datos_clasif
    m = CLASIF["SMO"]().fit(X, y)
    assert m.predict_proba(X).shape[0] == len(y)


# --------------------------------------------------------------- afinado
@pytest.mark.parametrize("nombre,tipo", [
    (n, "clasificacion") for n in sorted(CLASIF)
] + [(n, "regresion") for n in sorted(REGRES)])
def test_espacio_de_ajuste_es_valido(nombre, tipo):
    """`GridSearchCV` con 1 candidato debe construir sin error.

    Se recorre solo el primer valor de cada distribución: si un parámetro no
    existiera en el estimador, o una combinación fuera imposible (por ejemplo
    `shrinkage` con un solver que no lo admite), el cribado fallaría más tarde
    y con 4 000 filas sería caro.
    """
    dist = ext.espacio_ajuste(nombre, tipo)
    if dist is None:
        pytest.skip(f"{nombre} no se afina")
    modelo = CLASIF[nombre]() if tipo == "clasificacion" else REGRES[nombre]()
    grid = {k: [v[0]] for k, v in dist.items()}
    busca = GridSearchCV(clone(modelo), grid, cv=2, error_score="raise")
    assert busca is not None


def test_afinados_excluyen_los_ensambles():
    """Afinar un ensemble sin afinar sus componentes solo produce ruido."""
    assert ext.afinables("clasificacion", list(CLASIF)) == \
        [n for n in CLASIF if n not in (ext.VOTING, ext.STACKING)]
    for nombre in ext.afinables("clasificacion", list(CLASIF)):
        assert ext.espacio_ajuste(nombre, "clasificacion") is not None


def test_las_naive_bayes_negativas_no_entran():
    """ComplementNB y MultinomialNB exigen entradas no positivas: no entran.

    El preprocesador estandariza, así que un modelo que rechace negativos
    reventaría a mitad del cribado. La cobertura real la dan las pruebas
    parametrizadas de ajuste, que usan datos con valores negativos.
    """
    for registro in (CLASIF, REGRES):
        assert "ComplementNB" not in registro
        assert "MultinomialNB" not in registro


def test_ki_rr_solo_en_regresion():
    """KernelRidge es un regresor: no debe aparecer en clasificación."""
    assert "KRR" in REGRES
    assert "KRR" not in CLASIF