# -*- coding: utf-8 -*-
"""Preprocesamiento del bloque de construcción de modelos.

Convención de diseño (equivalente a un `FilteredClassifier` de Weka):
- El pipeline transforma features con imputación + escalado / encoding, luego
  aplica la **selección de atributos** y por último el clasificador/regresor.
- El balanceo NO queda dentro del pipeline persistido: se aplica solo en el
  entrenamiento (folds) y queda documentado en `metadata.json`; así, el
  software de escritorio solo necesita scikit-learn + joblib + pandas.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


def construir_preprocesador(numericos=None, categoricos=None):
    """ColumnTransformer listo para entrenar: imputa, escala y codifica.

    - Numericos: mediana + estandarización (aplica Weka para SVM/MLP/IBk/Logistic).
    - Categóricos: moda + one-hot con `handle_unknown='ignore'` (datos nuevos
      pueden traer categorías inéditas; se ignoran sin romper la predicción).
    """
    numericos = list(numericos or cfg.FEATURES_NUMERICOS)
    categoricos = list(categoricos or cfg.FEATURES_CATEGORICOS)

    num_pipe = Pipeline([
        ("imput", SimpleImputer(strategy="median")),
        ("esc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imput", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        [("num", num_pipe, numericos),
         ("cat", cat_pipe, categoricos)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.numericos = numericos
    ct.categoricos = categoricos
    return ct


def nombres_transformados(ct):
    """Nombres de columnas resultantes tras aplicar `ct` (tras fit o predict)."""
    nums = list(ct.numericos)
    cats = []
    transformers = {name: tr for name, tr in ct.named_transformers_.items()
                    if tr != "drop"}
    if "cat" in transformers:
        oh = transformers["cat"].named_steps["oh"]
        cats = list(oh.get_feature_names_out(ct.categoricos))
    return nums + cats


def matriz_design(df, nombres=None):
    """Devuelve X (numpy) y los nombres en orden desde un DataFrame de features."""
    cols = list(nombres or df.columns)
    X = df[cols].to_numpy(dtype=float, na_value=np.nan)
    return X, cols