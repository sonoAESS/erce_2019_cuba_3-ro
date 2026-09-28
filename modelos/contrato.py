# -*- coding: utf-8 -*-
"""Contrato de entrada de datos nuevos para el software de escritorio.

El `metadata.json` de cada tarea fija el contrato: features esperadas (orden y
tipo numérico/categórico). Al cargar un CSV/Excel nuevo en la app:
- Se comprueban columnas requeridas (faltantes y sobrantes, con aviso).
- Se tipan numéricas y categóricas.
- El pipeline (ColumnTransformer con `handle_unknown='ignore'`) tolera
  categorías inéditas en las categóricas; las numéricas fuera de rango no
  rompen la predicción.

La función `validar_y_tipar` es el único punto de entrada que necesita el
frontend de PySide6.
"""
import numpy as np
import pandas as pd

from . import config as cfg


def _dtype_num(col):
    return pd.to_numeric(col, errors="coerce")


def validar_y_tipar(df, metadata):
    """Valida `df` contra el contrato `metadata` y devuelve el df listo.

    Devuelve un dict: {ok, faltantes, sobrantes, df} donde `df` tiene las
    columnas en el orden canónico de `features_esperadas`.
    """
    esperadas = [str(c) for c in metadata["features_esperadas"]]
    df = df.copy()

    presentes = [str(c) for c in df.columns]
    faltantes = [c for c in esperadas if c not in presentes]
    sobrantes = [c for c in presentes if c not in esperadas]
    df = df.drop(columns=[c for c in sobrantes if c in df.columns])

    numericas = [c for c in esperadas if c in df.columns and
                 c in cfg.FEATURES_NUMERICOS]
    for c in numericas:
        df[c] = _dtype_num(df[c])
    for c in esperadas:
        if c in df.columns and c not in numericas:
            # object (no StringDtype): evita pd.NA, el imputer espera np.nan
            s = df[c].astype(object)
            df[c] = s.where(pd.notna(s), other=np.nan)

    df = df[[c for c in esperadas if c in df.columns]]
    return {
        "ok": not faltantes,
        "faltantes": faltantes,
        "sobrantes": sobrantes,
        "num_observaciones": int(len(df)),
        "df": df,
    }