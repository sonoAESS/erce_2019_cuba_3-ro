# -*- coding: utf-8 -*-
"""Generación de datasets de entrenamiento desde el paquete `erce`.

Cada dataset se escribe en `modelos/datos/<tarea>.csv` con TODAS las features
primero y la VARIABLE OBJETIVO en la ÚLTIMA columna (convención del proceso:
el IR y el balanceo se calculan sobre la última columna de la base de casos).

El dataset maestro se construye sobre `erce.data.dataframe_modulo_mat()`
(Módulo Nacional de Matemática, CSV adicionado por el cliente; 4899 estudiantes
tras deduplicar) e incorpora el contexto sociodemográfico de la base ERCE
(ISECF, condiciones de vida) y los objetivos de matemática (PM3, nivel NL_VP1).
"""
import os
import sys

import pandas as pd

from . import config as cfg

_cols_numericas = {
    *cfg.FEATURES_NUMERICOS,
    "PM3", "REPC",
}


def _es_num(col_df):
    return col_df.name in _cols_numericas


def dataset_maestro():
    """DataFrame maestro con features + objetivos (sin transformar aún)."""
    from erce import data as D

    df = D.dataframe_modulo_mat()

    # Variables derivadas (nivel matemático = NL_VP1 del archivo PM3)
    df["riesgo"] = (df["NL_VP1"] == "I").astype(int)
    df["superacion"] = df["NL_VP1"].isin(["III", "IV"]).astype(int)

    out = df[cfg.FEATURES].copy()
    for c in cfg.FEATURES:
        if c in cfg.FEATURES_CATEGORICOS:
            out[c] = out[c].astype("category")
    out["NL_VP1"] = df["NL_VP1"].astype(str)
    out["REPC"] = df["REPC"]
    out["PM3"] = df["PM3"]
    return out


def _columnas_objetivo(tarea, master):
    """Devuelve el nombre y la serie objetivo de una tarea."""
    spec = cfg.TAREAS[tarea]
    if spec.get("derivada"):
        return spec["columna"] or tarea, spec["derivada"](master)
    return spec["columna"], master[spec["columna"]]


def _guardar(df_validos, tarea):
    os.makedirs(cfg.DATOS_DIR, exist_ok=True)
    ruta = os.path.join(cfg.DATOS_DIR, f"{tarea}.csv")
    df_validos.to_csv(ruta, index=False)
    return ruta


def generar_datasets(master=None):
    """Escribe los 6 datasets (`features` + objetivo en última columna).

    Devuelve dict {tarea: ruta_csv}.
    """
    master = dataset_maestro() if master is None else master
    rutas = {}
    for tarea, spec in cfg.TAREAS.items():
        obj_nombre, y = _columnas_objetivo(tarea, master)
        feats = pd.DataFrame(master[cfg.FEATURES], copy=True)
        tabla = pd.concat([feats.reset_index(drop=True),
                           y.reset_index(drop=True).rename(obj_nombre)], axis=1)
        # Filas con objetivo válido (la clase/objetivo nunca debe ser NaN)
        tabla = tabla[tabla[obj_nombre].notna()].reset_index(drop=True)
        # Coerción de tipos numéricos (los 0/1 quedan como float y pasan a int si aplica)
        for c in tabla.columns:
            if c not in cfg.FEATURES_CATEGORICOS and c != obj_nombre:
                tabla[c] = pd.to_numeric(tabla[c], errors="coerce")
        if obj_nombre in {"riesgo", "superacion", "REPC"}:
            tabla[obj_nombre] = tabla[obj_nombre].astype(int)
        rutas[tarea] = _guardar(tabla, tarea)
    return rutas


def cargar_dataset(tarea):
    """Lee el CSV de una tarea (features + objetivo en última columna)."""
    ruta = os.path.join(cfg.DATOS_DIR, f"{tarea}.csv")
    if not os.path.exists(ruta):
        generar_datasets()
    df = pd.read_csv(ruta)
    return df


def separar_X_y(df, tarea):
    """Separa features (X) y objetivo (y) asumiendo objetivos en última columna."""
    spec = cfg.TAREAS[tarea]
    obj = spec.get("columna") or tarea
    return df.drop(columns=[obj]), df[obj]


if __name__ == "__main__":
    sys.path.insert(0, cfg.BASE_DIR)
    rutas = generar_datasets()
    for tarea, ruta in rutas.items():
        print(f"{tarea:4s} {ruta}")