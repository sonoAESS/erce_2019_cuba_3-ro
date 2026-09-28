# -*- coding: utf-8 -*-
"""Generación de datasets de entrenamiento desde el paquete `erce`.

Cada dataset se escribe en `modelos/datos/<tarea>.csv` con TODAS las features
primero y la VARIABLE OBJETIVO en la ÚLTIMA columna (convención del proceso:
el IR y el balanceo se calculan sobre la última columna de la base de casos).

El dataset maestro se construye sobre `erce.data.dataframe_estudiantes()`
(5273 estudiantes) e incorpora los índices de calidad del Módulo Nacional
(merge por IDSTUD, con NaN imputados más adelante por el pipeline).
"""
import os
import sys

import pandas as pd

from . import config as cfg

_cols_numericas = {
    *cfg.FEATURES_NUMERICOS,
    "PL3", "PM3", "NL_VP1", "REPC",
}


def _es_num(col_df):
    return col_df.name in _cols_numericas


def dataset_maestro():
    """DataFrame maestro con features + objetivos (sin transformar aún)."""
    from erce import data as D

    df = D.dataframe_estudiantes()
    mod = D.dataframe_modulo()

    # Columnas del Módulo Nacional (índices, ítems y contexto escolar) por
    # estudiante; desde el módulo LECT (superconjunto del MAT).
    cols_mod = [c for c in cfg.FEATURES_NUMERICOS if c in mod.columns]
    df = df.merge(mod[["IDSTUD"] + cols_mod].drop_duplicates("IDSTUD"),
                  on="IDSTUD", how="left", validate="one_to_one")

    # Variables derivadas
    df["provincia"] = df["ubi1_cen_esc"].astype(str)
    df["riesgo"] = (df["NL_VP1"] == "I").astype(int)
    df["superacion"] = df["NL_VP1"].isin(["III", "IV"]).astype(int)

    out = df[cfg.FEATURES].copy()
    for c in cfg.FEATURES:
        if c in cfg.FEATURES_CATEGORICOS:
            out[c] = out[c].astype("category")
    out["NL_VP1"] = df["NL_VP1"].astype(str)
    out["REPC"] = df["REPC"]
    out["PL3"] = df["PL3"]
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