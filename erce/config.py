# -*- coding: utf-8 -*-
"""Rutas y constantes del proyecto ERCE 2019 – Cuba, 3er grado."""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS_DIR = os.path.join(BASE_DIR, "07.ERCE-2019-FINAL")

LLECE = os.path.join(DATOS_DIR, "LLECE-BBDD FINALES")
ICCP = os.path.join(DATOS_DIR, "ICCP-BBDD FINALES")
REPORTES = os.path.join(DATOS_DIR, "LLECE-REPORTES FINALES")
CODEBOOK = os.path.join(BASE_DIR, "2. Libros de código")

ARCHIVOS = {
    "pl": os.path.join(LLECE, "PL3.csv"),
    "pm": os.path.join(LLECE, "PM3.csv"),
    "fa": os.path.join(LLECE, "FA_A3.csv"),
    "qp": os.path.join(LLECE, "QP3.csv"),
    "mod_lect": os.path.join(ICCP, "CUBA3_LECT_MOD_NAC_PRUEBA.csv"),
    "mod_mat": os.path.join(ICCP, "CUBA3_MAT_MODULO_NACIONAL.csv"),
    "prom_lect": os.path.join(ICCP, "CUBA3_LECT_PROMEDIOS_ESCUELAS_TODAS.csv"),
    "prom_mat": os.path.join(ICCP, "CUBA3_MAT_PROMEDIOS_ESCUELAS_TODAS.csv"),
    "coord": os.path.join(ICCP, "CUBA3_Coordenadas_escuelas_con_NA.csv"),
}

ENCODINGS = ["utf-8", "latin-1", "cp1252", "ISO-8859-1"]

VARIABLES_PV = ["VP1", "VP2", "VP3", "VP4", "VP5"]

# Etiquetas de las provincias / categorías relevantes
AREA_ESC_LABEL = {1: "Urbana", 2: "Rural"}
GENERO_LABEL = {1: "Niño", 2: "Niña"}
SEX_LABEL = {1: "Niño", 2: "Niña", 0: "Niño"}  # 0 es inconsistente en el dataset
SEX_LABEL_FA = {0.0: "Niño", 1.0: "Niña"}
GRUPO_LABEL = {
    1: "Nivel I", 2: "Nivel II", 3: "Nivel III", 4: "Nivel IV",
}


def read_csv(path, low_memory=False, **kwargs):
    """Lee un CSV probando distintas codificaciones."""
    errores = []
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=low_memory, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            errores.append(f"{enc}: {e}")
    raise ValueError(f"No se pudo leer {path}: {errores}")
