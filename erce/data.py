# -*- coding: utf-8 -*-
"""Carga y preparación de los datos del ERCE 2019 – Cuba, 3er grado.

Convenciones del estudio:
- Los puntajes se reportan mediante 5 valores plausibles (VP1..VP5 / MAT_1..MAT_5).
- Las estimaciones de medias usan los pesos de muestreo (WSEN).
- La escala oficial tiene media 700 y desviación 100 para la región.
"""
import numpy as np
import pandas as pd

from . import config as cfg


def _pv_mean(df, cols):
    """Media de los valores plausibles por fila (puntaje final del estudiante)."""
    return df[cols].mean(axis=1, skipna=True)


def cargar_pl3():
    return cfg.read_csv(cfg.ARCHIVOS["pl"])


def cargar_pm3():
    return cfg.read_csv(cfg.ARCHIVOS["pm"])


def cargar_fa():
    return cfg.read_csv(cfg.ARCHIVOS["fa"])


def cargar_qp():
    return cfg.read_csv(cfg.ARCHIVOS["qp"])


def cargar_mod_lect():
    return cfg.read_csv(cfg.ARCHIVOS["mod_lect"])


def cargar_mod_mat():
    return cfg.read_csv(cfg.ARCHIVOS["mod_mat"])


def cargar_promedios():
    lect = cfg.read_csv(cfg.ARCHIVOS["prom_lect"])
    mat = cfg.read_csv(cfg.ARCHIVOS["prom_mat"])
    coord = cfg.read_csv(cfg.ARCHIVOS["coord"])
    return lect, mat, coord


def dataframe_estudiantes():
    """DataFrame maestro a nivel estudiante con lectura, matemática y cuestionario.

    Une PL3 (lectura), PM3 (matemática) y FA_A3 (cuestionario) por IDSTUD,
    y añade la provincia/municipio desde el mapa de escuelas.
    """
    pl = cargar_pl3()
    pm = cargar_pm3()
    fa = cargar_fa()

    # Puntaje final: media de valores plausibles
    pl["PL3"] = _pv_mean(pl, cfg.VARIABLES_PV)
    pm["PM3"] = _pv_mean(pm, cfg.VARIABLES_PV)
    pm = pm.rename(columns={f"VP{i}": f"MVP{i}" for i in range(1, 6)})
    fa["LAN"] = _pv_mean(fa, [f"LAN_{i}" for i in range(1, 6)])
    fa["MAT"] = _pv_mean(fa, [f"MAT_{i}" for i in range(1, 6)])

    base = pl[["IDSTUD", "IDSCHOOL", "PL3", "NL_VP1", "SEX", "RURAL", "EDU", "WT", "WSEN", "GRUPO"]].copy()
    pv_lect = [f"VP{i}" for i in range(1, 6)]
    pv_mat = [f"MVP{i}" for i in range(1, 6)]
    brr_cols = [f"BRR{i}" for i in range(1, 101)]
    base = pd.concat([base, pl[pv_lect].reset_index(drop=True)], axis=1)
    base = pd.concat([base, pl[brr_cols].reset_index(drop=True)], axis=1)
    base = base.merge(
        pm[["IDSTUD", "PM3"] + pv_mat], on="IDSTUD", how="left", validate="one_to_one"
    )

    # Cuestionario de estudiante (contexto familiar, ISECF, hábitos, actitudes)
    fa_cols = [
        "IDSTUD", "ISECF", "EDAD", "PREE", "REPC", "AUSE", "ATRE", "LIBH",
        "TSTU", "AAEG3", "ORGEN", "EFMAT", "VIOES", "INVAP", "LAN", "MAT",
    ]
    # Ítems del cuestionario de estudiantes (E3IT*): se incorporan como features.
    # Se excluyen E3IT21_01/02: son los mismos conceptos que E3IT21_01/02_N del
    # Módulo Nacional (0/1 normalizado) — el módulo manda para evitar doble conteo.
    fa_cols += [c for c in fa.columns
                if c.upper().startswith("E3IT") and c not in {"E3IT21_01", "E3IT21_02"}]
    base = base.merge(fa[fa_cols], on="IDSTUD", how="left", validate="one_to_one")

    # Territorio (provincia y municipio) desde el mapa de escuelas
    lect, _, _ = cargar_promedios()
    esc_map = lect[["IDSCHOOL", "ubi1_cen_esc", "ubi2_cen_esc"]].drop_duplicates("IDSCHOOL")
    base = base.merge(esc_map, on="IDSCHOOL", how="left", validate="many_to_one")

    # Clasificaciones legibles
    base["genero"] = base["SEX"].map({1.0: "Niña", 0.0: "Niño"}).fillna("Sin dato")
    base["zona"] = base["RURAL"].map({1: "Rural", 0: "Urbana", 0.0: "Urbana"}).fillna("Sin dato")
    base["edu_padres"] = base["EDU"].map({1.0: "Terciaria", 0.0: "≤ Secundaria"}).fillna("Sin dato")

    return base


def dataframe_escuelas():
    """Promedios por escuela + coordenadas, para análisis territorial y mapa."""
    lect, mat, coord = cargar_promedios()
    esc = lect[["IDSCHOOL", "nom_esc", "ubi1_cen_esc", "ubi2_cen_esc", "num_hab_esc", "PL3"]].copy()
    esc = esc.merge(
        mat[["IDSCHOOL", "PM3"]], on="IDSCHOOL", how="left", validate="one_to_one"
    )
    esc = esc.merge(coord[["IDSCHOOL", "geo_esc_x", "geo_esc_y"]], on="IDSCHOOL", how="left")
    esc["nom_esc"] = esc["nom_esc"].fillna("Escuela sin nombre")
    return esc


def dataframe_modulo():
    """Módulo Nacional: índices de calidad escolar/docente/directivo.

    El módulo de lectura contiene los valores plausibles y los índices; el de
    matemática solo los índices. Se deduplican los registros repetidos por
    IDSTUD (difieren solo en el profesor) y se incorpora el puntaje de
    matemática (PM3) desde el archivo LLECE PM3.csv.
    """
    ml = cargar_mod_lect()
    pm = cargar_pm3()

    # Deduplicar (un estudiante aparece 2 veces con distinto Id_prof)
    ml = ml.sort_values("Id_prof").drop_duplicates("IDSTUD", keep="first").copy()
    pm = cargar_pm3()
    pm = pm[["IDSTUD"] + cfg.VARIABLES_PV].copy()
    pm["PM3"] = _pv_mean(pm, cfg.VARIABLES_PV)

    idx_cols = [
        "IDSTUD", "IDSCHOOL", "IDCLASS",
        "Indice_preg", "Indice_refuerzo", "Indice_global",
        "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
        "Indice_calidad_clases_dir",
        "PPIT45_N", "sec_3", "can_prof_3", "prof_hom_3", "prof_muj_3",
        "num_hab_esc", "area_esc", "edad", "género",
        "VP1", "VP2", "VP3", "VP4", "VP5",
        "NL_VP1", "PS_STR_NEW", "GRUPO", "WT", "WSEN",
    ]
    # Ítems del Módulo Nacional como features: cuestionario del estudiante
    # (E3IT21_.._N), del docente (PPIT44_.., PPIT45, PPIT46_..) y del directivo
    # (DDIT41_..). Se incluyen cuando existen en el archivo.
    idx_cols += [c for c in ml.columns
                 if c.startswith(("E3IT21_", "PPIT44_", "PPIT45", "PPIT46_", "DDIT41_"))
                 and c not in idx_cols]
    df = ml[[c for c in idx_cols if c in ml.columns]].copy()
    df["PL3"] = _pv_mean(df, cfg.VARIABLES_PV)
    df = df.merge(pm, on="IDSTUD", how="left", validate="one_to_one")
    df["area_esc"] = df["area_esc"].map(cfg.AREA_ESC_LABEL).fillna("Sin dato")
    df["género"] = df["género"].map(cfg.GENERO_LABEL).fillna("Sin dato")
    return df


def dataframe_modulo_mat():
    """Módulo Nacional de Matemática + contexto sociodemográfico ERCE.

    Fuente de features: el CSV adicionado por el cliente (módulo nacional de
    Matemática, prueba M3). Se deduplican los registros repetidos por IDSTUD
    (difieren solo en Id_prof) y se cruza con PM3.csv (puntaje y nivel de
    Matemática) y con FA_A3.csv (contexto sociodemográfico y repitencia).
    """
    mm = cargar_mod_mat()
    mm = mm.sort_values("Id_prof").drop_duplicates("IDSTUD", keep="first").copy()

    pm = cargar_pm3()
    pm["PM3"] = _pv_mean(pm, cfg.VARIABLES_PV)
    pm = pm[["IDSTUD", "PM3", "NL_VP1", "GRUPO", "WT", "WSEN"]].copy()

    fa = cargar_fa()
    fa_cols = ["IDSTUD", "ISECF", "EDU", "PREE", "REPC", "AUSE", "ATRE",
               "LIBH", "TSTU"]
    fa = fa[fa_cols].copy()

    df = mm.merge(pm, on="IDSTUD", how="left", validate="one_to_one")
    df = df.merge(fa, on="IDSTUD", how="left", validate="one_to_one")

    df["género"] = df["género"].map(cfg.GENERO_LABEL).fillna("Sin dato")
    df["area_esc"] = df["area_esc"].map(cfg.AREA_ESC_LABEL).fillna("Sin dato")
    df["provincia"] = df["ubi1_cen_esc"].astype(str)
    return df


def dataframe_regional():
    """Datos regionales del informe ERCE 2019 (Cuba vs región), 3er grado."""
    return pd.DataFrame({
        "Área": ["Lectura", "Matemática"],
        "Cuba": [730, 751],
        "Región": [697, 698],
        "Diferencia": [33, 53],
    })
