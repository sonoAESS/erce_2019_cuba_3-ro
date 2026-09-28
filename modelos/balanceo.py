# -*- coding: utf-8 -*-
"""Balanceo de clases fiel al proceso de Weka.

Reglas sobre el IR de la última columna (clase):

- IR <= 1.2            -> sin balancear.
- 1.2 < IR <= 5        -> SMOTE.
- 5   < IR <= 9        -> SMOTE + RandomUnderSampler.
- IR > 9               -> SMOTEENN (SMOTE + Edited Nearest Neighbours).

Siempre se compara contra el modelo sin balancear; si balancear no mejora la
métrica, se conserva el modelo sin balancear (como filtrar/  no filtrar en Weka).
"""
import numpy as np
import pandas as pd

from . import config as cfg

NOMBRES_WEEKA = {
    "smote": "Smote (Weka)",
    "smote_rus": "Smote + SpreadSubsample (Weka)",
    "smoteenn": "SMOTEENN (Weka)",
}


def calcular_ir(y):
    """IR = instancias mayoritarias / instancias minoritarias (última columna)."""
    conteos = pd.Series(y).value_counts().to_numpy(dtype=float)
    if len(conteos) < 2:
        return None
    return float(conteos.max()) / float(conteos.min())


def elegir_balanceo(ir, multiclase=False):
    """Estrategia de balanceo según el IR (ver docstring del módulo)."""
    if ir is None or ir <= cfg.IR_BALANCEADO:
        return None
    if multiclase:
        return "smote"
    if ir <= cfg.IR_SMOTE_MAX:
        return "smote"
    if ir <= cfg.IR_SMOTE_RUS_MAX:
        return "smote_rus"
    return "smoteenn"


def crear_balanceador(nombre, seed=cfg.SEED):
    """Devuelve un sampler de imbalanced-learn (aplicado SOLO en entrenamiento)."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN
    from imblearn.under_sampling import RandomUnderSampler

    if nombre == "smote":
        return SMOTE(random_state=seed)
    if nombre == "smote_rus":
        return SMOTEENN(smote=SMOTE(random_state=seed), random_state=seed)
    if nombre == "smoteenn":
        return SMOTEENN(smote=SMOTE(random_state=seed), random_state=seed)
    if nombre == "random_under":
        return RandomUnderSampler(random_state=seed)
    return RandomOverSampler(random_state=seed)


def aplicar_balanceo(X, y, nombre, seed=cfg.SEED):
    """Aplica el balanceo (si `nombre` es None o 'none') y devuelve (Xr, yr)."""
    if nombre in (None, "none"):
        return X, y
    sampler = crear_balanceador(nombre, seed=seed)
    Xr, yr = sampler.fit_resample(X, y)
    return Xr, yr