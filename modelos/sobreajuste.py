# -*- coding: utf-8 -*-
"""Control de overfitting / underfitting del experimento.

Recursos:
- Hold-out reservado (estratificado si clasificación, 20% por defecto) que
  NUNCA participa en la selección de modelos -> estimación final imparcial.
- Brecha entrenamiento vs validación por algoritmo: si es grande -> alerta de
  sobreajuste; si ambas métricas son bajas -> alerta de subajuste.
- Curvas de aprendizaje (puntaje vs tamaño de entrenamiento) del modelo final.
"""
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve

from . import config as cfg


def dividir_holdout(X, y, task_type, frac=cfg.HOLDOUT_FRACCION, seed=cfg.SEED):
    """Reserva de test (devuelve Xtr, Xte, ytr, yte). Estratificada si clasificación."""
    stratify = y if task_type == "clasificacion" else None
    return train_test_split(X, y, test_size=frac, random_state=seed,
                            stratify=stratify)


def detectar_sobreajuste(metrica_train, metrica_test, umbral=cfg.BREAK_OVERFIT,
                         piso=0.45):
    """Diagnostica la brecha train-test de una métrica (mayor = mejor).

    Devuelve el estado textual y numérico para el reporte.
    """
    brecha = float(metrica_train - metrica_test)
    if brecha > umbral:
        estado, detalle = "sobreajuste", (
            f"gran brecha entre entrenamiento ({metrica_train:.3f}) y "
            f"validación ({metrica_test:.3f})")
    elif metrica_test < piso:
        estado, detalle = "underfitting", (
            f"el modelo no aprendió (validación {metrica_test:.3f} < {piso})")
    else:
        estado, detalle = "ok", (
            f"brecha controlada ({brecha:+.3f}) con validación {metrica_test:.3f}")
    return {
        "estado": estado,
        "detalle": detalle,
        "brecha": round(brecha, 4),
        "metrica_train": round(float(metrica_train), 4),
        "metrica_test": round(float(metrica_test), 4),
    }


def curva_aprendizaje(estimador, X, y, task_type, scoring,
                      cv=cfg.CV_FOLDS_SELECCION, sizes=None,
                      n_jobs=-1, seed=cfg.SEED):
    """Resultado de `learning_curve` embellecido (medias y ±DE)."""
    if sizes is None:
        sizes = np.linspace(0.1, 1.0, 6)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sizes_ef, train, test = learning_curve(
            estimador, X, y, cv=cv, scoring=scoring, train_sizes=sizes,
            n_jobs=n_jobs, random_state=seed)
    out = []
    for i, s in enumerate(sizes_ef):
        out.append({
            "n_entrenamiento": int(s),
            "train_media": float(train[i].mean()),
            "train_de": float(train[i].std()),
            "test_media": float(test[i].mean()),
            "test_de": float(test[i].std()),
        })
    return out


def resumen_curva(curva):
    """Último punto de la curva (todo el set) para comparar con hold-out."""
    ult = curva[-1]
    return {
        "n_entrenamiento": ult["n_entrenamiento"],
        "train_media": ult["train_media"],
        "test_media": ult["test_media"],
    }