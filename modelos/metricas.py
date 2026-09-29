# -*- coding: utf-8 -*-
"""Métricas de evaluación fieles al reporte de Weka.

Clasificación (Explorer de Weka):
- Correctly/Incorrectly Classified Instances (exactitud y su complemento).
- Kappa statistic.
- Por clase: TP Rate, FP Rate, Precision, Recall, F-Measure, MCC,
  ROC Area y PRC Area (curvas 1-vs-rest).
- Matriz de confusión.

Regresión (Weka):
- Correlation coefficient (r), MAE, RMSE, RAE (rel. abs. error) y RRSE.
- R² (varianza explicada).

Además, métricas por fold (para el Experimenter y el test de Friedman).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support,
                             r2_score, roc_auc_score)
from scipy.stats import pearsonr


# ------------------------------------------------------------- clasificación
def _tasa_tp_fp(y_true, y_pred, clase, clases):
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    i = clases.index(clase)
    tp, fn = cm[i, i], cm[i, :].sum() - cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    tn = cm.sum() - (tp + fp + fn)
    return tp / (tp + fn) if (tp + fn) else 0.0, fp / (fp + tn) if (fp + tn) else 0.0


def _normalizar_probas(y_prob, clases):
    """Deja `y_prob` con tantas columnas como clases (binario robusto).

    Algunos modelos en folds muy desbalanceados (p.ej. clf_repitencia) devuelven
    una sola columna de probabilidad; se completa como `[1-p, p]` (la columna
    única se asume de la clase positiva/última), igual que el fallback de
    decisión en `_probas`.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    k = len(clases)
    if y_prob.ndim == 1:
        return np.c_[1 - y_prob, y_prob] if k == 2 else np.ones((len(y_prob), k)) / k
    if y_prob.shape[1] == k:
        return y_prob
    if y_prob.shape[1] == 1 and k == 2:
        p = y_prob[:, 0]
        return np.c_[1 - p, p]
    return np.ones((y_prob.shape[0], k)) / k


def _roc_area(y_true_bin, prob_clase):
    if len(np.unique(y_true_bin)) < 2:
        return 0.0
    return roc_auc_score(y_true_bin, prob_clase)


def calcular_metricas_clasificacion(y_true, y_pred, y_prob, clases):
    """Devuelve un dict con el resumen Weka (kappa, exactitud), por clase y matriz."""
    # `clases` y los objetivos deben compartir representación: los objetivos
    # numéricos (float/int) no casan con unas `clases` en str y viceversa, y sin
    # alinear ambos lados confusion_matrix, las tasas por clase y MCC quedan
    # todo en cero. Se normaliza todo a str.
    clases = [str(c) for c in clases]
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    if y_prob is not None:
        y_prob = _normalizar_probas(y_prob, clases)
    exact = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    prec, rec, f1, n_clase = precision_recall_fscore_support(
        y_true, y_pred, labels=clases, zero_division=0)

    filas = []
    for i, clase in enumerate(clases):
        tpr, fpr = _tasa_tp_fp(y_true, y_pred, clase, clases)
        mcc = matthews_corrcoef(
            np.array(y_true) == clase,
            np.array(y_pred) == clase) if len(np.unique(np.array(y_true) == clase)) > 1 else 0.0
        ybin = (np.asarray(y_true) == clase).astype(int)
        prob_c = y_prob[:, i]
        roc = _roc_area(ybin, prob_c)
        prc = average_precision_score(ybin, prob_c) if len(np.unique(ybin)) > 1 else 0.0
        filas.append({
            "Clase": str(clase),
            "TP_Rate": round(tpr, 4),
            "FP_Rate": round(fpr, 4),
            "Precision": round(prec[i], 4),
            "Recall": round(rec[i], 4),
            "F_Measure": round(f1[i], 4),
            "MCC": round(mcc, 4),
            "ROC_Area": round(roc, 4),
            "PRC_Area": round(prc, 4),
            "n": int(n_clase[i]),
        })
    por_clase = pd.DataFrame(filas)

    f1_macro = round(float(np.mean([f1[i] for i in range(len(clases))])), 4)
    if y_prob is None:
        auc = 0.0
    elif len(clases) > 2:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    else:
        auc = (_roc_area(y_true, y_prob[:, 1])
               if len(np.unique(y_true)) > 1 else 0.0)
    return {
        "exactitud": round(float(exact), 4),
        "error": round(float(1 - exact), 4),
        "kappa": round(float(kappa), 4),
        "f1_macro": f1_macro,
        "auc_macro": round(float(auc), 4),
        "n": int(len(y_true)),
        "por_clase": por_clase,
        "matriz": cm,
        "clases": clases,
    }


def metricas_fold_clasif(y_true, y_pred, y_prob, clases):
    """Métricas compactas por fold, para el Friedman y la comparativa."""
    m = calcular_metricas_clasificacion(y_true, y_pred, y_prob, clases)
    return {
        "exactitud": m["exactitud"],
        "f1_macro": m["f1_macro"],
        "auc_macro": m["auc_macro"],
        "kappa": m["kappa"],
    }


# ------------------------------------------------------------- regresión
def calcular_metricas_regresion(y_true, y_pred):
    """r (correlación), MAE, RMSE, RAE, RRSE y R² (estilo Weka + R²)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = y_true[np.isfinite(y_pred)]
    y_pred = y_pred[np.isfinite(y_pred)]
    n = len(y_true)
    if n == 0:
        return {k: np.nan for k in ("r", "mae", "rmse", "rae", "rrse", "r2", "n")}
    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    den = np.abs(y_true - y_true.mean()).sum() or 1.0
    rae = np.abs(y_true - y_pred).sum() / den
    den2 = ((y_true - y_true.mean()) ** 2).sum() or 1.0
    rrse = np.sqrt(((y_pred - y_true) ** 2).sum() / den2)
    return {
        "r": round(float(r), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "rae": round(float(rae), 4),
        "rrse": round(float(rrse), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(n),
    }


def metricas_fold_regresion(y_true, y_pred):
    m = calcular_metricas_regresion(y_true, y_pred)
    return {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]}


# ------------------------------------------------------------- línea base
# Sin referencia, un f1_macro de 0.49 o un r2 negativo se leen como desempeño
# cuando en realidad son lo que consigue adivinar siempre lo más frecuente (o la
# media). Estos dos helpers calculan esa referencia sobre el mismo holdout para
# que cada métrica se pueda leer contra su suelo.
def linea_base_clasificacion(y_true):
    """Predicción constante de la clase mayoritaria (kappa 0 y AUC 0.5)."""
    y_true = np.asarray([str(v) for v in y_true])
    clases, conteo = np.unique(y_true, return_counts=True)
    y_pred = np.full(len(y_true), clases[np.argmax(conteo)])
    p = np.column_stack([(y_pred == c).astype(float) for c in clases])
    return calcular_metricas_clasificacion(y_true, y_pred, p, list(clases))


def linea_base_regresion(y_true):
    """Predicción constante igual a la media del objetivo."""
    y_true = np.asarray(y_true, dtype=float)
    y_true = y_true[np.isfinite(y_true)]
    y_pred = np.full(len(y_true), y_true.mean())
    import warnings

    from scipy.stats import ConstantInputWarning
    with warnings.catch_warnings():
        # r no está definido para una predicción constante: es el suelo (0).
        warnings.simplefilter("ignore", ConstantInputWarning)
        return calcular_metricas_regresion(y_true, y_pred)