# -*- coding: utf-8 -*-
"""Análisis estadístico del ERCE 2019 – Cuba, 3er grado.

Metodología:
- Puntajes: media de los 5 valores plausibles por estudiante (estimación puntual).
- Medias ponderadas con el peso de muestreo WSEN (escala LLECE).
- Errores estándar e IC con el método de réplicas BRR (Fay, factor 0.5),
  combinando la varianza de muestreo con la imputación (valores plausibles).
- Pruebas: t de Student (Welch), ANOVA de una vía, chi-cuadrado, Pearson.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

FAY = 0.5


PV_LECT = [f"VP{i}" for i in range(1, 6)]
PV_MAT = [f"MVP{i}" for i in range(1, 6)]


def _cols_brr(df):
    return [f"BRR{i}" for i in range(1, 101) if f"BRR{i}" in df.columns]


def media_ponderada(df, score, peso="WSEN"):
    """Media ponderada del puntaje (media simple de PVs ya resuelta por estudiante)."""
    x = df[score].to_numpy(dtype=float)
    w = df[peso].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan, 0
    return float(np.average(x[mask], weights=w[mask])), float(mask.sum())


def media_ponderada_pv(df, pv_cols, peso="WSEN"):
    """Media ponderada estimada a partir de valores plausibles (agrega los 5)."""
    medias = []
    n = 0
    for c in pv_cols:
        m, n = media_ponderada(df, c, peso)
        medias.append(m)
    return float(np.nanmean(medias)), int(n)


def errores_estandar_brr(df, pv_cols, peso="WSEN", estadistico=None):
    """Error estándar por réplicas BRR (Fay) para la media de valores plausibles.

    estadistico(df) -> float (default: media ponderada de los PV).
    """
    if estadistico is None:
        estadistico = lambda d: media_ponderada_pv(d, pv_cols, peso)[0]

    brr = _cols_brr(df)
    T0 = estadistico(df)

    # Varianza de muestreo: promedio en los PV de las réplicas
    # Medias de réplica por PV
    replicas = []
    for pv in pv_cols:
        r_pv = []
        for b in brr:
            d = df[[pv, b]]
            x = d[pv].to_numpy(float)
            w = d[b].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
            r_pv.append(float(np.average(x[mask], weights=w[mask])) if mask.sum() else np.nan)
        replicas.append(r_pv)
    replicas = np.array(replicas)  # (5, 100)

    var_muestreo = np.nanmean(((replicas - replicas.mean(axis=1, keepdims=True)) ** 2) / ((1 - FAY) ** 2))

    # Varianza de imputación (valores plausibles)
    medias_pv = [media_ponderada(df, c, peso)[0] for c in pv_cols]
    var_pv = np.var(medias_pv, ddof=1)
    var_imputacion = (1 + 1 / len(pv_cols)) * var_pv

    var_total = var_muestreo + var_imputacion
    return float(np.sqrt(var_total)), float(T0), var_muestreo, var_imputacion


def tabla_resumen(df, score, grupo, peso="WSEN"):
    """Resumen ponderado del puntaje por grupo: n, media, ee, IC95, min, max."""
    pv = PV_MAT if score == "PM3" else PV_LECT
    filas = []
    for g, sub in df.groupby(grupo, dropna=True):
        if len(sub) < 2:
            continue
        med, n = media_ponderada(sub, score, peso)
        if not np.isfinite(med):
            continue
        try:
            ee, *_ = errores_estandar_brr(sub, pv, peso)
        except Exception:
            ee = float(sub[score].std(ddof=1) / np.sqrt(n))
        filas.append({
            grupo: g,
            "n": int(n),
            "Media": round(med, 1),
            "EE": round(ee, 2),
            "IC95_inf": round(med - 1.96 * ee, 1),
            "IC95_sup": round(med + 1.96 * ee, 1),
            "DE": round(float(sub[score].std(ddof=1)), 1),
            "Mínimo": round(float(sub[score].min()), 1),
            "Máximo": round(float(sub[score].max()), 1),
        })
    return pd.DataFrame(filas).sort_values("Media", ascending=False)


def prueba_t_grupos(df, score, grupo, peso="WSEN"):
    """t de Student (Welch) entre los 2 grupos más frecuentes de `grupo`."""
    cats = df[grupo].value_counts(dropna=True)
    cats = cats[cats.index != "Sin dato"]
    if len(cats) < 2:
        return None
    top2 = cats.index[:2].tolist()
    a = df[df[grupo] == top2[0]][score].dropna()
    b = df[df[grupo] == top2[1]][score].dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {
        "grupo": grupo,
        "cat_a": top2[0], "cat_b": top2[1],
        "media_a": round(a.mean(), 1), "media_b": round(b.mean(), 1),
        "diferencia": round(a.mean() - b.mean(), 1),
        "t": round(t, 3), "p": p,
        "cohen_d": round(d, 3),
        "significativo": p < 0.05,
    }


def prueba_anova(df, score, grupo):
    """ANOVA de una vía (fórmulas tipo III aproximadas, sin ponderar)."""
    cats = df[grupo].value_counts(dropna=True)
    cats = cats[cats.index != "Sin dato"]
    grupos = [df[df[grupo] == g][score].dropna().to_numpy() for g in cats.index if (df[df[grupo] == g][score].dropna().size > 1)]
    if len(grupos) < 2:
        return None
    F, p = stats.f_oneway(*grupos)
    # eta^2 = SSB/SST
    all_x = np.concatenate(grupos)
    ssb = sum(len(g) * (g.mean() - all_x.mean()) ** 2 for g in grupos)
    sst = sum(((g - all_x.mean()) ** 2).sum() for g in grupos)
    eta = ssb / sst if sst > 0 else 0
    return {"F": round(F, 3), "p": p, "eta2": round(eta, 3), "n_grupos": len(grupos)}


def prueba_chi2(df, col1, col2):
    """Chi-cuadrado de independencia entre dos variables categóricas."""
    tabla = pd.crosstab(df[col1].fillna("Sin dato"), df[col2].fillna("Sin dato"))
    chi2, p, dof, _ = stats.chi2_contingency(tabla)
    # V de Cramér
    n = tabla.to_numpy().sum()
    min_dim = min(tabla.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if (n > 0 and min_dim > 0) else np.nan
    return {"chi2": round(chi2, 3), "p": p, "dof": int(dof), "v_cramer": round(v, 3)}


def correlaciones(df, variables, score="PL3"):
    """Correlación de Pearson de cada variable con el puntaje."""
    out = []
    for v in variables:
        if v not in df.columns:
            continue
        sub = df[[v, score]].dropna()
        if len(sub) < 3 or sub[v].nunique() < 2:
            continue
        r, p = stats.pearsonr(sub[v], sub[score])
        out.append({"variable": v, "r": round(r, 3), "p": p, "n": len(sub)})
    return pd.DataFrame(out).sort_values("r", ascending=False)


def cuartiles_isect(df, score, n=4):
    """Corta el índice socioeconómico en cuartiles y agrega medias ponderadas."""
    sub = df[df["ISECF"].notna()].copy()
    try:
        sub["q"] = pd.qcut(sub["ISECF"], n, labels=[f"Q{i}" for i in range(1, n + 1)], duplicates="drop")
    except ValueError:
        sub["q"] = pd.cut(sub["ISECF"], n, labels=[f"Q{i}" for i in range(1, n + 1)])
    out = []
    for q, g in sub.groupby("q", observed=True):
        med, nq = media_ponderada(g, score, "WSEN")
        out.append({"Cuartil": q, "n": int(nq), "Media": round(med, 1)})
    return pd.DataFrame(out)


def media_por_nivel(df, nivel_col, score):
    """Media ponderada del puntaje por nivel de desempeño (NL_VP*)."""
    out = []
    for nivel, g in df.groupby(nivel_col, dropna=True):
        med, n = media_ponderada(g, score, "WSEN")
        out.append({"Nivel": nivel, "n": int(n), "Media": round(med, 1)})
    return pd.DataFrame(out).sort_values("Media", ascending=False)


def resumen_nacional(df):
    """Métricas nacionales para la portada."""
    out = {}
    for score, nombre, pv in [("PL3", "Lectura", PV_LECT), ("PM3", "Matemática", PV_MAT)]:
        med, n = media_ponderada(df, score, "WSEN")
        de = float(df[score].std(ddof=1))
        out[nombre] = {"media": round(med, 1), "de": round(de, 1), "n": int(n)}
        try:
            ee, *_ = errores_estandar_brr(df, pv, "WSEN")
            out[nombre]["ee"] = round(ee, 2)
        except Exception:
            out[nombre]["ee"] = None
    return out
