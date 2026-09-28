# -*- coding: utf-8 -*-
"""Test no paramétrico de Friedman + post-hoc de Nemenyi (Experimenter de Weka).

Entrada: métricas por fold de cada algoritmo (matriz folds x algoritmos).
- Friedman: ¿existen diferencias estadísticas entre los algoritmos?
- Rango medio por algoritmo (menor = mejor).
- Diferencia crítica (CD) por Nemenyi y diagrama CD para el artículo.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range


def tabla_por_algoritmo(metricas, algoritmo_col="algoritmo"):
    """Convierte una tabla (filas=fold x config, con columna de métrica) en
    matriz `folds x algorithms` para Friedman, agregando por fold.

    metricas: pandas.DataFrame con columnas ['fold', algoritmo_col, 'metrica'].
    Devuelve dict {algoritmo: np.array(por fold)} con el MISMO orden de folds.
    """
    df = metricas.copy()
    piv = df.pivot_table(index="fold", columns=algoritmo_col, values="metrica",
                         aggfunc="mean")
    return {c: piv[c].to_numpy(dtype=float) for c in piv.columns}


def rangos_medios(tabla):
    """Rango medio por algoritmo (promedio de rangos en cada fold)."""
    algos = list(tabla)
    mat = np.array([tabla[a] for a in algos])  # k x folds
    n = mat.shape[1]
    ranked = np.apply_along_axis(lambda v: np.argsort(np.argsort(-v)) + 1, 0, mat)
    return {a: float(ranked[i].mean()) for i, a in enumerate(algos)}


def test_friedman(tabla):
    """Friedman sobre los algoritmos (bloques = folds). Devuelve stat, p, rangos."""
    algos = list(tabla)
    arrays = [tabla[a] for a in algos]
    stat, p = friedmanchisquare(*arrays)
    return {"algoritmos": algos, "stat": float(stat), "p": float(p),
            "n_folds": len(arrays[0]), "rangos": rangos_medios(tabla),
            "k": len(algos)}


def diferencia_critica(n_folds, k, alpha=0.05):
    """Diferencia crítica de Nemenyi: q_alpha * sqrt(k(k+1)/(6n))."""
    if k < 2 or n_folds < 1:
        return float("nan")
    q = studentized_range.ppf(1 - alpha, k, np.inf)
    cd = q / np.sqrt(2.0) * np.sqrt(k * (k + 1) / (6.0 * n_folds))
    return float(cd)


def posthoc_nemenyi(tabla):
    """Matriz de p-valores Nemenyi (via scikit-posthocs), longitudes iguales."""
    im = {a: np.asarray(v) for a, v in tabla.items()}
    n_min = min(len(v) for v in im.values())
    mat = pd.DataFrame({a: v[:n_min] for a, v in im.items()})
    import scikit_posthocs as sp
    return sp.posthoc_nemenyi_friedman(mat)


def ranking_final(tabla, alpha=0.05):
    """Reporte completo: Friedman, CD, Nemenyi y tabla de rangos."""
    f = test_friedman(tabla)
    cd = diferencia_critica(f["n_folds"], f["k"], alpha)
    rangos = pd.DataFrame([
        {"#": i + 1, "algoritmo": a, "rango_medio": r}
        for i, (a, r) in enumerate(
            sorted(f["rangos"].items(), key=lambda kv: kv[1]))
    ])
    out = {
        "friedman_stat": f["stat"],
        "friedman_p": f["p"],
        "significativo": bool(f["p"] < alpha),
        "cd_nemenyi": cd,
        "n_folds": f["n_folds"],
        "ranking": rangos,
        "nemenyi": posthoc_nemenyi(tabla),
    }
    return out


def grafico_cd(rangos_df, cd, ax=None):
    """Diagrama de diferencia crítica (CD) tipo Demsar/2006.

    Eje horizontal = escala de rangos (1..k, menor = mejor). Cada algoritmo se
    sitúa en su rango medio; el intervalo [rango-CD/2, rango+CD/2] se dibuja y
    los algoritmos cuyos intervalos se solapan se unen con una línea gruesa.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, max(2.4, 0.42 * len(rangos_df))))
    df = rangos_df.sort_values("rango_medio").reset_index(drop=True)
    k = len(df)
    pos = df["rango_medio"].to_numpy(float)
    algos = df["algoritmo"].to_numpy()
    ystep = 1.0

    for i, (algo, r) in enumerate(zip(algos, pos)):
        y = -(i * ystep)
        ax.hlines(y, max(r - cd / 2, 0.8), min(r + cd / 2, k + 0.2),
                  color="0.35", lw=3)
        ax.text(r, y + 0.32, algo, ha="center", va="bottom", fontsize=9)
        ax.text(r, y - 0.22, f"{r:.2f}", ha="center", va="top", fontsize=8)

    for i in range(k):
        for j in range(i + 1, k):
            if abs(pos[i] - pos[j]) <= cd:
                ax.plot([pos[i], pos[j]], [-(i * ystep), -(j * ystep)],
                        color="0.1", lw=2.2, alpha=0.6)

    ax.set_xlim(0.6, k + 0.4)
    ax.set_ylim(-(k - 1) * ystep - 1.0, 1.0)
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel(f"Rango medio · Diferencia crítica = {cd:.2f}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.invert_xaxis()
    ax.set_title("Diagrama de diferencia crítica (Nemenyi)", fontsize=11)
    try:
        ax.figure.tight_layout()
    except Exception:
        pass
    return ax.figure