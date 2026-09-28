# -*- coding: utf-8 -*-
"""Gráficos informativos de calidad de publicación (cliente, artículo, software).

Todos se exportan en PNG (300 dpi), SVG y PDF desde `artefactos/reporte/`.
Paleta derivada del diseño web (variante editorial, azul claro).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

from . import config as cfg

C = cfg.PALETA
_EXT = (".png", ".svg", ".pdf")


def configurar_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": C["blanco"],
        "axes.facecolor": C["blanco"],
        "axes.edgecolor": C["gris_claro"],
        "axes.labelcolor": C["marino"],
        "text.color": C["marino"],
        "xtick.color": C["gris"],
        "ytick.color": C["gris"],
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "grid.color": C["gris_claro"],
        "grid.alpha": 0.5,
        "legend.frameon": False,
    })


def guardar_fig(fig, carpeta, nombre):
    """Guarda PNG(300dpi) + SVG + PDF con `bbox_inches='tight'`."""
    os.makedirs(carpeta, exist_ok=True)
    for ext in _EXT:
        fig.savefig(os.path.join(carpeta, nombre + ext),
                    dpi=300 if ext == ".png" else None,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _barras(df, x, y, etiqueta_y, titulo, ordenar=True, tamaño=(7.5, 4.5)):
    fig, ax = plt.subplots(figsize=tamaño)
    d = df.sort_values(y) if ordenar else df
    colores = [C["azul_claro"]] * len(d)
    colores[-1] = C["azul"]
    bars = ax.bar(d[x], d[y], color=colores, zorder=3)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=35)
    for b, v in zip(bars, d[y]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=C["gris"])
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def grafico_comparativa(df, metrica, carpeta, nombre, titulo=None):
    """Barras de una métrica por algoritmo (media sobre folds).

    df: DataFrame con columnas 'algoritmo' y metrica (+ error si 'de').
    """
    d = df[["algoritmo", metrica]].rename(columns={metrica: "valor"})
    fig = _barras(d, "algoritmo", "valor", metrica,
                  titulo or f"Comparativa de algoritmos · {metrica}")
    guardar_fig(fig, carpeta, nombre)


def grafico_matriz_confusion(y_true, y_pred, clases, carpeta, nombre, titulo):
    """Matriz de confusión anotada (tabla de clasificación estilo Weka)."""
    clases = [str(c) for c in clases]
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(clases)), labels=clases, rotation=30, ha="right")
    ax.set_yticks(range(len(clases)), labels=clases)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real")
    ax.set_title(titulo)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n({cmn[i, j]:.0%})",
                    ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else C["marino"],
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_roc(y_true, y_prob, clases, carpeta, nombre, titulo):
    """Curvas ROC (1-vs-rest) para clasificación multi/binaria."""
    clases = [str(c) for c in clases]
    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    yb = np.asarray(y_true).astype(str)
    for i, cl in enumerate(clases):
        binario = (yb == cl).astype(int)
        if len(np.unique(binario)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binario, y_prob[:, i])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{cl} (AUC = {a:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_curva_aprendizaje(curva, carpeta, nombre, titulo):
    """Curva de aprendizaje: métricas medias ± DE vs tamaño de entrenamiento."""
    d = pd.DataFrame(curva)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(d["n_entrenamiento"], d["train_media"], yerr=d["train_de"],
                label="Entrenamiento", color=C["azul"], marker="o", capsize=3)
    ax.errorbar(d["n_entrenamiento"], d["test_media"], yerr=d["test_de"],
                label="Validación (CV)", color=C["rojo"], marker="o", capsize=3)
    ax.set_xlabel("Filas de entrenamiento"); ax.set_ylabel("Puntaje")
    ax.set_title(titulo); ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_importancia(imp_df, carpeta, nombre, titulo, top=14):
    """Importancia de atributos del modelo ganador (barras horizontales)."""
    d = imp_df.sort_values("importancia").tail(top)
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.35 * len(d))))
    ax.barh(d["variable"], d["importancia"], color=C["azul_claro"], zorder=3)
    for y, v in zip(range(len(d)), d["importancia"]):
        ax.text(v, y, f"{v:.3f}", va="center", fontsize=7, color=C["gris"])
    ax.set_xlabel("Importancia")
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_correlaciones(nombres, corr, carpeta, nombre, titulo):
    """Heatmap de correlaciones lineales entre atributos seleccionados."""
    corr = np.asarray(corr)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    tick = range(len(nombres))
    ax.set_xticks(tick, labels=nombres, rotation=90, fontsize=6)
    ax.set_yticks(tick, labels=nombres, fontsize=6)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_ir(y, carpeta, nombre, titulo):
    """Distribución de clases / desbalanceo (IR) de la variable objetivo."""
    vc = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([str(v) for v in vc.index], vc.values, color=C["azul_claro"],
           edgecolor=C["azul"], zorder=3)
    ax.set_ylabel("Instancias")
    ax.set_title(titulo)
    for x, v in zip(range(len(vc)), vc.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_scatter_regresion(y_true, y_pred, carpeta, nombre, titulo, r2=None):
    """Real vs predicho para regresión."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=C["azul_claro"],
               edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, ls="--", color=C["rojo"], lw=1.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observado"); ax.set_ylabel("Predicho")
    ax.set_title(titulo + (f"\nR² = {r2:.3f}" if r2 is not None else ""))
    ax.grid(alpha=0.4)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_residuos(y_true, y_pred, carpeta, nombre, titulo):
    """Residuos (observado - predicho) vs predicho."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    res = np.asarray(y_pred) - np.asarray(y_true)
    ax.scatter(y_pred, res, s=8, alpha=0.35, color=C["azul_claro"])
    ax.axhline(0, color=C["rojo"], lw=1.3)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Residuo (pred - obs)")
    ax.set_title(titulo)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)


def grafico_comparativa_balanceo(df, carpeta, nombre):
    """Barras agrupadas: F1 con y sin balanceo por algoritmo."""
    piv = df.pivot_table(index="algoritmo", columns="balanceo",
                         values="f1_macro", aggfunc="mean")
    piv = piv.reindex(sorted(piv.columns, key=lambda c: (c != "none", c)), axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    piv.plot(kind="bar", ax=ax, color=[C["azul_claro"], C["rojo"], C["ambar"]],
             width=0.82, zorder=3)
    ax.set_ylabel("F1 macro")
    ax.set_title("Efecto del balanceo (estilo Weka)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.4)
    ax.legend(title="Balanceo")
    fig.tight_layout()
    guardar_fig(fig, carpeta, nombre)