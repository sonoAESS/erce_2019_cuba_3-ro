# -*- coding: utf-8 -*-
"""Figuras del reporte de resultados (`docs/reporte_resultados.md`).

Escriben en `docs/imagenes/` (versionado en git), porque `artefactos/` es
regenerable y está ignorado. Dos fuentes:

1. Cuatro gráficos resumen por tarea, calculados desde los `metadata.json`
   de `artefactos/<tarea>/` (hold-out, línea base, CV 10, estabilidad, IR).
2. Copia con nombre estable de las figuras por tarea que ya genera
   `modelos/plots.py` durante el entrenamiento (`artefactos/reporte/`),
   que son las que se calculan sobre el hold-out real.

Uso:
    entorno/bin/python scripts/imagenes_reporte.py

Requiere haber corrido antes `entorno/bin/python -m modelos.entrenar --tareas todas`.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modelos import config as cfg  # noqa: E402
from modelos.plots import configurar_matplotlib  # noqa: E402

C = cfg.PALETA
_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SALIDA = os.path.join(_RAIZ, "docs", "imagenes")
ARTEF = os.path.join(_RAIZ, "artefactos", "reporte")

TAREAS = ["clf_nivel", "clf_riesgo", "clf_superacion", "clf_repitencia"]
ETIQUETA = {
    "clf_nivel": "Nivel I–IV",
    "clf_riesgo": "Riesgo (nivel I)",
    "clf_superacion": "Alto desempeño",
    "clf_repitencia": "Repitencia",
    "reg_mat": "Puntaje de\nMatemática",
}
# figuras por tarea que ya existen en artefactos/reporte/ (nombre -> destino)
COPIAS = {
    ("clf_nivel", "roc"): "05_roc_clf_nivel.png",
    ("clf_repitencia", "roc"): "06_roc_clf_repitencia.png",
    ("clf_repitencia", "matriz_confusion"): "07_matriz_confusion_clf_repitencia.png",
    ("clf_nivel", "aprendizaje"): "08_aprendizaje_clf_nivel.png",
    ("reg_mat", "friedman_cd"): "09_friedman_cd_reg_mat.png",
    ("reg_mat", "importancia"): "10_importancia_reg_mat.png",
    ("reg_mat", "scatter"): "11_scatter_reg_mat.png",
}


def cargar(tarea):
    ruta = os.path.join(os.path.dirname(ARTEF), tarea, "metadata.json")
    if not os.path.exists(ruta):
        sys.exit(f"No existe {ruta}: corre primero `modelos.entrenar --tareas todas`.")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def guardar(fig, nombre, dpi=200):
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight", facecolor=C["blanco"])
    plt.close(fig)
    print(f"  {nombre}  ({os.path.getsize(ruta) // 1024} KB)")


def fig_modelo_vs_base(meta):
    """01 — hold-out de cada tarea frente a su línea base."""
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"width_ratios": [3.1, 1]})
    x = np.arange(len(TAREAS))
    ax = ejes[0]
    modelo = [meta[t]["estimacion"]["holdout"]["f1_macro"] for t in TAREAS]
    base = [meta[t]["estimacion"]["linea_base_holdout"]["f1_macro"] for t in TAREAS]
    ax.bar(x - 0.2, modelo, 0.38, color=C["azul"], label="Modelo", zorder=3)
    ax.bar(x + 0.2, base, 0.38, color=C["gris_claro"],
           label="Línea base (lo más frecuente)", zorder=3)
    for i, (m, b) in enumerate(zip(modelo, base)):
        ax.text(i, max(m, b) + 0.03, f"+{m - b:.3f}", ha="center",
                fontsize=9, color=C["verde"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETA[t] for t in TAREAS], fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_title("Clasificación · F1 macro (hold-out)", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    ax = ejes[1]
    r = meta["reg_mat"]["estimacion"]
    ax.bar([0 - 0.2], [r["holdout"]["r2"]], 0.38, color=C["azul"], zorder=3)
    ax.bar([0 + 0.2], [r["linea_base_holdout"]["r2"]], 0.38, color=C["gris_claro"], zorder=3)
    ax.text(0, r["holdout"]["r2"] + 0.03, f"+{r['holdout']['r2']:.3f}", ha="center",
            fontsize=9, color=C["verde"], fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels([ETIQUETA["reg_mat"]], fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_title("Regresión · R² (hold-out)", loc="left")
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    guardar(fig, "01_modelo_vs_base.png")


def fig_auc(meta):
    """02 — AUC de hold-out con la referencia del azar."""
    filas = sorted(((t, meta[t]["estimacion"]["holdout"]["auc_macro"]) for t in TAREAS),
                   key=lambda f: f[1])
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y = np.arange(len(filas))
    ax.barh(y, [v for _, v in filas], 0.55, color=C["azul_claro"], zorder=3)
    ax.axvline(0.5, color=C["rojo"], ls="--", lw=1.2, zorder=4)
    ax.text(0.515, len(filas) - 0.35, "azar = 0,5", color=C["rojo"], fontsize=9)
    for i, (_, v) in enumerate(filas):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=C["marino"])
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQUETA[t].replace("\n", " ") for t, _ in filas], fontsize=9)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("AUC de hold-out")
    ax.set_title("Capacidad de discriminación por tarea", loc="left")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    guardar(fig, "02_auc_holdout.png")


def fig_brecha(meta):
    """03 — hold-out, CV 10 y estabilidad: la brecha train–test."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(TAREAS))
    for i, t in enumerate(TAREAS):
        e = meta[t]["estimacion"]
        ax.plot([i - 0.18] * 2, [e["holdout"]["f1_macro"], e["cv_weka"]["f1_macro_media"]],
                color=C["gris_claro"], lw=1.4, zorder=2)
        ax.scatter(i - 0.18, e["holdout"]["f1_macro"], s=70, color=C["azul"], zorder=3,
                   label="hold-out" if i == 0 else None)
        ax.scatter(i - 0.18, e["cv_weka"]["f1_macro_media"], s=55, marker="s",
                   color=C["azul_claro"], zorder=3, label="CV 10-fold" if i == 0 else None)
        ax.scatter(i + 0.18, e["estabilidad_3x5cv"]["media"], s=55, marker="^",
                   color=C["ambar"], zorder=3, label="estabilidad 3×5" if i == 0 else None)
        ax.text(i + 0.18, e["estabilidad_3x5cv"]["media"] + 0.035, e["brecha"]["estado"],
                ha="center", fontsize=8, color=C["gris"])
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETA[t] for t in TAREAS], fontsize=9)
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 0.88)
    ax.set_title("Hold-out, CV y estabilidad: la brecha train–test por tarea", loc="left")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    guardar(fig, "03_brecha_clasificacion.png")


def fig_ir(meta):
    """04 — razón de desequilibrio (escala log) con las reglas de balanceo."""
    tareas = TAREAS + ["reg_mat"]
    irs = [float(meta[t].get("ir") or 0) for t in tareas]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y = np.arange(len(tareas))
    colores = [C["rojo"] if v > 5 else C["ambar"] if v > 1.2 else C["azul_claro"] for v in irs]
    ax.barh(y, [max(v, 1.0) for v in irs], 0.55, color=colores, zorder=3)
    for umbral, etiqueta in ((1.2, "sin balancear"), (5, "smote"), (9, "smote_rus")):
        ax.axvline(umbral, color=C["gris_claro"], ls=":", lw=1, zorder=2)
        ax.text(umbral, len(tareas) - 0.35, etiqueta, rotation=90, fontsize=7,
                color=C["gris"], va="top")
    for i, v in enumerate(irs):
        if v > 0:
            ax.text(max(v, 1.0) * 1.08, i, f"{v:.2f}", va="center", fontsize=9, color=C["marino"])
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQUETA[t].replace("\n", " ") for t in tareas], fontsize=9)
    ax.set_xlabel("Razón de desequilibrio (escala logarítmica)")
    ax.set_title("Desequilibrio de clases por tarea", loc="left")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    guardar(fig, "04_ir_por_tarea.png")


def copiar_figuras():
    print("Copiando figuras por tarea desde artefactos/reporte/:")
    for (tarea, tipo), destino in COPIAS.items():
        origen = os.path.join(ARTEF, tarea, f"{tipo}_{tarea}.png")
        if not os.path.exists(origen):
            sys.exit(f"No existe {origen}: corre primero `modelos.entrenar --tareas todas`.")
        shutil.copyfile(origen, os.path.join(SALIDA, destino))
        print(f"  {destino}  ({os.path.getsize(os.path.join(SALIDA, destino)) // 1024} KB)")


def main():
    os.makedirs(SALIDA, exist_ok=True)
    configurar_matplotlib()
    meta = {t: cargar(t) for t in TAREAS + ["reg_mat"]}
    print(f"Figuras resumen en {os.path.relpath(SALIDA)}:")
    fig_modelo_vs_base(meta)
    fig_auc(meta)
    fig_brecha(meta)
    fig_ir(meta)
    copiar_figuras()
    print("Listo: 4 figuras nuevas + 7 copiadas.")


if __name__ == "__main__":
    main()
