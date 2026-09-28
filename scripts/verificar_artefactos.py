# -*- coding: utf-8 -*-
"""Verificación de artefactos (QA post-entrenamiento).

Recarga cada `pipeline.joblib` y `metadata.json` de `artefactos/`, comprueba:
1. El contrato (features esperadas vs pipeline).
2. Que el pipeline transforma y predice sobre una muestra real.
3. Que el target predicho es consistente (clases conocidas en clasificación).
4. Que se generaron los reportes (imágenes + tablas).

Uso: python scripts/verificar_artefactos.py [--tarea clf_nivel]
"""
import argparse
import glob
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modelos import config as cfg  # noqa: E402
from modelos.contrato import validar_y_tipar  # noqa: E402


def cargar_tarea(tarea):
    carpeta = os.path.join(cfg.ARTEFACTOS_DIR, tarea)
    ruta_pipe = os.path.join(carpeta, "pipeline.joblib")
    if not os.path.exists(ruta_pipe):
        return None, None
    blob = joblib.load(ruta_pipe)
    with open(os.path.join(carpeta, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return blob, metadata


def verificar_tarea(tarea, df_ejemplo):
    errores = []
    blob, metadata = cargar_tarea(tarea)
    if blob is None:
        return {"tarea": tarea, "ok": False,
                "errores": ["sin artefactos (no entrenado)"],
                "n_predicciones": 0, "ejemplo_prediccion": [],
                "balanceo": None, "algoritmo": None}
    pipeline = blob["pipeline"]

    # 1) contrato: faltantes/sobrantes y tipado correcto
    ok = validar_y_tipar(df_ejemplo, metadata)
    if not ok["ok"]:
        errores.append(f"faltan columnas: {ok['faltantes']}")
    X = ok["df"]

    # 2) transformar + predecir sin excepción
    Xs = pipeline.named_steps["sel"].transform(
        pipeline.named_steps["pre"].transform(X))
    y_pred = pipeline.named_steps["mod"].predict(Xs)

    # 3) clases conocidas en clasificación
    if metadata.get("clases"):
        desconocidas = [v for v in pd.Series(y_pred).dropna().unique()
                        if str(v) not in metadata["clases"]]
        if desconocidas:
            errores.append(f"predice clases fuera del contrato: {desconocidas}")

    # 4) reportes
    rep = os.path.join(cfg.REPORTE_DIR, tarea)
    pngs = glob.glob(os.path.join(rep, "*.png"))
    if len(pngs) < 4:
        errores.append(f"solo {len(pngs)} PNG en reporte")
    for f in ("comparativa.csv", "fold_detalle.csv", "friedman_ranking.csv"):
        if not os.path.exists(os.path.join(rep, f)):
            errores.append(f"falta {f}")

    return {
        "tarea": tarea, "ok": not errores, "errores": errores,
        "n_predicciones": int(len(y_pred)),
        "ejemplo_prediccion": [str(v) for v in y_pred[:3]],
        "balanceo": metadata.get("balanceo"),
        "algoritmo": metadata.get("algoritmo"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tarea", default=None)
    args = p.parse_args()
    tareas = [args.tarea] if args.tarea else list(cfg.TAREAS)

    df = pd.read_csv(os.path.join(cfg.DATOS_DIR, "clf_nivel.csv")).iloc[:50]
    resumen = []
    fallos = 0
    for t in tareas:
        r = verificar_tarea(t, df.drop(columns=[df.columns[-1]]))
        estado = "OK" if r["ok"] else "FALLO"
        print(f"[{estado}] {t}: {r['errores'] or 'predicción correcta'}")
        fallos += 0 if r["ok"] else 1
        resumen.append(f"{t}->{r['ejemplo_prediccion']} "
                       f"({r['algoritmo']}, balanceo={r['balanceo']})")
    for linea in resumen:
        print(f"  {linea}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()