# -*- coding: utf-8 -*-
"""CLI de entrenamiento: `python -m modelos.entrenar`.

Ejecuta el experimento completo (proceso fiel a Weka) para una o varias
tareas y guarda artefactos + reportes en `artefactos/`.

Ejemplos:
    python -m modelos.entrenar --tareas clf_nivel clf_riesgo
    python -m modelos.entrenar --tareas todas --muestras 1500 --seed 42
    python -m modelos.entrenar --tareas reg_lect --folds-seleccion 5 --folds-final 10
"""
import argparse
import json
import sys

from . import config as cfg
from .experimento import ejecutar


def crear_argparse():
    p = argparse.ArgumentParser(
        description="Entrena los modelos ERCE 2019 (proceso Weka) y genera "
                    "artefactos/ reportes/ para la app de escritorio.")
    p.add_argument("--tareas", nargs="+", default=["todas"],
                   help="Tareas a entrenar: clf_nivel clf_riesgo clf_superacion "
                        "clf_repitencia reg_lect reg_mat, o 'todas'.")
    p.add_argument("--muestras", type=int, default=None,
                   help="Límite de filas por tarea (para pruebas rápidas).")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Semilla base del experimento.")
    p.add_argument("--folds-seleccion", type=int, default=cfg.CV_FOLDS_SELECCION,
                   help="CV interna de selección (default 5).")
    p.add_argument("--folds-final", type=int, default=cfg.CV_FOLDS_FINAL,
                   help="CV final estilo Weka (default 10).")
    p.add_argument("--quiet", action="store_true",
                   help="No imprimir progreso por tarea.")
    return p


def main(argv=None):
    args = crear_argparse().parse_args(argv)
    tareas = list(cfg.TAREAS) if "todas" in args.tareas else args.tareas
    desconocidas = [t for t in tareas if t not in cfg.TAREAS]
    if desconocidas:
        sys.exit(f"Tareas desconocidas: {desconocidas}. Disponibles: {list(cfg.TAREAS)}")
    print(f"Tareas: {tareas} · seed={args.seed} · "
          f"folds-selección={args.folds_seleccion} · folds-final={args.folds_final}")
    resultados = ejecutar(
        tareas=tareas, muestras=args.muestras, verbose=not args.quiet,
        folds_seleccion=args.folds_seleccion, folds_final=args.folds_final,
        semilla=args.seed)
    print(f"\nManifiesto actualizado: {cfg.ARTEFACTOS_DIR}/manifiesto.json")
    print(json.dumps({t: r["estimacion"]["holdout"] for t, r in resultados.items()},
                     indent=1, default=str))
    return resultados


if __name__ == "__main__":
    main()