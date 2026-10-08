# -*- coding: utf-8 -*-
"""Orquestador del experimento Weka-fiel de clasificación/predicción.

Flujo por tarea (proceso idéntico en espíritu al Experimenter de Weka):

1. Cargar dataset (`features` + objetivo en última columna).
2. Reservar hold-out estratificado (20%) — nunca se usa para seleccionar.
3. CV interna (5-fold estratificada): por fold y combinación
   (selector × algoritmo × balanceo) se ajusta la cadena
   preprocesamiento → selección de atributos → balanceo (si el IR lo exige)
   → modelo, y se evalúa exactitud / F1 / AUC (o RMSE/MAE/R² en regresión).
4. Comparativa por (selector, algoritmo, balanceo) con media ± DE.
5. Friedman + Nemenyi sobre los algoritmos (bloque = fold) → ranking + CD.
6. Ganador = mejor config; evaluación imparcial sobre hold-out + CV 10-fold
   (default Weka) + estabilidad 3×5 (media ± DE).
7. Control de overfitting (brecha train-test) + curvas de aprendizaje.
8. Persistencia: pipeline de despliegue (sin balanceador), metadata de
   contrato, importancia, matrices/ROC y gráficos publicables.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score)

from . import balanceo, config as cfg, datos, friedman, metricas, plots, sobreajuste
from .clasificadores import obten_clasificadores
from .preprocesamiento import construir_preprocesador, nombres_transformados
from .regresores import obten_regresores
from .seleccion_atributos import crear_selector, ETIQUETAS_WEEKA


# ------------------------------------------------------------------ util
def _principal(tipo):
    return "f1_macro" if tipo == "clasificacion" else "rmse"


def _scoring_sklearn(tipo):
    return "f1_macro" if tipo == "clasificacion" else "neg_root_mean_squared_error"


def _nombre_original(nombre_pp, numericos):
    if nombre_pp in numericos:
        return nombre_pp
    for c in cfg.FEATURES_CATEGORICOS:
        if nombre_pp.startswith(c + "_"):
            return c
    return nombre_pp


def _probas(modelo, X, clases=None):
    """Probabilidades para clasificadores (predict_proba o decisión).

    Si se pasa `clases`, las columnas se reordenan para que queden en ese
    orden: `predict_proba` devuelve los nombres en `modelo.classes_` (orden
    del modelo), mientras `clases` es el orden de aparición en los datos y
    `calcular_metricas_clasificacion` / `grafico_roc` leen la columna i-ésima
    como la clase i-ésima de `clases`. Sin reordenar, las filas de
    `por_clase.csv` y las curvas ROC por clase quedan mal etiquetadas
    (p.ej. `clf_nivel`, cuyas clases llegan como ['III','I','II','IV']).
    """
    P = None
    if hasattr(modelo, "predict_proba"):
        try:
            P = np.asarray(modelo.predict_proba(X), dtype=float)
        except Exception:
            P = None
    if P is None and hasattr(modelo, "decision_function"):
        d = modelo.decision_function(X)
        if d.ndim == 1:
            P = np.c_[1 - d, d]
        else:
            P = np.asarray(d, dtype=float)
    if P is None or P.ndim != 2 or not clases:
        return P
    objetivo = [str(c) for c in clases]
    propias = [str(c) for c in getattr(modelo, "classes_", range(P.shape[1]))]
    if len(propias) != P.shape[1] or propias == objetivo:
        return P
    alineada = np.zeros((P.shape[0], len(objetivo)))
    for j, c in enumerate(propias):
        if c in objetivo:
            alineada[:, objetivo.index(c)] = P[:, j]
    return alineada


def _importancias(modelo, X, y, nombres_sel, tipo):
    n = min(cfg.MUESTRA_PERMUTACION, X.shape[0])
    if hasattr(modelo, "feature_importances_"):
        imp = np.abs(np.asarray(modelo.feature_importances_, dtype=float))
    elif hasattr(modelo, "coef_"):
        c = np.asarray(modelo.coef_, dtype=float)
        imp = np.abs(c.mean(axis=0) if c.ndim > 1 else c)
    else:
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.choice(X.shape[0], n, replace=False)
        try:
            pi = permutation_importance(modelo, X[idx], np.asarray(y)[idx],
                                        scoring=_scoring_sklearn(tipo),
                                        n_repeats=5, random_state=cfg.SEED,
                                        n_jobs=-1)
            imp = pi.importances_mean
        except Exception:
            imp = np.ones(X.shape[1])
    return pd.DataFrame({"variable": nombres_sel, "importancia": np.round(imp, 5)})


# ------------------------------------------------------------------- tarea
def ejecutar_tarea(tarea, muestras=None, folds_seleccion=cfg.CV_FOLDS_SELECCION,
                   folds_final=cfg.CV_FOLDS_FINAL, semilla=cfg.SEED,
                   una_config=None, verbose=True):
    spec = cfg.TAREAS[tarea]
    tipo = spec["tipo"]
    clasif = tipo == "clasificacion"
    principal = _principal(tipo)

    df = datos.cargar_dataset(tarea)
    if muestras:
        df = df.iloc[:muestras]
    objetivo = str(df.columns[-1])
    X, y = datos.separar_X_y(df, tarea)
    y_serie = y.copy()

    if clasif:
        clases = [str(c) for c in pd.Series(y).dropna().unique()]
        modelado_s = obten_clasificadores(semilla)
    else:
        clases = None
        modelado_s = obten_regresores(semilla)

    selectores_n = ([una_config["selector"]] if una_config
                    else list(cfg.SELECTORES_NOMBRES))
    modelos_n = ([una_config["algoritmo"]] if una_config else list(modelado_s))

    # 1) hold-out reservado
    Xtr_df, Xte_df, ytr_ser, yte_ser = sobreajuste.dividir_holdout(X, y_serie, tipo)
    pre_ref = construir_preprocesador()
    pre_ref.fit(Xtr_df)
    nombres_pp = nombres_transformados(pre_ref)
    ir_global = balanceo.calcular_ir(ytr_ser.values) if clasif else None

    _vprint = (lambda m: print(m)) if verbose else (lambda m: None)
    _vprint(f"\n=== {tarea} ({spec['descripcion']}) · n={len(df)} · "
            f"IR(entrenamiento)={ir_global}")

    # Warnings benignos del entrenamiento (no convergencia acotada por early
    # stopping, varianza nula en una clase, pivote del discretizador, etc.).
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 2) CV de selección
    folds = (StratifiedKFold(folds_seleccion, shuffle=True, random_state=semilla)
             if clasif else KFold(folds_seleccion, shuffle=True, random_state=semilla))
    filas = []
    for fold_n, (tr, va) in enumerate(folds.split(X, y_serie)):
        pre = construir_preprocesador()
        X_tr = pre.fit_transform(X.iloc[tr])
        X_va = pre.transform(X.iloc[va])
        y_tr = np.asarray(y_serie.iloc[tr])
        y_va = np.asarray(y_serie.iloc[va])
        ir_fold = balanceo.calcular_ir(y_tr) if clasif else None
        estrategia = (balanceo.elegir_balanceo(ir_fold,
                                               multiclase=len(clases) > 2)
                      if clasif else None)
        balanceos = ["none"] + ([estrategia] if estrategia else [])

        for sel_n in selectores_n:
            es_wrapper = sel_n == "wrapper"
            # Los rankers (CFS/chi2/infogain) no usan el estimador: se
            # ajustan una sola vez por fold y se reutilizan para todos los
            # algoritmos. El wrapper sí depende del modelo base.
            selectores_ajustados = {}
            if not es_wrapper:
                selector = crear_selector(sel_n, tipo, seed=semilla)
                selector.fit(X_tr, y_tr)
                selectores_ajustados[None] = (selector,
                                              selector.transform(X_tr),
                                              selector.transform(X_va))
            for mod_n in modelos_n:
                if es_wrapper:
                    modelo = modelado_s[mod_n]()
                    selector = crear_selector(sel_n, tipo, estimador=modelo,
                                              seed=semilla)
                    selector.fit(X_tr, y_tr)
                    X_sel_tr = selector.transform(X_tr)
                    X_sel_va = selector.transform(X_va)
                else:
                    modelo = modelado_s[mod_n]()
                    X_sel_tr, X_sel_va = selectores_ajustados[None][1:]
                for bal in balanceos:
                    Xb, yb = balanceo.aplicar_balanceo(X_sel_tr, y_tr, bal, seed=semilla)
                    modelo.fit(Xb, yb)
                    yp = modelo.predict(X_sel_va)
                    if clasif:
                        yprob = _probas(modelo, X_sel_va, clases)
                        mf = metricas.metricas_fold_clasif(y_va, yp, yprob, clases)
                    else:
                        mf = metricas.metricas_fold_regresion(y_va, yp)
                    filas.append({"fold": fold_n, "selector": sel_n,
                                  "selector_weka": ETIQUETAS_WEEKA[sel_n],
                                  "algoritmo": mod_n, "balanceo": bal,
                                  "ir_fold": ir_fold, **mf})
    _vprint(f"Configuraciones evaluadas: {len(filas)}")

    tabla = pd.DataFrame(filas)
    agg_cols = (["exactitud", "f1_macro", "auc_macro", "kappa"] if clasif
                else ["rmse", "mae", "r2"])
    agg = (tabla.groupby(["selector", "selector_weka", "algoritmo", "balanceo"])
           [agg_cols].agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["principal_mean"] = agg[f"{principal}_mean"].astype(float)

    # 4) Friedman por algoritmo (bloque = fold; valor del mejor config por fold)
    mejor_por_fold = (tabla.groupby(["fold", "algoritmo"])[principal]
                      .agg("max" if clasif else "min").reset_index())
    if mejor_por_fold["algoritmo"].nunique() >= 3:
        # `friedman.rangos_medios` asigna rango 1 al valor MÁS GRANDE
        # (argsort(argsort(-v))). En clasificación eso es el F1 mayor, pero en
        # regresión el RMSE más pequeño es el mejor, así que se niega igual
        # que hace `paso_5_friedman` del cuaderno. Sin este signo el ranking
        # y el diagrama CD salían invertidos: en reg_mat aparecía IBk como
        # rango 1.0 siendo el de MAYOR RMSE (75,4–78,8 frente a 71,1 de
        # LinearRegression), y LinearRegression quedaba el último.
        _metrica = (mejor_por_fold[principal] if clasif
                    else -mejor_por_fold[principal])
        fr = friedman.ranking_final(friedman.tabla_por_algoritmo(
            pd.DataFrame({"fold": mejor_por_fold["fold"],
                          "algoritmo": mejor_por_fold["algoritmo"],
                          "metrica": _metrica}),
            algoritmo_col="algoritmo"))
    else:
        fr = {"friedman_stat": float("nan"), "friedman_p": float("nan"),
              "significativo": False, "cd_nemenyi": float("nan"),
              "ranking": pd.DataFrame(
                  mejor_por_fold.groupby("algoritmo")[principal].mean()
                  .sort_values(ascending=_mejor_es_min(tipo))
                  .rename("rango_medio").reset_index()),
              "nemenyi": pd.DataFrame(), "n_folds": 0}

    # 5) ganador: siempre MEJOR primero (rango 1 arriba). En regresión el RMSE
    # menor va primero (ascending=True) y en clasificación el F1 mayor
    # (ascending=False), es decir `ascending=_mejor_es_min(tipo)`. El `not`
    # que había invertía el orden: `iloc[0]` devolvía la PEOR configuración
    # de la rejilla y los cinco modelos se entrenaron con ella (las tablas
    # de artefactos/reporte/<tarea>/comparativa.csv lo confirman, la
    # desplegada iba la última de todas).
    mejor = (agg.sort_values("principal_mean",
                             ascending=_mejor_es_min(tipo)).iloc[0])
    config = {"selector": mejor["selector"], "algoritmo": mejor["algoritmo"],
              "balanceo": mejor["balanceo"]}

    # 6) modelo final sobre el set de selección y evaluación imparcial
    pipe, nombres_sel = _entrenar_final(Xtr_df, ytr_ser, tipo, config, modelado_s, semilla)
    pre_f, sel_f, mod_f = pipe.named_steps["pre"], pipe.named_steps["sel"], pipe.named_steps["mod"]
    Xte_s = sel_f.transform(pre_f.transform(Xte_df))
    y_te_pred = mod_f.predict(Xte_s)
    y_te_prob = _probas(mod_f, Xte_s, clases) if clasif else None
    # Referencia del mismo holdout (clase mayoritaria / media del objetivo):
    # sin ella un f1_macro de 0.49 o un r2 negativo se leen como desempeño, y el
    # diagnóstico de brecha no distingue un modelo útil de uno que se aprovecha
    # del desbalance de clases.
    base = (metricas.linea_base_clasificacion(yte_ser) if clasif
            else metricas.linea_base_regresion(yte_ser))
    base_hold = ({k: base[k] for k in ("exactitud", "f1_macro", "auc_macro", "kappa")}
                 if clasif else {k: base[k] for k in ("r", "mae", "rmse", "r2")})
    if clasif:
        mm = metricas.calcular_metricas_clasificacion(yte_ser, y_te_pred, y_te_prob, clases)
        hold = {k: mm[k] for k in ("exactitud", "f1_macro", "auc_macro", "kappa")}
        y_tr_pred = mod_f.predict(sel_f.transform(pre_f.transform(Xtr_df)))
        _mt = metricas.metricas_fold_clasif(ytr_ser, y_tr_pred,
                                            _probas(mod_f, sel_f.transform(pre_f.transform(Xtr_df)), clases),
                                            clases)["f1_macro"]
        _supera = hold["f1_macro"] - base_hold["f1_macro"]
        dg = sobreajuste.detectar_sobreajuste(_mt, hold["f1_macro"],
                                              linea_base=base_hold["f1_macro"])
    else:
        mm = metricas.calcular_metricas_regresion(yte_ser, y_te_pred)
        hold = {k: mm[k] for k in ("r", "mae", "rmse", "rae", "rrse", "r2")}
        y_tr_pred = mod_f.predict(sel_f.transform(pre_f.transform(Xtr_df)))
        _mt = metricas.metricas_fold_regresion(ytr_ser, y_tr_pred)["rmse"]
        # La métrica se lleva negada (mayor = mejor) y el piso de 0.45 no aplica
        # a un RMSE: aquí el suelo es la media del objetivo.
        _supera = hold["r2"] - base_hold["r2"]
        dg = sobreajuste.detectar_sobreajuste(-_mt, -hold["rmse"], piso=None,
                                              linea_base=-base_hold["rmse"])
    base_hold["supera_a_linea_base"] = round(float(_supera), 4)

    cv_final = _cv_ganador(Xtr_df, ytr_ser, tipo, config, modelado_s, folds_final, semilla)
    estab = _cv_estabilidad(Xtr_df, ytr_ser, tipo, config, modelado_s, semilla)
    curva = sobreajuste.curva_aprendizaje(
        _pipeline_cv(config, tipo, modelado_s, semilla), Xtr_df, ytr_ser,
        tipo, _scoring_sklearn(tipo))
    imp = _importancias(mod_f, Xte_s, yte_ser.values, nombres_sel, tipo)
    imp["variable"] = imp["variable"].map(lambda n: _nombre_original(n, pre_f.numericos))

    # 7) persistencia
    carpeta = os.path.join(cfg.ARTEFACTOS_DIR, tarea)
    os.makedirs(carpeta, exist_ok=True)
    rep = os.path.join(cfg.REPORTE_DIR, tarea)
    os.makedirs(rep, exist_ok=True)

    _guardar_tablas(tarea, tabla, agg, fr, tipo)
    _guardar_graficos(tarea, tipo, tabla, agg, fr, pipe, nombres_pp, nombres_sel,
                      Xtr_df, ytr_ser, Xte_df, yte_ser, imp, curva,
                      clasif, y_te_pred, y_te_prob, classe_or_kwargs=clases)

    imp.to_csv(os.path.join(carpeta, "importancia.csv"), index=False)
    if clasif:
        mm["por_clase"].to_csv(os.path.join(carpeta, "por_clase.csv"), index=False)
        pd.DataFrame(mm["matriz"], columns=mm["clases"], index=mm["clases"]) \
            .to_csv(os.path.join(carpeta, "confusion.csv"))
    else:
        pd.Series(hold).to_frame("valor").to_csv(
            os.path.join(carpeta, "metricas_regresion.csv"))

    artef = {
        "modelo_id": tarea, "tarea": tarea, "tipo": tipo,
        "grupo": spec.get("grupo", ""), "descripcion": spec["descripcion"],
        "target": objetivo,
        "clases": [str(c) for c in clases] if clasif else None,
        "features_esperadas": [str(c) for c in X.columns],
        "features_por_preproceso": nombres_pp,
        "features_seleccionadas": nombres_sel,
        "algoritmo": config["algoritmo"], "selector": config["selector"],
        "selector_weka": ETIQUETAS_WEEKA[config["selector"]],
        "balanceo": config["balanceo"], "ir": round(float(ir_global or 0), 3),
        "estimacion": {
            "holdout": hold,
            "linea_base_holdout": base_hold,
            "cv_weka": cv_final,
            "estabilidad_3x5cv": estab,
            "brecha": {"estado": dg["estado"], "detalle": dg["detalle"],
                       "brecha": dg["brecha"]},
            "curva_aprendizaje": [{"n": c["n_entrenamiento"],
                                   "train": c["train_media"],
                                   "test": c["test_media"]} for c in curva],
        },
        "friedman": {"stat": fr["friedman_stat"], "p": fr["friedman_p"],
                     "significativo": fr["significativo"],
                     "cd": round(fr["cd_nemenyi"], 4),
                     "ranking": fr["ranking"].to_dict("records")},
        "fecha": pd.Timestamp.now().isoformat(),
        "librerias": {"sklearn": _v_lib("sklearn"), "numpy": np.__version__,
                      "pandas": pd.__version__, "imblearn": _v_lib("imblearn")},
    }
    _guardar_pipeline_despliegue(tarea, config, modelado_s, X, y_serie, tipo, semilla, artef)
    with open(os.path.join(carpeta, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(artef, f, ensure_ascii=False, indent=2, default=str)

    _vprint(f"   Ganador: {config['algoritmo']} ({config['selector']}, "
            f"balanceo={config['balanceo']}) → {principal} = {hold[principal]:.3f} · "
            f"brecha: {dg['estado']}")
    _lbl = "f1_macro" if clasif else "r2"
    _vprint(f"   Línea base ({'clase mayoritaria' if clasif else 'media'}): "
            f"{_lbl} = {base_hold[_lbl]:.3f} → el modelo "
            f"{'supera' if _supera > 0 else 'NO supera'} la base por "
            f"{abs(_supera):.3f}")
    return artef


def _mejor_es_min(tipo):
    return tipo == "regresion"


# ----------------------------------------------------------------- helpers
def _pipeline_cv(config, tipo, modelado_s, semilla):
    from sklearn.pipeline import Pipeline
    modelo = modelado_s[config["algoritmo"]]()
    selector = crear_selector(config["selector"], tipo, estimador=modelo, seed=semilla)
    return Pipeline([("pre", construir_preprocesador()), ("sel", selector),
                     ("mod", modelo)])


def _entrenar_final(Xtr_df, ytr, tipo, config, modelado_s, semilla):
    """Cadena (sin balanceador dentro) ajustada sobre el set de selección."""
    from sklearn.pipeline import Pipeline
    modelo = modelado_s[config["algoritmo"]]()
    selector = crear_selector(config["selector"], tipo, estimador=modelo, seed=semilla)
    pre = construir_preprocesador()
    X_pp = pre.fit_transform(Xtr_df)
    selector.fit(X_pp, np.asarray(ytr))
    X_sel = selector.transform(X_pp)
    est = (balanceo.elegir_balanceo(balanceo.calcular_ir(np.asarray(ytr)),
                                    multiclase=len(np.unique(ytr)) > 2)
           if tipo == "clasificacion" else None)
    if est:
        X_sel, y_aj = balanceo.aplicar_balanceo(X_sel, np.asarray(ytr), est, seed=semilla)
    else:
        X_sel, y_aj = X_sel, np.asarray(ytr)
    modelo.fit(X_sel, y_aj)
    pipe = Pipeline([("pre", pre), ("sel", selector), ("mod", modelo)])
    nombres = [n for i, n in enumerate(nombres_transformados(pre)) if selector.support_[i]]
    return pipe, nombres


def _cv_ganador(Xtr_df, ytr, tipo, config, modelado_s, k, semilla):
    """CV k-fold fiel a Weka de la configuración ganadora (dentro del set de selección)."""
    folds = (StratifiedKFold(k, shuffle=True, random_state=semilla)
             if tipo == "clasificacion" else KFold(k, shuffle=True, random_state=semilla))
    lista = []
    for tr, va in folds.split(Xtr_df, ytr):
        pipe_f, _ = _entrenar_final(Xtr_df.iloc[tr], ytr.iloc[tr],
                                    tipo, config, modelado_s, semilla)
        pre2, sel2, mod2 = (pipe_f.named_steps["pre"],
                            pipe_f.named_steps["sel"], pipe_f.named_steps["mod"])
        Xv = pre2.transform(Xtr_df.iloc[va])
        Xvs = sel2.transform(Xv)
        yp = mod2.predict(Xvs)
        if tipo == "clasificacion":
            _clases_cv = [str(c) for c in ytr.unique()]
            lista.append(metricas.metricas_fold_clasif(
                ytr.iloc[va], yp, _probas(mod2, Xvs, _clases_cv), _clases_cv))
        else:
            lista.append(metricas.metricas_fold_regresion(ytr.iloc[va], yp))
    dfr = pd.DataFrame(lista)
    out = {}
    for c in dfr.columns:
        out[f"{c}_media"] = round(float(dfr[c].mean()), 4)
        out[f"{c}_de"] = round(float(dfr[c].std()), 4)
    return out


def _cv_estabilidad(Xtr_df, ytr, tipo, config, modelado_s, semilla):
    rk = (RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg.REPETICIONES_VALIDACION,
                                  random_state=semilla)
          if tipo == "clasificacion"
          else RepeatedKFold(n_splits=5, n_repeats=cfg.REPETICIONES_VALIDACION,
                             random_state=semilla))
    scores = cross_val_score(_pipeline_cv(config, tipo, modelado_s, semilla),
                             Xtr_df, ytr, cv=rk, scoring=_scoring_sklearn(tipo), n_jobs=1)
    return {"media": round(float(scores.mean()), 4),
            "de": round(float(scores.std()), 4), "reps": int(len(scores))}


def _guardar_tablas(tarea, tabla, agg, fr, tipo):
    rep = os.path.join(cfg.REPORTE_DIR, tarea)
    os.makedirs(rep, exist_ok=True)
    tabla.to_csv(os.path.join(rep, "fold_detalle.csv"), index=False)
    cols = ["selector", "selector_weka", "algoritmo", "balanceo"] + [
        c for c in agg.columns if c.rsplit("_", 1)[0] in
        ("exactitud", "f1_macro", "auc_macro", "kappa", "rmse", "mae", "r2")]
    agg[cols].to_csv(os.path.join(rep, "comparativa.csv"), index=False)
    fr["ranking"].to_csv(os.path.join(rep, "friedman_ranking.csv"), index=False)
    fr["nemenyi"].to_csv(os.path.join(rep, "friedman_nemenyi.csv"))


def _guardar_graficos(tarea, tipo, tabla, agg, fr, pipe, nombres_pp, nombres_sel,
                      Xtr_df, ytr, Xte_df, yte, imp, curva, clasif,
                      y_te_pred, y_te_prob, classe_or_kwargs=None):
    rep = os.path.join(cfg.REPORTE_DIR, tarea)
    os.makedirs(rep, exist_ok=True)
    principal = _principal(tipo)

    mejor_por_algo = agg.loc[agg.groupby("algoritmo")["principal_mean"]
                             .apply(lambda s: s.idxmax() if clasif else s.idxmin(),
                                    include_groups=False)].copy()
    plots.grafico_comparativa(
        mejor_por_algo.rename(columns={"principal_mean": principal}),
        principal, rep, f"comparativa_{tarea}", f"Comparativa · {tarea} ({principal})")

    if tabla["balanceo"].nunique() > 1:
        plots.grafico_comparativa_balanceo(tabla, rep, f"balanceo_{tarea}")
    plots.guardar_fig(friedman.grafico_cd(fr["ranking"], fr["cd_nemenyi"]),
                      rep, f"friedman_cd_{tarea}")

    pre_f, sel_f, mod_f = pipe.named_steps["pre"], pipe.named_steps["sel"], pipe.named_steps["mod"]
    Xte_s = sel_f.transform(pre_f.transform(Xte_df))
    if clasif and classe_or_kwargs:
        plots.grafico_matriz_confusion(yte.values, y_te_pred, classe_or_kwargs, rep,
                                       f"matriz_confusion_{tarea}",
                                       f"Matriz de confusión · {tarea}")
        if y_te_prob is not None:
            plots.grafico_roc(yte.values, y_te_prob, classe_or_kwargs, rep,
                              f"roc_{tarea}", f"Curvas ROC · {tarea}")
    else:
        yp = np.asarray(y_te_pred)
        plots.grafico_scatter_regresion(yte.values, yp, rep, f"scatter_{tarea}",
                                        f"Observado vs predicho · {tarea}")
        plots.grafico_residuos(yte.values, yp, rep, f"residuos_{tarea}",
                               f"Residuos · {tarea}")
    if curva:
        plots.grafico_curva_aprendizaje(curva, rep, f"aprendizaje_{tarea}",
                                        f"Curva de aprendizaje · {tarea}")
    plots.grafico_importancia(imp, rep, f"importancia_{tarea}",
                              f"Importancia de atributos · {tarea}")

    sel = pipe.named_steps["sel"]
    sel_idx = [i for i in range(len(nombres_pp)) if sel.support_[i]]
    if len(sel_idx) > 2:
        sel_idx = sel_idx[::max(1, len(sel_idx) // 10)]
        sel_n = [nombres_pp[i] for i in sel_idx]
        Xtr_c = pre_f.transform(Xtr_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.corrcoef(Xtr_c[:, sel_idx].T)
        plots.grafico_correlaciones(sel_n, corr, rep, f"correlaciones_{tarea}",
                                    f"Correlación entre atributos seleccionados · {tarea}")
    plots.grafico_ir(ytr, rep, f"clases_{tarea}", f"Distribución de la clase · {tarea}")


def _guardar_pipeline_despliegue(tarea, config, modelado_s, X_all, y_all, tipo,
                                 semilla, artef):
    """Pipeline de despliegue entrenado con TODOS los datos disponibles.

    `metadata` queda incrustado para que el software de escritorio cargue
    contrato + pipeline con una sola lectura.
    """
    import joblib
    from sklearn.pipeline import Pipeline
    modelo = modelado_s[config["algoritmo"]]()
    selector = crear_selector(config["selector"], tipo, estimador=modelo, seed=semilla)
    pre = construir_preprocesador()
    X_pp = pre.fit_transform(X_all)
    selector.fit(X_pp, np.asarray(y_all))
    X_sel = selector.transform(X_pp)
    est = (balanceo.elegir_balanceo(balanceo.calcular_ir(np.asarray(y_all)),
                                    multiclase=len(np.unique(y_all)) > 2)
           if tipo == "clasificacion" else None)
    if est:
        X_sel, y_aj = balanceo.aplicar_balanceo(X_sel, np.asarray(y_all), est, seed=semilla)
    else:
        X_sel, y_aj = X_sel, np.asarray(y_all)
    modelo.fit(X_sel, y_aj)
    pipeline = Pipeline([("pre", pre), ("sel", selector), ("mod", modelo)])
    ruta = os.path.join(cfg.ARTEFACTOS_DIR, tarea, "pipeline.joblib")
    joblib.dump({"pipeline": pipeline, "metadata": artef}, ruta, compress=3)
    return ruta


def _v_lib(nombre):
    try:
        import importlib
        return getattr(importlib.import_module(nombre), "__version__", "?")
    except Exception:
        return "?"


def evaluar_entrada(pipeline, df):
    """Predice sobre un DataFrame de entrada con el pipeline de despliegue."""
    modelo = pipeline.named_steps["mod"]
    Xs = pipeline.named_steps["sel"].transform(pipeline.named_steps["pre"].transform(df))
    pred = modelo.predict(Xs)
    proba = _probas(modelo, Xs)
    return pred, proba


# ------------------------------------------------------------------- CLI
def ejecutar(tareas=None, muestras=None, verbose=True, **kwargs):
    """Ejecuta el experimento para una lista de tareas (default: todas)."""
    tareas = list(tareas or cfg.TAREAS)
    resultados = {}
    for t in tareas:
        resultados[t] = ejecutar_tarea(t, muestras=muestras, verbose=verbose, **kwargs)
    guardar_manifiesto(resultados)
    return resultados


def guardar_manifiesto(resultados):
    man = {
        "proyecto": "ERCE 2019 · Cuba – 3er grado – Modelos de predicción",
        "generado": pd.Timestamp.now().isoformat(),
        "proceso": "Weka-fiel: AttributeSelection + balanceo por IR + Friedman/Nemenyi",
        "tareas": {t: {k: r.get(k) for k in (
            "modelo_id", "tipo", "grupo", "descripcion", "clases",
            "features_esperadas", "algoritmo", "selector", "selector_weka",
            "balanceo", "ir", "estimacion", "friedman")} for t, r in resultados.items()},
    }
    with open(os.path.join(cfg.ARTEFACTOS_DIR, "manifiesto.json"), "w",
              encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2, default=str)
    return man