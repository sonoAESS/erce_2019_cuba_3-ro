# modelos/ — Bloque de construcción de modelos ERCE 2019

Produce los modelos pre-entrenados (proceso fiel a Weka) que consumirá la app
de escritorio de predicción/clasificación. Ver
[`docs/metodologia.md`](docs/metodologia.md) para el detalle metodológico.

## Módulos

| Módulo | Contenido |
|--------|-----------|
| `config.py` | rutas, semilla, umbrales de IR/balanceo/CV, tareas, features, paleta |
| `datos.py` | genera `datos/<tarea>.csv` (objetivo en última columna) |
| `preprocesamiento.py` | imputación/escalado/encoding (ColumnTransformer) |
| `seleccion_atributos.py` | 4 selectores estilo Weka (CFS, chi², InfoGain, wrapper) |
| `balanceo.py` | IR y samplers (SMOTE, SMOTE+RUS, SMOTEENN) |
| `clasificadores.py` / `regresores.py` | 7 y 6 algoritmos (equivalentes Weka) |
| `metricas.py` | reporte estilo Weka Explorer (Kappa, por clase, AUC, etc.) |
| `sobreajuste.py` | hold-out, brecha train-test, curvas de aprendizaje |
| `friedman.py` | Friedman + Nemenyi + diagrama CD |
| `plots.py` | gráficos publicables (PNG 300 dpi + SVG + PDF) |
| `experimento.py` | orquestador del proceso completo |
| `entrenar.py` | CLI (`python -m modelos.entrenar`) |
| `contrato.py` | validación de datos nuevos contra `metadata.json` |

## Uso rápido

```bash
python -m modelos.entrenar                    # todas las tareas
python -m modelos.entrenar --tareas reg_mat   # una tarea
python scripts/verificar_artefactos.py        # QA de artefactos
python -m pytest -q                          # tests
```

## Salida

- `artefactos/<tarea>/{pipeline.joblib, metadata.json, importancia.csv, ...}`
- `artefactos/reporte/<tarea>/{imágenes	y tablas}`
- `artefactos/manifiesto.json`

La app de escritorio (proyecto aparte) cargará `pipeline.joblib`, leerá
`metadata.json` como contrato y usará `modelos.pipeline`/`contrato` para
clasificar/predecir filas nuevas desde CSV/Excel.