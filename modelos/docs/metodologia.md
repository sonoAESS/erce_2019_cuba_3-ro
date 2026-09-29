# Metodología del bloque de construcción de modelos

Este documento describe el proceso **fiel a Weka** implementado en `modelos/`
para construir los modelos de predicción/clasificación del ERCE 2019 (Cuba,
3er grado) y su uso desde la futura aplicación de escritorio.

## 1. Datos y tareas

- Fuente: paquete `erce` — Módulo Nacional de **Matemática** (CSV adicionado
  por el cliente, prueba M3; 4899 estudiantes de 3er grado tras deduplicar por
  IDSTUD) + contexto sociodemográfico de la base ERCE (FA_A3) y objetivos
  (PM3, NL_VP1 matemático).
- Cada dataset (`modelos/datos/<tarea>.csv`, NO versionado, regenerable con
  `python -m modelos.datos`) tiene **las features primero y la variable
  objetivo en la ÚLTIMA columna**, que es donde se mide el IR y se decide el
  balanceo (misma convención de "la última columna de la base de casos").
- 43 features: 40 numéricas + 3 categóricas (`provincia`, `género`,
  `area_esc`). Las numéricas incluyen los 21 ítems del Módulo Nacional
  (estudiante: `E3IT21_*`; docente: `PPIT44_*`, `PPIT45_N`, `PPIT46_*`;
  directivo: `DDIT41_*`), los 6 índices de calidad, el contexto escolar
  (`sec_3`, `can_prof_3`, `prof_hom_3`, `prof_muj_3`, `num_hab_esc`, `edad`)
  y el contexto sociodemográfico (`ISECF`, `EDU`, `PREE`, `AUSE`, `ATRE`,
  `LIBH`, `TSTU`). Se **excluyeron** `REPC` (objetivo de `clf_repitencia`),
  IDs/nombres/coordenadas y los ítems del cuestionario que no están en el
  módulo.

| Tarea | Tipo | Objetivo | Utilidad |
|------|------|----------|----------|
| `clf_nivel` | clasificación | Nivel Matemática I–IV | distribuir desempeño |
| `clf_riesgo` | clasificación | Nivel I vs resto | alerta temprana |
| `clf_superacion` | clasificación | Nivel III/IV vs resto | alto desempeño |
| `clf_repitencia` | clasificación | REPC 0/1 | caso desbalanceado (IR≈18) |
| `reg_mat` | regresión | puntaje Matemática | estimar nota según módulo y contexto |

## 2. Proceso (equivalente al Experimenter/Explorer de Weka)

```
datos -> hold-out reservado (20 %, estratificado si clasificación)
       -> CV de selección (5-fold estratificada):
            por fold y combinación (selector x algoritmo x balanceo):
               preprocesar -> seleccionar atributos -> balancear (si IR) -> entrenar
            se guardan exactitud / F1 / AUC / Kappa  (o RMSE / MAE / R²)
       -> Friedman + Nemenyi sobre los algoritmos (bloque = fold)
       -> ganador = mejor configuración (media CV)
       -> validación imparcial sobre HOLD-OUT (nunca usado para seleccionar)
       -> CV final 10-fold estilo Weka + 3x5 CV (media ± DE) de estabilidad
       -> detección de sobreajuste (brecha train vs test) + curva de aprendizaje
       -> persistencia: pipeline de despliegue + metadata de contrato + reportes
```

### 2.1 Preprocesamiento (`preprocesamiento.py`)
ColumnTransformer: numéricas → mediana + `StandardScaler`; categóricas →
moda + one-hot con `handle_unknown='ignore'` (datos nuevos con categorías
inéditas se ignoran sin romper la predicción). Es el equivalente de un
filtro dentro de un `FilteredClassifier` de Weka.

### 2.2 Selección de atributos (`seleccion_atributos.py`)
4 evaluadores estilo Weka, usables como paso de un pipeline:

| Weka | Implementación | Búsqueda |
|------|----------------|----------|
| `CfsSubsetEval` | mérito: `k·mean(r_cf)/√(k+k(k−1)mean(r_ff))` | GreedyStepwise hacia adelante con parada temprana |
| `ChiSquaredAttributeEval` | chi-cuadrado (discretizando) / F univariado en regresión | Ranker, top-K |
| `InfoGainAttributeEval` | ganancia de información (`mutual_info_*`) | Ranker, top-K |
| `WrapperSubsetEval` | rendimiento del propio algoritmo base (CV interna) | GreedyStepwise hacia adelante + tope |

Con 43 atributos, los evaluadores de **subconjunto** (CFS y Wrapper) acotan la
búsqueda mediante una **preselección univariada previa** (rankers chi-cuadrado /
F, equivalente a un `Ranker` de Weka antes de la sub-búsqueda): CFS explora
hasta `PRESELECCION_MAX=20` candidatos y Wrapper hasta `PRESELECCION_WRAPPER=12`
(además de `MAX_WRAPPER_FEATURES=8`, CV interna con 2 folds paralelos y parada
temprana cuando el mérito deja de mejorar). Los rankers puros (chi²/infogain)
no usan el estimador y se ajustan **una sola vez por fold**, reutilizándose
para todos los algoritmos.

### 2.3 Balanceo (`balanceo.py`)
Regla sobre el **IR = mayoría / minoría de la última columna**:

| IR | Acción |
|----|--------|
| ≤ 1.2 | sin balancear |
| ≤ 5 | SMOTE |
| ≤ 9 | SMOTE + RandomUnderSampler |
| > 9 | SMOTEENN |

Siempre se compara el modelo balanceado contra el **sin balancear** y se
conserva el mejor por la métrica principal. **El balanceador queda fuera del
pipeline persistido** (solo se aplica en el entrenamiento): la app de
escritorio solo necesita scikit-learn + joblib + pandas.

### 2.4 Algoritmos (`clasificadores.py`, `regresores.py`)
- Clasificación (7): `NaiveBayes`, `IBk`, `J48`, `SMO`, `Logistic`,
  `RandomForest`, `MultilayerPerceptron`.
- Regresión (6): `LinearRegression`, `SMOreg`, `IBk`, `M5P`, `RandomForest`,
  `MultilayerPerceptron`.

### 2.5 Métricas (`metricas.py`)
- Clasificación: reporte idéntico al Explorer de Weka (Correctly/Incorrectly,
  Kappa, y por clase: TP Rate, FP Rate, Precision, Recall, F-Measure, MCC,
  ROC Area, PRC Area) + matriz de confusión y AUC macro (OvR).
- Regresión: r, MAE, RMSE, RAE, RRSE y R².

### 2.6 Significancia (`friedman.py`)
Friedman (bloque = fold) + post-hoc de Nemenyi + diagrama de diferencia
crítica (Demsar 2006). Se responden: ¿difiere el mejor del resto? ¿qué
algoritmos forman un mismo grupo estadístico?

### 2.7 Sobreajuste (`sobreajuste.py`)
- Hold-out reservado: estimación imparcial.
- Brecha train-test por defecto > 0.08 → alerta de sobreajuste; validación
  < 0.45 → underfitting.
- Curvas de aprendizaje del modelo final.

## 3. Artefactos y contrato

Cada tarea deja `artefactos/<tarea>/`:

- `pipeline.joblib` — dict `{"pipeline", "metadata"}`: Pipeline
  pre→selector→modelo **entrenado con todos los datos**, listo para predecir
  filas nuevas.
- `metadata.json` — contrato completo (features esperadas, clases, algoritmo,
  selector, balanceo, IR, estimaciones, Friedman, librerías).
- `importancia.csv`, `confusion.csv` / `metricas_regresion.csv`,
  `por_clase.csv`.

Y en `artefactos/reporte/<tarea>/`: imágenes PNG (300 dpi) + SVG + PDF
(comparativa, efecto del balanceo, diagrama CD, matriz de confusión, ROC,
curvas de aprendizaje, importancia, correlación, distribución de la clase) y
tablas (`fold_detalle.csv`, `comparativa.csv`, `friedman_ranking.csv`,
`friedman_nemenyi.csv`).

`artefactos/manifiesto.json` resume todas las tareas.

`modelos/contrato.py::validar_y_tipar` es el contrato de entrada: detecta
faltantes/sobrantes y tipa numéricas/categóricas; el pipeline tolera
categorías inéditas. Es la única función que necesita el frontend de PySide6.

## 4. Uso

```bash
# Recrear datasets intermedios (opcional)
python -m modelos.datos

# Entrenar todas las tareas (proceso completo)
python -m modelos.entrenar

# Solo algunas, con prueba rápida de filas
python -m modelos.entrenar --tareas clf_riesgo reg_mat --muestras 1500

# QA de artefactos: recarga pipelines y comprueba contrato y predicción
python scripts/verificar_artefactos.py

# Tests unitarios
python -m pytest -q
```

Las semillas son fijas (`SEED=42`) para que el experimento sea reproducible.

## 5. Limitaciones y decisiones

- El wrapper de atributos es el paso más costoso; hay un tope de candidatos
  (preselección univariada previa) y de atributos seleccionados.
- `REPC` no entra como feature (ver §1).
- Los datos intermedios `modelos/datos/*.csv` son regenerables y no se
  versionan; `artefactos/` sí se versiona para que la app de escritorio tenga
  contratos reproducibles.
- Multiclase con IR > 5 se balancea con SMOTE (por binomio multiclase).