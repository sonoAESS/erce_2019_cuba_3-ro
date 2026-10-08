# Informe de ejecución — Módulo Nacional de Matemática, 3.er grado (ERCE 2019 · Cuba)

**Fecha de la ejecución:** 7 de octubre de 2026
**Alcance:** ejecución completa de los tres cuadernos de `cuadernos/`, de los
pruebas del repositorio y del QA de artefactos, con registro de los datos de
entrada, las decisiones tomadas y los resultados obtenidos.

---

## 1. Qué se ejecutó

| # | Qué | Cómo | Resultado |
|---|---|---|---|
| 1 | Preprocesamiento y exploración | `jupyter nbconvert --to notebook --execute --inplace cuadernos/exploracion.ipynb` | sin errores (≈45 s) · escribe los 3 datasets curados |
| 2 | Modelos y estimación | ídem sobre `cuadernos/estimacion.ipynb` | sin errores (≈30 min) · escribe `artefactos/indice_global/` |
| 3 | Análisis por preguntas de interés | ídem sobre `cuadernos/analisis_preguntas.ipynb` | sin errores (≈2 min) · figuras + tablas del estudio |
| 4 | Pruebas del proyecto | `entorno/bin/python -m pytest -q` | **119 passed, 4 skipped** |
| 5 | QA de artefactos | `entorno/bin/python scripts/verificar_artefactos.py` | contrato + predicción correcta en todos los pipelines |
| 6 | Re-entrenamiento de los 5 modelos | `entorno/bin/python -m modelos.entrenar --tareas todas` | ver §5.3 |

**Orden obligatorio:** `exploracion.ipynb` antes que `estimacion.ipynb` (este
lee los CSV que aquel escribe). `analisis_preguntas.ipynb` es independiente:
trabaja sobre las bases LLECE, no sobre el Módulo Nacional curado.

**Entorno** (`entorno/`, Python 3.12.13): pandas 3.0.5 · numpy 2.5.1 ·
scikit-learn 1.9.1 · imbalanced-learn · plotly · nbconvert.

Los cuadernos se ejecutan con `--ExecutePreprocessor.timeout=-1` porque la CV
de selección + afinado + estabilidad de `estimacion.ipynb` supera con creces
el timeout por defecto.

---

## 2. Datos

### 2.1 El archivo del cliente: `CUBA3_MAT_MODULO_NACIONAL.csv`

| Propiedad | Valor |
|---|---|
| Tamaño | 1.294.275 bytes · **5.595 filas × 49 columnas** |
| Codificación | **no es UTF-8**: falla con el primer acento (`Saíz`); se lee en `latin-1` vía `erce.config.read_csv` |
| Nulos | **0** en las 5.595 × 49 celdas |
| Grano real | **estudiante × docente**: 4.899 estudiantes únicos, 696 aparecen dos veces |
| Duplicados | 678 con dos docentes distintos (el módulo lo respondió cada docente de la sección) · 18 de la escuela 1096 con la misma ficha repetida |

**Contenido:** tres bloques de ítems (estudiante `E3IT21_*`, docente
`PPIT44/45/46_*`, directivo `DDIT41_*`), seis índices derivados, contexto de la
escuela (`num_hab_esc`, `can_prof_3`, coordenadas geográficas) y una capa de
identificadores (`IDSCHOOL`, `Id_prof`, …).

**Defectos de calidad detectados** (todos corregibles, ninguno bloqueante):

| Defecto | Detalle |
|---|---|
| `edad` no es la edad | valores 1–8 concentrados en 2 y 3: es un **código de banda**; `edad_anos = edad + 6` (7–14 años), verificado contra `FA_A3` en 4.857 estudiantes emparejados (22 sin `EDAD`) |
| Escalas mezcladas | `Indice_prep_didact_prof` y `Indice_calidad_clases_prof` vienen en **0–1**; los otros cuatro índices en 0–100 |
| Coordenadas como texto | `geo_esc_x` / `geo_esc_y` llegan como `str` (rango lat 20,16–23,87 / lon −84,34 a −75,54) |
| Localidad duplicada | `San Agustín` / `San Agustin` → 199 → 192 categorías tras normalizar |
| Columnas redundantes | `sec_3 ≡ can_prof_3 ≡ prof_hom_3 + prof_muj_3` (3 columnas, 2 grados de libertad) · `M3` constante |
| Baja varianza | `Indice_calidad_clases_dir` con 85–86 % de respuestas en 100; ítems del directivo con 94–98 % de un solo valor |

### 2.2 Los dos hallazgos que gobiernan todo el diseño

1. **Los seis índices son medias exactas de sus propios ítems** — se reproduce
   la fórmula en las 5.595 / 5.595 filas (correlación 1,000). Demostración
   práctica de la fuga: `Indice_global ~ sus 5 ítems E3IT21` da **R² = 1,000000**,
   mientras `Indice_global ~ contexto escolar` da R² = 0,0012. De ahí la regla
   de exclusión: los componentes del objetivo **nunca** entran como variables.
2. **Las tres fuentes del módulo son ortogonales.** Índices del estudiante
   correlacionan 0,87–0,88 entre sí (misma dimensión medida dos veces), pero
   entre bloques la correlación es ruido: máximo `|r|` con el objetivo por
   bloque no circular: directivo 0,036 · contexto 0,030 · PPIT45 0,035 ·
   PPIT46 0,056 · PPIT44 0,059 · **máximo global no circular 0,0592** (el
   bloque circular `E3IT21` llega a 0,7638 solo por definición).

### 2.3 Las bases del estudio (usadas por `analisis_preguntas.ipynb`)

`PL3.csv` y `PM3.csv` (5 valores plausibles VP1..VP5, escala regional 700 / DE
100), `FA_A3.csv` (cuestionario del estudiante, ISECF), `QP3.csv`
(cuestionario del profesor), los promedios por escuela con coordenadas (248
escuelas con puntaje) y el Módulo Nacional ICCP.

### 2.4 Datasets curados que consumen los modelos

Se escriben en `modelos/datos/` con la convención del proyecto (**variables
primero, objetivo en la última columna**), regenerables y fuera de git:

| Archivo | Forma | Objetivo | Distribución |
|---|---|---|---|
| `reg_indice_global.csv` | 4.899 × 29 | `Indice_global` (0–100) | 48,6 ± 36,0 |
| `clf_indice_bajo.csv` | 4.899 × 29 | `bajo` (0/1) | 1 : 2.513 · 0 : 2.386 (IR ≈ 1,05) |
| `clf_indice_nivel.csv` | 4.899 × 29 | `nivel` (bajo/medio/alto) | 2.513 / 1.658 / 728 (IR ≈ 3,45) |
| `seleccion_indice_global.csv` | 84 filas | — | frecuencia de selección de 28 variables × 3 tareas |

---

## 3. Decisiones tomadas

### 3.1 Preprocesamiento (`exploracion.ipynb`, función `curar()` con bitácora)

| # | Paso | Decisión | Efecto |
|---|---|---|---|
| 1 | Tipado | coordenadas y `edad` a numérico | — |
| 2 | Texto | normaliza acentos y espacios de la localidad | 199 → 192 categorías |
| 3 | Coordenadas | **una sola coordenada por escuela** (mediana) | 1 escuela con 2 valores resuelta |
| 4 | Duplicados | **estrategia B**: media de los dos docentes, primer valor del resto | 5.595 → 4.899 filas |
| 5 | `edad` | `edad_anos = edad + 6` | rango 7–14 años |
| 6 | Índices | los dos índices en 0–1 multiplicados por 100 | media 0,51 → 50,8 |
| 7 | Asimetría | `log_num_hab_esc = log1p(num_hab_esc)` | DE 8.417 → 0,71 |
| 8 | Exclusión circular | fuera `E3IT21_*`, `Indice_preg`, `Indice_refuerzo` | los componentes del objetivo no pueden ser variables |
| 9 | Categóricas | se conservan `provincia` (13), `género` (2), `area_esc` (2) | `municipio` (94) y `localidad` (199) descartados por cardinalidad |

**Sobre los duplicados se compararon tres estrategias y se eligió B:**

- **A** (la de `erce/data.py`: `sort_values("Id_prof")` + `drop_duplicates`) no
  es neutra: conserva siempre al docente 01, sesgando a los 256 estudiantes
  (37 % de los duplicados) donde el bloque docente difiere; desviación máxima
  0,5 en un ítem.
- **B** promedia los dos informes docentes: usa la información disponible, no
  pierde estudiantes y deja fracciones 0/1 que son coherentes con cómo el propio
  módulo construye sus índices.
- **C** (media de la escuela) da **exactamente lo mismo que B** en este archivo
  (diferencia 0,00), porque todos los estudiantes de una sección comparten el
  mismo par de docentes; colapsaría la variación entre estudiantes si el objeto
  de estudio fuera la escuela.

Las variables de baja varianza **no se borran**: un 94–98 % de respuestas
iguales es un dato del instrumento, no un error. Se marcan y se deja que los
selectores decidan.

### 3.2 Los tres objetivos y la regla de balanceo

Como el archivo no trae puntaje, el objetivo se toma del propio archivo: el
índice global de percepción del estudiante, en tres versiones que ejercitan los
tres casos de la regla por índice de desbalance (IR), calculado **por fold** y
nunca sobre el fold de validación:

| IR | Acción | Tarea que lo usa |
|---|---|---|
| ≤ 1,2 | sin balancear | `clf_indice_bajo` (IR ≈ 1,05) |
| 1,2 – 5 | SMOTE | `clf_indice_nivel` (IR ≈ 3,45) |
| 5 – 9 | SMOTE + submuestreo | — |
| > 9 | SMOTEENN | — |

El balanceo **siempre entra como opción `none`** en la rejilla: si balancear no
mejora, se conserva el modelo sin balancear (igual que en Weka).

### 3.3 Selección de características

Se usan los cuatro evaluadores de `modelos/seleccion_atributos.py` (CFS,
χ²/F, InfoGain, WrapperSubsetEval, con sus estrategias Weka), evaluados
**dentro de una CV de 5 folds**: ajustar el selector sobre todo el dataset y
puntuar sobre el mismo sería una selección optimista. Cada fold reajusta
preprocesador y selector, y se cuenta cuántas veces elige cada variable.

Lectura de resultado: frecuencia de selección (estable = se elige en muchos
folds) y consenso (elegida por varios evaluadores). **4 de 28 variables nunca
fueron seleccionadas** por ningún evaluador en ningún fold; las de mayor
consenso son `Indice_prep_didact_prof_100` (10 folds, 2,67 evaluadores),
`Indice_calidad_clases_prof_100` (14, 2,33) y `DDIT41_01_N` (12, 2,33).
La selección es **inestable por construcción** — el objetivo es casi ortogonal
a todo, así que los evaluadores se ven forzados a mirar variables sin señal; eso
es información sobre el instrumento, no un defecto del método.

### 3.4 Proceso de estimación (`estimacion.ipynb`, 12 pasos, fiel al `Experimenter` de Weka)

| # | Paso | Qué garantiza |
|---|---|---|
| 1 | Hold-out 20 % (estratificado en clasificación) | reserva que **nunca** participa en la selección |
| 2 | Rejilla: 4 selectores × familias baratas, y CFS × familias caras | barrido completo; incluye ampliación de 14–16 algoritmos |
| 3 | CV 5 folds | preprocesador **y** selector ajustados dentro del fold |
| 3b | Afinado del top-5 (`RandomizedSearchCV`, 20 iteraciones) | sampler dentro del pipeline; **nunca toca el hold-out** |
| 3c | Contraste contra la línea base | t pareado + Wilcoxon + **corrección de Holm** sobre los mismos folds |
| 4 | Comparativa por configuración | media ± DE, ordenada **mejor primero** |
| 5 | Friedman + Nemenyi | ¿las diferencias entre algoritmos son significativas? (rango 1 = mejor) |
| 6 | Cadena ganadora | `preprocesar → seleccionar → [balancear] → modelo` |
| 7 | Hold-out imparcial | el número que se reporta, **siempre junto a su línea base** |
| 8 | CV 10 folds (default de Weka) | cadena completa reentrenada por fold |
| 9 | Estabilidad 3×5 con semillas distintas | ¿aguanta el número del paso 7? |
| 10 | Sobreajuste | brecha train–test (primero comprueba si supera la base) + curva de aprendizaje |
| 11 | Importancia | `feature_importances_` → `|coef_|` → permutación sobre el hold-out |
| 12 | Persistencia | `pipeline.joblib` + `metadata.json` con contrato de entrada |

El registro ampliado (`modelos/algoritmos_extendidos.py`) **no toca** el registro
base de Weka que sustenta los cinco modelos de `artefactos/`: solo las tres
tareas de este cuaderno usan el ampliado.

### 3.5 Análisis por preguntas (`analisis_preguntas.ipynb`)

Once secciones, cada una con pregunta, proceder analítico y resultado: media
ponderada + EE BRR, ANOVA + η², t de Welch + d de Cohen, cuartiles del ISECF,
correlaciones de Pearson y mapas de promedios por escuela. Los valores
plausibles se promedian por fila (`VP1..VP5`) y las comparaciones se hacen con
los pesos muestrales (`WT`, `WSEN`).

---

## 4. Defectos detectados y corregidos durante esta ejecución

Los tres son del mismo tipo: **un orden invertido que hace que se elija lo
peor en lugar de lo mejor**. Los dos primeros afectaban a esta ejecución; el
tercero estaba en el bloque de los cinco modelos.

### 4.1 Signo de la diferencia contra la línea base (paso 3c) — `estimacion.ipynb`

`paso_3c_pareado` calculaba `m_alg = -g[principal]` en regresión mientras
`base` se quedaba en escala natural, con lo que `diferencia` salía como
`-2 × RMSE ≈ -72,87` en lugar del delta real `≈ -0,58`. Corregido a
`d = base - modelo` en regresión (mayor = mejor en ambos tipos) y el ordenado
pasa a `ascending=False` para ambos. **El hold-out no se veía afectado**: el
camino de la reserva era correcto, solo estaba mal la tabla del paso 3c.

### 4.2 Orden de la comparativa — `estimacion.ipynb`

`paso_4_comparativa` ordenaba con `ascending=not es_min(tipo)`, que deja
**los peores primero**, y `ejecutar_tarea` toma `comp.head(5)` como el top-5
que entra al afinado. Efecto: se afinaban las cinco **peores** configuraciones
de la rejilla y el ganador salía de ese cuadrante.

| Tarea | Top-5 que entraba (peores) | Top-5 correcto (mejores) |
|---|---|---|
| `reg_indice_global` | IBk ×3, LinearSVR, SMOreg (RMSE 36,6–39,5) | HistGB 34,15 · M5P 34,83 · RandomForest 35,20 |
| `clf_indice_bajo` | LDA/IBk/Logistic/J48/ExtraTrees con χ² (F1 0,39–0,43) | ExtraTrees wrapper 0,612 · J48 0,612 · HistGB 0,612 |
| `clf_indice_nivel` | NaiveBayes/BernoulliNB/LDA/Logistic (F1 0,18–0,23) | HistGB+smote 0,423 · ExtraTrees+smote 0,422 · J48 0,421 |

Corregido a `ascending=es_min(tipo)` (mejor primero: RMSE ascendente en
regresión, F1 descendente en clasificación). Además, el texto del veredicto
«el mejor resultado de la CV de selección fue X» tomaba `iloc[0]` de esa misma
tabla: con el orden corregido ahora sí muestra el mejor.

### 4.3 El mismo defecto en los cinco modelos — `modelos/experimento.py`

`mejor = agg.sort_values("principal_mean", ascending=not _mejor_es_min(tipo)).iloc[0]`
tenía el mismo `not`, introducido en el commit `8609b967` (previo a la
generación de los artefactos del 29-sep). La prueba está en las propias tablas
guardadas (`artefactos/reporte/<tarea>/comparativa.csv`): en **las cinco
tareas** la configuración desplegada era **la última de su rejilla**.

| Tarea | Desplegado | Rango (mejor → peor) | Mejor de la rejilla (CV 5) |
|---|---|---|---|
| `clf_nivel` | SMO/χ²/none · F1 0,130 | **56 / 56** | RandomForest/wrapper · 0,325 |
| `clf_riesgo` | IBk/χ²/smote · F1 0,429 | **56 / 56** | NaiveBayes/infogain · 0,602 |
| `clf_superacion` | SMO/χ²/none · F1 0,342 | **28 / 28** | SMO/wrapper · 0,633 |
| `clf_repitencia` | IBk/χ²/smoteenn · F1 0,428 | **56 / 56** | NaiveBayes/cfs · 0,604 |
| `reg_mat` | M5P/cfs · RMSE 81,9 | **24 / 24** | LinearRegression/wrapper · 71,1 |

Corregido a `ascending=_mejor_es_min(tipo)` (mejor primero en ambos tipos,
incluida la tabla de ranking del ramo `< 3 algoritmos`). **Implicación:** los
resultados de `docs/reporte_resultados.md` y la tabla de resultados del README
pertenecían a la corrida con el orden invertido; se regeneraron al re-entrenar
(ver §5.3).

### 4.4 Ranking de Friedman invertido en regresión — `modelos/experimento.py`

`friedman.rangos_medios` asigna **rango 1 al valor más grande**
(`argsort(argsort(-v))`, `modelos/friedman.py:32`). En clasificación eso es
correcto (F1 mayor = mejor), pero la llamada de `experimento.py` le pasaba el
**RMSE crudo** en regresión, con lo que el rango 1 iba al **peor** modelo. El
cuaderno sí lo hace bien: `paso_5_friedman` niega la métrica
(`metrica = tabla[principal] if tipo == "clasificacion" else -tabla[principal]`).

Evidencia en los artefactos de `reg_mat`: Friedman ponía a `IBk` como rango 1,0
con `LinearRegression` el último (5,0), cuando `IBk` es el de **mayor** RMSE
(75,4–78,8 según selector) frente a 71,09 de `LinearRegression`.

Corregido negando la métrica en regresión, igual que el cuaderno. Alcance real
del defecto: **solo el ranking y el diagrama CD**, no las conclusiones —
comprobado que los p-valores de Nemenyi no cambian (dependen de
`|Δrango|`, que se conserva al invertir todos los rangos) y que el ganador se
elige por `principal_mean`, no por el ranking. La tarea se relanzó para que
`friedman_ranking.csv`, `friedman_nemenyi.csv`, el diagrama CD y
`metadata.json` queden coherentes: el ranking de `reg_mat` pasó de
`IBk 1,0 · … · LinearRegression 5,0` (invertido) a
`SMOreg 1,8 · LinearRegression 2,0 · MultilayerPerceptron 2,6 · … · IBk 6,0`.

### 4.5 AUC congelado a 0,5 en `clf_indice_bajo` — `cuadernos/estimacion.ipynb`

`probas()` alineaba las columnas de probabilidad con `clases` comparando
`modelo.classes_` (numérico, `[0, 1]`) con `clases` (texto, `['1', '0']`, orden
de aparición). Como los tipos no casaban, **ninguna columna se copiaba**: la
matriz quedaba entera a 0,5, la renormalización la dejaba igual, y
`roc_auc_score` con puntuaciones constantes devuelve **exactamente 0,5**.

Efecto: todo el bloque de AUC de `clf_indice_bajo` era un número sin sentido —
hold-out `auc_macro = 0.5000`, CV 10 `0,5000 ± 0,0000` y la columna
`auc_macro` de **las 46 configuraciones** de `comparativa.csv` a 0,5 sin
variación. `exactitud`, `f1_macro`, `kappa`, el ganador y la línea base no se
veían afectados (no usan probabilidades).

Comprobado con el mismo modelo y el mismo split: el AUC real del hold-out es
**0,6567** (F1 0,6102 y exactitud 0,6184 reproducidos exactos). Corregido
normalizando ambos lados a `str` antes de alinear — la misma familia de bug que
la corrección previa de `modelos/metricas.py` documentada en
`docs/reporte_resultados.md` §2.6.

### 4.6 Probabilidades sin reordenar en el bloque de los 5 modelos — `modelos/experimento.py`

Relacionado con el anterior, por el lado contrario. `predict_proba` devuelve
las columnas en orden de `modelo.classes_` (orden del modelo), mientras que
`clases` es el orden de aparición en los datos, y `calcular_metricas_clasificacion`
y `plots.grafico_roc` leen la columna *i* como la clase *i* de `clases`.

Evidencia en `artefactos/<tarea>/por_clase.csv`:

| Tarea | `clases` (metadata) | `modelo.classes_` | Efecto |
|---|---|---|---|
| `clf_nivel` | `['III','I','II','IV']` | `['I','II','III','IV']` | filas de ROC/PRC mal etiquetadas (cada clase leía la columna del modelo de su posición) |
| `clf_superacion` | `['1.0','0.0']` | `['0.0','1.0']` | las dos filas salían a `0,3524`: la columna iba invertida respecto a la etiqueta, de modo que se reportaba `1 − AUC` (el correcto, `0,6476`, solo aparecía en la métrica global) |
| `clf_riesgo`, `clf_repitencia` | coinciden | coinciden | sin efecto |

Las métricas globales **sí** estaban bien: el `auc_macro` de hold-out usa la
columna de la clase positiva (la última ordenada), que coincide con la última
columna del modelo. Lo que quedaba mal era la tabla por clase y las curvas ROC
por clase. Corregido en `_probas()`, que ahora reordena las columnas al orden
de `clases` (y en `metricas`, que calcula la AUC binaria por identidad de clase
y reordena la multiclase al orden alfabético que exige scikit-learn).

Pruebas de regresión añadidas: `test_auc_binaria_con_clases_en_orden_de_aparicion`,
`test_auc_multiclase_reordena_a_orden_alfabetico_de_y_true` y
`test_probas_reordena_las_columnas_al_orden_de_clases` (suite: **122 passed,
4 skipped**). `clf_nivel` y `clf_superacion` se relanzaron para regenerar
`por_clase.csv` y las figuras por clase.



### 4.7 Matriz de correlación entre bloques entera a NaN — `cuadernos/exploracion.ipynb`

El primer bloque de código de la §7 (la matriz 6×6 de `|r|` medio entre pares de
columnas de dos bloques) salía **entera a NaN**, y el heatmap correspondiente se
dibujaba vacío. La causa: `raw[BLOQUES[a]].corrwith(raw[BLOQUES[b]])` —
`DataFrame.corrwith` entre dos frames **alinea las columnas por nombre**, y los
bloques no tienen nombres en común (`E3IT21_*` contra `PPIT44_*`, etc.), así que
cada par alineado era NaN y `np.nanmean` de un array todo-NaN devuelve NaN.

No se detectó antes porque la celda de configuración hace
`warnings.filterwarnings("ignore")` y el `RuntimeWarning: Mean of empty slice`
queda silenciado; además la conclusión de la §7 se apoyaba en la celda
siguiente, que correlaciona cada bloque contra la **Serie** `Indice_global`
(`corrwith` con una Serie sí funciona) y que nunca estuvo mal — de ahí que el
`|r| ≤ 0,06` del README y de este informe fuera correcto.

Corregido calculando una única matriz de correlación de todas las columnas
implicadas y rebanándola por bloques, con la diagonal ocupada por el `|r|` medio
**dentro** de cada bloque:

| | Estudiante | PPIT44 | PPIT45 | PPIT46 | DDIT41 | Contexto |
|---|---|---|---|---|---|---|
| Estudiante | **0,420** | 0,030 | 0,026 | 0,029 | 0,015 | 0,016 |
| PPIT44 | 0,030 | **0,778** | 0,033 | 0,160 | 0,046 | 0,056 |
| PPIT45 | 0,026 | 0,033 | — (1 columna) | 0,151 | 0,086 | 0,074 |
| PPIT46 | 0,029 | 0,160 | 0,151 | **0,533** | 0,107 | 0,038 |
| DDIT41 | 0,015 | 0,046 | 0,086 | 0,107 | **0,256** | 0,050 |
| Contexto | 0,016 | 0,056 | 0,074 | 0,038 | 0,050 | **0,169** |

Los valores confirman la narrativa de la §7: fuera de la diagonal todo queda
entre 0,015 y 0,160, y los dos máximos (PPIT44–PPIT46 y PPIT45–PPIT46) son
bloques del **mismo respondiente** (el docente). El estudiante correlaciona
≤ 0,030 con cualquier otro bloque. El cuaderno se re-ejecutó completo (44 celdas,
0 errores) y los nueve CSV de `modelos/datos/` quedaron **byte-idénticos** (mismo
hash), así que los artefactos de modelos siguen siendo coherentes.

---

## 5. Resultados

### 5.1 `exploracion.ipynb` — exploración, curación y selección

**Grano y limpieza resueltos:** 5.595 filas → **4.899 estudiantes** con la
estrategia B; localidad normalizada (199 → 192 categorías); una coordenada por
escuela; `edad_anos` en 7–14 años; dos índices llevados de 0–1 a 0–100;
`log1p` sobre habitantes (DE 8.417 → 0,71); excluidos los componentes del
objetivo. Resultado: **4.899 × 28 variables** (25 numéricas + 3 categóricas) más
el objetivo.

**Los tres objetivos quedaron balanceados como preveía la regla de IR:**

| Tarea | Distribución | IR | Acción |
|---|---|---|---|
| `reg_indice_global` | 48,6 ± 36,0 (0–100) | — | — |
| `clf_indice_bajo` | 1 : 2.513 / 0 : 2.386 | ≈1,05 | sin balancear |
| `clf_indice_nivel` | 2.513 / 1.658 / 728 | ≈3,45 | SMOTE |

**Selección de características (CV 5 folds, 4 evaluadores):** 4 de 28 variables
nunca se seleccionaron; las de mayor consenso (elegidas por más evaluadores en
más folds) fueron `Indice_prep_didact_prof_100`, `Indice_calidad_clases_prof_100`
y `DDIT41_01_N`. La selección es inestable por construcción: el objetivo es casi
ortogonal a todas las variables disponibles, así que los evaluadores se ven
forzados a elegir entre variables sin señal.

**Escritos:** los tres CSV de `modelos/datos/` (4.899 × 29) y
`seleccion_indice_global.csv` (84 filas de frecuencia).

**Coherencia interna y ortogonalidad entre bloques (§7):** la matriz de `|r|`
medio entre bloques (diagonal = dentro del bloque; tabla completa en §4.7)
confirma que las tres fuentes no se hablan: fuera de la diagonal el máximo es
0,160 (PPIT44–PPIT46, **mismo respondiente**) y el estudiante no pasa de 0,030
con ningún otro bloque. Por dentro, el bloque del estudiante es el más coherente
(0,420) y el del directivo el menos (0,256 — ítems con 94–98 % de un solo
valor, §2.1).

### 5.2 `estimacion.ipynb` — los tres modelos

Tres tareas × (barrido A: 4 selectores × familias baratas · barrido B: CFS ×
familias caras) + afinado del top-5 + estabilidad. **Las tres superan la línea
base en el hold-out**, cosa que la corrida con el orden invertido no lograba:

| Tarea | Ganador (default) | Afinado | Hold-out | Línea base | Δ | CV 10 | Estabilidad 3×5 | Brecha |
|---|---|---|---|---|---|---|---|---|
| `reg_indice_global` | HistGB + InfoGain | 34,01 (vs 34,15) | **r² 0,115** · RMSE 33,53 | r² 0,000 · RMSE 35,65 | **+0,115** | 34,08 ± 0,89 | −34,02 ± 0,61 | sobreajuste (+1,03) |
| `clf_indice_bajo` | HistGB + InfoGain | 0,6252 (vs 0,6121) | **F1 0,6102** | F1 0,3392 | **+0,2710** | 0,6115 ± 0,0442 | 0,6056 ± 0,0189 | ok (+0,051) |
| `clf_indice_nivel` | RandomForest + InfoGain + SMOTE | 0,4374 (vs 0,4205) | **F1 0,4410** | F1 0,2261 | **+0,2149** | 0,4362 ± 0,0165 | 0,4085 ± 0,0153 | underfitting (0,441 < 0,45) |

**AUC (columna corregida por los defectos §4.5 y §4.6):** el hold-out pasa de
ser un número ficticio a discriminar de verdad — `clf_indice_bajo` **0,6567**
(hold-out) y 0,6519 ± 0,0522 (CV 10), y `clf_indice_nivel` **0,6311** (hold-out)
y 0,6253 ± 0,0272 (CV 10). La columna `auc_macro` de las dos `comparativa.csv`
dejó de ser plana: ahora recorre `0,4849–0,6570` y `0,5093–0,6218`
respectivamente. F1, exactitud, kappa, ganador y línea base **no cambian** (no
usan probabilidades), y así lo confirma el re-run: los tres hold-out de la
tabla anterior son idénticos a los de la corrida previa a la corrección.

**Contraste contra la línea base (t pareado + Holm, sobre los mismos folds):**

| Tarea | Algoritmos superadores | Mejor diferencia |
|---|---|---|
| `reg_indice_global` | **2 de 14** | +2,0369 RMSE (HistGB) |
| `clf_indice_bajo` | **16 de 16** | +0,2764 F1 (ExtraTrees) |
| `clf_indice_nivel` | **16 de 16** | +0,2004 F1 (RandomForest) |

**Friedman** (diferencias entre algoritmos, 5 folds): las tres significativas —
reg `p = 1,65e-06`, CD 8,87, mejor rango HistGB (1,60) · `clf_indice_bajo`
`p = 4,91e-07`, CD 10,32, mejor rango ExtraTrees (2,40) · `clf_indice_nivel`
`p = 4,20e-07`, CD 10,32, mejor rango RandomForest (2,60). Que el p-valor sea
chico significa que **sí hay diferencias entre algoritmos**; la decisión final
se toma con el hold-out.

**Cadenas finales (8 variables cada una):**

- reg: `PPIT45_N`, `PPIT46_04_N`, `DDIT41_03_N`, `Indice_prep_didact_prof_100`,
  `log_num_hab_esc`, `num_hab_esc`, `area_esc_1`, `area_esc_2`
- `clf_indice_bajo`: `DDIT41_03_N`, `Indice_prep_didact_prof_100`,
  `Indice_calidad_clases_prof_100`, `log_num_hab_esc`, `num_hab_esc`,
  `can_prof_3`, `prof_muj_3`, `provincia_Ciego de Avila`
- `clf_indice_nivel`: `PPIT44_04_N`, `DDIT41_02_N`, `DDIT41_05_N`,
  `Indice_prep_didact_prof_100`, `log_num_hab_esc`, `num_hab_esc`,
  `provincia_Ciego de Avila`, `area_esc_2`

**Importancia por permutación (5 repeticiones):** en las tres tareas dominan
`log_num_hab_esc` y `Indice_prep_didact_prof_100` (la preparación didáctica del
docente), seguidos de `PPIT45_N` y `num_hab_esc`.

**Duraciones:** reg 304,9 s · `clf_indice_bajo` 732,0 s · `clf_indice_nivel`
2.217,4 s (≈37 min, la tarea con más rejilla).

**Ejercicio de verificación:** los tres `pipeline.joblib` se validan con
`modelos/contrato.validar_y_tipar` — 3/3 `ok=True`, 0 columnas faltantes ni
sobrantes, predicción correcta sobre 50 filas de prueba.

**Comparación con la corrida anterior (con el bug de orden):**

| Tarea | Corrida 2 (bug) | Corrida 3 (corregido) |
|---|---|---|
| reg | r² **−0,0066** (no superaba la base) | r² **+0,115** |
| `clf_indice_bajo` | F1 0,4739 | F1 **0,6102** |
| `clf_indice_nivel` | F1 0,3083 | F1 **0,4410** |

Es la diferencia que produce afinar las cinco **mejores** en lugar de las cinco
**peores** configuraciones de la rejilla.

### 5.3 Los cinco modelos del bloque de modelos (`modelos.entrenar`)

Corridos con el orden corregido del §4.3: en las cinco tareas la configuración
desplegada es la **mejor** de su rejilla (rank 1), no la última.

| Tarea | Config desplegada | Vars | Hold-out (principal) | Línea base | Δ | CV 10 | Estabilidad 3×5 | Brecha |
|---|---|---|---|---|---|---|---|---|
| `clf_nivel` | RandomForest · wrapper · SMOTE | 4 | **F1 0,3262** · AUC 0,581 | 0,1243 | **+0,2019** | 0,3136 ± 0,0234 | 0,3057 ± 0,0156 | sobreajuste |
| `clf_riesgo` | NaiveBayes · InfoGain · none | 8 | **F1 0,5608** · AUC 0,640 | 0,4319 | **+0,1289** | 0,5805 ± 0,0240 | 0,5737 ± 0,0343 | ok |
| `clf_superacion` | SMO · wrapper · none | 3 | **F1 0,6084** · AUC 0,648 | 0,3423 | **+0,2661** | 0,6331 ± 0,0374 | 0,6250 ± 0,0114 | ok |
| `clf_repitencia` | NaiveBayes · CFS · none | 4 | **F1 0,5241** · AUC 0,769 | 0,4868 | **+0,0373** | 0,5157 ± 0,0307 | 0,5807 ± 0,0359 | ok |
| `reg_mat` | LinearRegression · wrapper · none | 8 | **R² 0,1774** · RMSE 68,88 | R² 0,0000 · RMSE 75,94 | **+0,1774** | R² 0,184 ± 0,030 | −71,61 ± 1,55 | ok |

Ranking sobre la rejilla (`comparativa.csv`): **1/56 · 1/56 · 1/28 · 1/56 · 1/24**.
Friedman significativo en las cinco (p entre 0,0017 y 0,0426, CD ≈ 3,4–4,0).

**Antes / después de corregir el orden:**

| Tarea | Corrida anterior (bug) | Corrida corregida |
|---|---|---|
| `clf_nivel` | SMO/χ² · F1 0,137 (Δ +0,013) | RandomForest/wrapper/SMOTE · F1 **0,3262** (Δ +0,2019) |
| `clf_riesgo` | IBk/χ²/SMOTE · F1 0,494 (Δ +0,062) | NaiveBayes/InfoGain · F1 **0,5608** (Δ +0,1289) |
| `clf_superacion` | SMO/χ² · F1 0,342 (Δ **0,000**) | SMO/wrapper · F1 **0,6084** (Δ +0,2661) |
| `clf_repitencia` | IBk/χ²/SMOTEENN · F1 0,487 (Δ **0,000**) | NaiveBayes/CFS · F1 **0,5241** (Δ +0,0373) |
| `reg_mat` | M5P/CFS · **R² −0,099** | LinearRegression/wrapper · **R² +0,1774** |

**Lectura de los números nuevos:**

- Los cinco superan ahora la línea base —ninguno estaba por debajo antes—, pero
  la **capacidad predictiva sigue siendo limitada**: AUC de hold-out entre 0,58
  y 0,77, y R² 0,18 en la regresión. Es señal real, no azar, y a la vez techo de
  los datos.
- `clf_repitencia` es el más débil (Δ +0,037): en una tarea con IR 18,6 el
  ganador no balanceó, y la exactitud (0,739) es muy inferior a la clase
  mayoritaria (0,949); lo que sí aporta es un AUC de 0,769.
- `clf_nivel` mejora mucho el F1 (0,326 vs 0,124) pero con brecha de
  **sobreajuste**: la CV 10 (0,314) y la estabilidad (0,306) sostienen casi todo
  el número del hold-out.
- Las variables elegidas ya **no son las mismas en las cuatro
  clasificaciones** (4, 8, 3 y 4 variables) ni todas del directivo: entran
  `Indice_global`, `ISECF`, `EDU`, `LIBH`, `num_hab_esc`, `edad` y `género`.

*(la tabla de Friedman de `reg_mat` se regeneró tras corregir el defecto §4.4:
el re-entrenamiento de esa tarea se relanzó y sus artefactos se verifican abajo)*

**Friedman de `reg_mat` antes / después del defecto §4.4** (misma corrida, p
idéntico `1,70e-03` y CD 3,37; lo que se invierte es el ranking):

| # | Antes (invertido) | Después (corregido) |
|---|---|---|
| 1 | IBk · rango 1,0 | SMOreg · rango 1,8 |
| 2 | … | **LinearRegression · rango 2,0** (el desplegado) |
| 6 | LinearRegression · rango 5,0 | IBk · rango 6,0 (el peor en RMSE) |

**Re-ejecuciones por los defectos §4.5/§4.6:** `clf_nivel` y `clf_superacion`
se re-entrenaron para regenerar `por_clase.csv` y las figuras por clase con las
columnas de probabilidad alineadas; los F1/AUC globales son idénticos a los de
la tabla (0,3262/0,581 y 0,6084/0,648). `clf_riesgo` y `clf_repitencia` no
requieren re-corrida —su orden de clases coincide con el del modelo— ni
`reg_mat`, que no usa probabilidades.

### 5.4 `analisis_preguntas.ipynb` — el estudio

Once preguntas respondidas con tablas y figuras. Síntesis del cuaderno:

| # | Pregunta | Lectura | Matemática | Método |
|---|---|---|---|---|
| 1 | Rendimiento vs región | 730 (región 697) | 751 (región 698, **máx. regional**) | media ponderada + EE BRR |
| 2 | Provincias | brechas significativas, η² ≈ 0,01 | ídem | ANOVA + η² |
| 3 | Urbana vs rural | d ≈ 0,18 | d ≈ 0,26 | t de Welch + d de Cohen |
| 4 | Nivel socioeconómico | ≈60 pts (0,6 DE), r ≈ 0,26 | brecha y r similares | cuartiles ISECF + Pearson |
| 5 | Género | niñas +22 pts (d ≈ 0,24) | nula / no significativa | t de Welch + d de Cohen |
| 6 | Factores del estudiante | apoyo familiar +; violencia − | mismo patrón | correlaciones de Pearson |
| 7 | Calidad escolar (Módulo Nacional) | índice global **r ≈ 0,30** | similar (r ≈ 0,27 con PM3) | correlaciones de Pearson |
| 8 | Distribución geográfica | heterogeneidad intra-territorial | ídem | promedios por escuela + mapa |

**Conclusión del estudio (ambas asignaturas):** el promedio alto de Cuba
convive con una alta dispersión interna; las brechas relevantes son las
socioeconómicas (≈0,6 DE entre extremos del ISECF) y las que se observan
**dentro** de cada provincia, más que entre provincias. Los promedios de
índices del módulo son: `Indice_calidad_clases_dir` 95,5 ·
`Indice_refuerzo` 50,0 · `Indice_global` 48,6 · `Indice_preg` 47,3 ·
`Indice_calidad_clases_prof` 1,0 · `Indice_prep_didact_prof` 0,5 (estos dos
últimos todavía en escala 0–1 en la base sin curar).

> **Tensión que conviene dejar escrita:** el estudio (§9) correlaciona el
> módulo con el **puntaje real** (`PL3`/`PM3`) y obtiene `r ≈ 0,27–0,30`, mientras
> que `exploracion.ipynb` §7 afirma `|r| ≤ 0,06`. No hay contradicción: son
> objetivos distintos. El `|r| ≤ 0,06` es con `Indice_global` —la percepción que
> el propio estudiante declara sobre su clase— usando **solo variables no
> circulares**; el `r ≈ 0,30` es con el puntaje de la prueba, y ahí los ítems del
> estudiante (`E3IT21_*`) sí traen señal porque están en la misma escala de la
> experiencia escolar. Es exactamente el argumento del §5.2: para estimar el
> desempeño hay que unir `PM3.csv`, como ya hace el bloque de modelos.


---

## 6. Verificación

Verificación final del 8/oct/2026, **después** de las re-ejecuciones de los
defectos §4.4–§4.6:

- `entorno/bin/python -m pytest -q` → **122 passed, 4 skipped** (los tres tests
  nuevos son las pruebas de regresión de los defectos §4.5/§4.6:
  `test_auc_binaria_con_clases_en_orden_de_aparicion`,
  `test_auc_multiclase_reordena_a_orden_alfabetico_de_y_true` y
  `test_probas_reordena_las_columnas_al_orden_de_clases`).
- `scripts/verificar_artefactos.py` → contrato, predicción y reportes correctos
  en todos los pipelines persistidos.
- Los tres cuadernos terminan **sin celdas con error** (comprobado leyendo
  `outputs` de los `.ipynb`; la última corrida de `estimacion.ipynb`, 54 celdas,
  0 errores).
- Coherencia interna de los artefactos: `metadata.json`, `por_clase.csv` y
  `confusion.csv` de cada tarea comparten el mismo orden de `clases`
  (comprobado programáticamente para las cinco tareas del bloque de modelos y
  las tres del cuaderno).
- `cuadernos/estimacion.ipynb` valida además la entrada de los tres pipelines
  con `modelos/contrato.validar_y_tipar`, el mismo punto de entrada que usa el
  software de escritorio: 3/3 contratos `ok=True`, 0 columnas faltantes o
  sobrantes, predicción sobre las 50 filas de prueba.

---

## 7. Cómo se reproduce

```bash
# 1) datasets curados (obligatorio primero)
entorno/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 cuadernos/exploracion.ipynb

# 2) modelos y estimación sobre esos datasets
entorno/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 cuadernos/estimacion.ipynb

# 3) estudio sobre las bases LLECE (independiente)
entorno/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 cuadernos/analisis_preguntas.ipynb

# 4) cinco modelos del bloque de modelos + QA + pruebas
entorno/bin/python -m modelos.entrenar --tareas todas
entorno/bin/python scripts/verificar_artefactos.py
entorno/bin/python -m pytest -q

# 5) figuras del reporte de resultados (usa los artefactos del paso 4)
entorno/bin/python scripts/imagenes_reporte.py
```

### Dónde está cada cosa

| Ruta | Qué es | En git |
|---|---|---|
| `cuadernos/exploracion.ipynb` | EDA, curación con bitácora y selección | sí |
| `cuadernos/estimacion.ipynb` | modelos y estimación de las 3 tareas | sí |
| `cuadernos/analisis_preguntas.ipynb` | estudio: 11 preguntas de interés | sí |
| `modelos/datos/*.csv` | datasets curados, regenerables | no (gitignored) |
| `artefactos/` | pipelines, metadata, métricas, figuras | no (gitignored) |
| `scripts/imagenes_reporte.py` | genera las 4 figuras resumen del reporte y copia 7 figuras por tarea a `docs/imagenes/` | sí |
| `docs/imagenes/` | 11 PNG versionados (resumen + ROC/confusión/Friedman/importancia/scatter/aprendizaje) | sí |
| `docs/reporte_resultados.md` | informe de resultados para el cliente, con las figuras incrustadas | sí |
| `docs/informe_ejecucion.md` | este informe | sí |
