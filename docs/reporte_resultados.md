# Reporte de resultados — Modelos de predicción ERCE 2019 (Cuba, 3er grado)

Fecha: 8 de octubre de 2026 (revisión de resultados; versión inicial del 29 de septiembre)
Alcance: clasificación y regresión sobre **Matemática**, con el Módulo Nacional
como conjunto de variables.

Este documento tiene dos partes. La **sección 1** es el resultado, escrita para
leerse sin conocimientos técnicos. La **sección 2** es el anexo técnico, con el
detalle metodológico y las tablas de resultados completas.

---

## 1. Resumen para el cliente

### 1.1 Qué se construyó

Cinco modelos, todos alimentados con el **Módulo Nacional de Matemática** que
entregó el cliente (43 variables: lo que responden estudiantes, docentes y
directivos, más el contexto sociodemográfico del estudiante y su escuela) y con
el resultado real de la prueba de Matemática de la base ERCE como variable a
predecir:

| Modelo | Pregunta que responde | Tipo |
|--------|-----------------------|------|
| `clf_nivel` | ¿En qué nivel de desempeño está el estudiante (I–IV)? | clasificación |
| `clf_riesgo` | ¿Está en riesgo (nivel I)? | clasificación |
| `clf_superacion` | ¿Alcanza alto desempeño (niveles III–IV)? | clasificación |
| `clf_repitencia` | ¿Repitió el año? | clasificación |
| `reg_mat` | ¿Qué puntaje de Matemática obtiene? | regresión |

El módulo se aplicó a **4 879 estudiantes** de 3er grado (de los 4 899 del CSV,
20 no tienen puntaje de Matemática registrado y quedaron fuera).

### 1.2 El resultado principal

**Los cinco modelos superan la línea base, pero la señal es débil.** La
comparación obligada es contra lo más simple posible —adivinar siempre lo más
frecuente— y contra un 20 % de estudiantes reservado que el modelo nunca vio:

| Modelo | F1 / R² del modelo | Adivinar lo más frecuente | Δ | AUC |
|--------|--------------------|---------------------------|---|-----|
| `clf_nivel` | 0.326 | 0.124 | +0.202 | 0.581 |
| `clf_riesgo` | 0.561 | 0.432 | +0.129 | 0.640 |
| `clf_superacion` | 0.608 | 0.342 | +0.266 | 0.648 |
| `clf_repitencia` | 0.524 | 0.487 | +0.037 | 0.769 |
| `reg_mat` | R² = 0.177 | R² = 0.000 | +0.177 | — |

Tres lecturas importantes de esa tabla:

1. **La mejora es real, pero el techo está puesto por los datos.** En la
   regresión, el módulo explica el **18 % de la varianza** del puntaje de
   Matemática; en las tres tareas de desempeño, un AUC de 0.58–0.65 es una
   discriminación pobre a moderada. Es suficiente para descartar el azar (la
   CV 10-fold y la estabilidad 3×5 sostienen el número) e insuficiente para
   identificar estudiantes individuales.

2. **El F1 no es un porcentaje de acierto.** En `clf_repitencia`, donde el 95 %
   no repitió, la clase mayoritaria sola saca 0.487; el modelo llega a 0.524
   (Δ +0.037), pero con un **AUC de 0.769**: sí ordena a los estudiantes por
   probabilidad de repetir, aunque apenas gane en el punto de corte. Es la
   única tarea con capacidad de discriminación aceptable, y no mide
   desempeño.

3. **Estos números no eran visibles hasta corregir la herramienta.** La
   versión anterior de este reporte concluía que *ninguna* tarea superaba la
   línea base. Ese veredicto era un **artefacto de dos bugs de la
   implementación** (el afinado seleccionaba las cinco *peores*
   configuraciones de la rejilla, y el ranking de Friedman estaba invertido en
   regresión), no un resultado de los datos. La corrección está documentada y
   con pruebas de regresión en `docs/informe_ejecucion.md` §4; todos los
   números de este informe provienen de corridas posteriores a la corrección.


### 1.3 Qué significa

El módulo mide **percepciones**: cómo se siente el estudiante, cómo percibe al
docente y al directivo, qué tan bien cree que va la escuela. La prueba de
Matemática mide **rendimiento**. Resulta que las percepciones no anticipan el
rendimiento en este caso.

Esto no dice que las condiciones de vida o la calidad percibida sean irrelevantes
para el aprendizaje: dice que, **medidas con este cuestionario y para este
colectivo, no permiten distinguir anticipadamente a un estudiante de otro** con
la precisión que exige una decisión individual. Es una diferencia importante y
esperable, porque se han documentado relaciones
reales entre contexto socioeconómico y aprendizaje con otro tipo de evidencia
—inspecciones en el aula y evaluaciones de proceso— que aquí no están.

El detalle que más llama la atención en las variables seleccionadas:
`Indice_global` —la valoración agregada de calidad— entra en **las cinco
tareas**, junto con variables de contexto socioeconómico (`ISECF`, `EDU`,
`AUSE`, `LIBH`, `TSTU`) y de tamaño de la escuela (`num_hab_esc`). El módulo no
aporta "un indicador mágico": aporta astringes de contexto que, por sí solos,
dejan el 82 % de la varianza sin explicar.


### 1.4 Qué se recomienda

1. **No desplegar estos modelos como sistema de alerta individual.** Con AUC
   0.58–0.65 en las tres tareas de desempeño, un modelo que "apunta" a un
   estudiante acierta apenas por encima del azar: la tasa de falsos positivos
   consume el supuesto beneficio de la alerta. Lo que sí son utilizables es la
   **probabilidad ordenada a nivel de grupo** (por escuela o por sección) para
   focalizar recursos, y eso con `clf_repitencia` (AUC 0.769).

2. **Reorientar la pregunta de la alerta temprana.** Si el objetivo real es
   advertir sobre estudiantes en riesgo de Matemática, la vía con respaldo es
   la que ya existe en la base: historial de asistencia, rezago escolar previo
   y el resultado de la propia evaluación del año anterior. La prueba de
   Matemática *sí* está disponible; lo que falta es poder usarla antes de que
   ocurra, no predecirla con percepciones.

3. **Reutilizar el módulo para lo que sí mide.** Las percepciones de clima
   escolar, la relación familia-escuela y la percepción de la gestión son
   insumos válidos para caracterizar centros, focalizar recursos y diagnosticar
   instituciones. El error sería usarlas como si midieran desempeño con una
   precisión que no tienen.

4. **Si se insiste en el modelo,** hace falta ampliar el conjunto de variables con
   información que describa al estudiante (no a la escuela): asistencia,
   resultados de años anteriores y nota de la prueba previa. Con
   las variables actuales el techo está puesto por los datos, no por el modelo.


---

## 2. Anexo técnico

### 2.1 Datos

- **Fuentes:** CSV del cliente (`CUBA3_MAT_MODULO_NACIONAL.csv`, byte-idéntico a
  `mod_mat` en `erce/config.py`) + `PM3.csv` (puntaje y nivel de Matemática,
  `GRUPO`, `WT`, `WSEN`) + `FA_A3.csv` (contexto sociodemográfico y `REPC`).
- **Deduplicación:** el CSV tiene 5 595 filas y 4 899 estudiantes únicos; los
  duplicados difieren solo en `Id_prof` y se conservan con
  `sort_values("Id_prof").drop_duplicates("IDSTUD", keep="first")`.
- **Cruce:** 4 879 de esos estudiantes tienen puntaje de Matemática; 20 quedan
  fuera por `PM3` nulo. Por eso `clf_nivel`, `clf_riesgo`, `clf_superacion` y
  `reg_mat` trabajan con 4 879 filas y `clf_repitencia` con 4 763 (faltan
  además los que no tienen `REPC` registrado).
- **Variable objetivo en última columna** (convención Weka: el IR y el balanceo
  se calculan sobre la última columna de la base de casos).

### 2.2 Variables (43 = 40 numéricas + 3 categóricas)

| Grupo | Variables |
|-------|-----------|
| Estudiante y contexto escolar | `edad`, `sec_3`, `can_prof_3`, `prof_hom_3`, `prof_muj_3`, `num_hab_esc` |
| Índices de calidad (6) | `Indice_global`, `Indice_preg`, `Indice_refuerzo`, `Indice_prep_didact_prof`, `Indice_calidad_clases_prof`, `Indice_calidad_clases_dir` |
| Ítems estudiante (5) | `E3IT21_01_N` … `E3IT21_05_N` |
| Ítems docente (11) | `PPIT44_01_N` … `PPIT44_05_N`, `PPIT45_N`, `PPIT46_01_N` … `PPIT46_05_N` |
| Ítems directivo (5) | `DDIT41_01_N` … `DDIT41_05_N` |
| Contexto sociodemográfico (7) | `ISECF`, `EDU`, `PREE`, `AUSE`, `ATRE`, `LIBH`, `TSTU` |
| Categóricas (3) | `provincia`, `género`, `area_esc` |

**Excluidas:** `REPC` (es el objetivo de `clf_repitencia`), identificadores,
nombres, coordenadas, y los ítems del cuestionario de estudiantes que no están
en el módulo.

### 2.3 Metodología (fiel a Weka)

- **Hold-out 20 %** reservado, estratificado en clasificación, que **nunca**
  participa en la selección de modelos → la estimación final es imparcial.
- **Selección:** CV 5-fold sobre el 80 % de selección, cruzando
  3 selectores de atributos (CFS, chi², wrapper) × 7 algoritmos × los
  balanceos aplicables por IR.
- **Balanceo por IR:** ≤1.2 `none`; ≤5 `smote`; ≤9 `smote_rus`; >9 `smoteenn`
  (multiclase: `smote`). Aplicado **dentro** de la CV, nunca antes.
- **Evaluación final:** CV 10-fold Weka de la configuración ganadora + estabilidad
  3 repeticiones × 5 folds + curva de aprendizaje + Friedman con Nemenyi
  (CD a 0.05).
- **Despliegue:** la cadena va **sin balanceador** (Weka evalúa sin balancear).
- `SEED = 42`; el proceso es reproducible.

### 2.4 Resultados por tarea

Hold-out (n = 976; 953 en `clf_repitencia`) frente a la línea base del mismo
holdout. `linea_base_holdout` queda en el `metadata.json` de cada modelo.

| Tarea | Ganador | IR | f1/r2 modelo | f1/r2 base | Δ | ROC | Diagnóstico |
|-------|---------|----|--------------|------------|---|-----|-------------|
| `clf_nivel` | RandomForest (wrapper, smote) | 2.21 | 0.326 | 0.124 | +0.202 | 0.581 | sobreajuste |
| `clf_riesgo` | NaiveBayes (InfoGain, none) | 3.16 | 0.561 | 0.432 | +0.129 | 0.640 | ok |
| `clf_superacion` | SMO (wrapper, none) | 1.08 | 0.608 | 0.342 | +0.266 | 0.648 | ok |
| `clf_repitencia` | NaiveBayes (CFS, none) | 18.64 | 0.524 | 0.487 | +0.037 | 0.769 | ok |
| `reg_mat` | LinearRegression (wrapper, none) | — | R² 0.177 | R² 0.000 | +0.177 | r 0.424 | ok |

La CV 10-fold de la configuración ganadora confirma el hold-out en las cinco
tareas (F1 0.314 ± 0.023 · 0.581 ± 0.024 · 0.633 ± 0.037 · 0.516 ± 0.031 ·
R² 0.184 ± 0.030), y la estabilidad 3×5 repite el mismo orden
(0.306 · 0.574 · 0.625 · 0.581 · −71.61 RMSE). En `clf_nivel` el hold-out
(0.326) está algo por encima de la CV (0.314): es sobreajuste leve, no un salto.

**Variables seleccionadas** (ya no coinciden entre tareas):

| Tarea | Variables |
|-------|-----------|
| `clf_nivel` | `edad`, `sec_3`, `num_hab_esc`, `Indice_global` |
| `clf_riesgo` | `num_hab_esc`, `Indice_global`, `Indice_refuerzo`, `Indice_prep_didact_prof`, `E3IT21_02_N`, `E3IT21_04_N`, `E3IT21_05_N`, `ISECF` |
| `clf_superacion` | `prof_muj_3`, `Indice_global`, `LIBH` |
| `clf_repitencia` | `edad`, `Indice_global`, `ISECF`, `género_Niño` |
| `reg_mat` | `Indice_global`, `Indice_preg`, `E3IT21_04_N`, `ISECF`, `EDU`, `AUSE`, `LIBH`, `TSTU` |

**Friedman + Nemenyi:** las cinco diferencias entre algoritmos son
estadísticamente significativas (p entre 0.0017 y 0.0426, CD ≈ 3.4–4.0), pero
eso compara algoritmos **con este conjunto de variables**, no calidad
predictiva. El ranking corrige el bug de orden documentado en
`docs/informe_ejecucion.md` §4.4: en `reg_mat`, IBk (rank 1,0 = el peor RMSE)
ha sido reemplazado por SMOreg (1,8) y LinearRegression (2,0).

### 2.5 Diagnóstico de `reg_mat` — de R² negativo a R² 0.177

El R² negativo de la versión anterior **no era un hallazgo del modelo sino un
artefacto del código de selección**: el afinado elegía sistemáticamente la
*peor* configuración de la rejilla (M5P/CFS, R² −0.099). Corregido el orden, la
configuración desplegada es la mejor de su rejilla y el resultado es:

- Hold-out: r 0.4245, RMSE 68.88 (frente a 75.94 de la línea base), MAE 54.75,
  RAE 0.907, **R² 0.177**.
- CV 10-fold: R² 0.184 ± 0.030, RMSE 71.55 ± 2.99.
- Diagnóstico de brecha: `ok` (train y test consistentes), frente al
  `sobreajuste` del modelo anterior.

Sigue siendo una explicación parcial: el 82 % de la varianza del puntaje queda
sin explicar, y la brecha frente a la predicción por la media (ΔR² +0.177) es
real pero modesta.

### 2.6 Correcciones aplicadas durante este trabajo

Tres familias de bugs afectaron los resultados, todas corregidas y con pruebas
de regresión:

1. **`modelos/metricas.py` (septiembre).** `clases` se construía con `str()`
   mientras el objetivo conservaba su tipo numérico; `labels=clases` no casaba
   con `y_true`, `confusion_matrix` devolvía una matriz de ceros y se arrastraba
   a `TP_Rate`, `FP_Rate`, `MCC`, `ROC_Area`, `PRC_Area` y a las gráficas. Se
   normalizan ambos lados a `str`.

2. **Selección y orden (8 de octubre).** El afinado de `cuadernos/estimacion.ipynb`
   y de `modelos/experimento.py` tomaba las cinco **peores** configuraciones de
   la rejilla, y el ranking de Friedman se invertía en regresión
   (`modelos/friedman.py` da rango 1 al valor más grande y la regresión pasaba
   el RMSE sin negar). Eso producía los veredictos "nulo" y "peor que la base"
   de la versión anterior de este reporte.

3. **Alineación de probabilidades (8 de octubre).** En `estimacion.ipynb`,
   `probas()` no alineaba las columnas de probabilidad (`modelo.classes_` es
   numérico, `clases` es texto) y **todos los AUC de `clf_indice_bajo` salían
   congelados a 0.5000**. En `modelos/experimento.py`, `_probas()` no reordenaba
   las columnas al orden de `clases`, de modo que `por_clase.csv` y las curvas
   ROC por clase de `clf_nivel` y `clf_superacion` estaban mal etiquetadas
   (`clf_superacion` reportaba `1 − AUC`, es decir 0.352 en vez de 0.648).

Los artefactos versionados en `artefactos/` provienen de corridas posteriores a
las tres correcciones; el detalle, la verificación y las pruebas de regresión
están en `docs/informe_ejecucion.md` §4 y §6.

Además se corrigió la propagación de `NaN` en las derivadas de riesgo y
superación: `NL_VP1` ausente se convertía en clase 0, lo que inflaba
`clf_riesgo` y `clf_superacion` a 4 899 filas frente a 4 879 de `clf_nivel`.

### 2.7 Reproducir

```bash
entorno/bin/python -m modelos.datos       # regenera modelos/datos/*.csv
entorno/bin/python -m modelos.entrenar --tareas todas
entorno/bin/python scripts/verificar_artefactos.py
entorno/bin/python -m pytest -q            # 122 tests
```

Los tres cuadernos (`exploracion`, `estimacion`, `analisis_preguntas`) se
reproducen con `jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.timeout=-1`. Artefactos por modelo en
`artefactos/<tarea>/`: `pipeline.joblib`, `metadata.json`, tablas y gráficas; el
resumen comparativo en `artefactos/manifiesto.json`. El detalle del proceso está
en `docs/metodologia.md` y el relato completo de esta ejecución en
`docs/informe_ejecucion.md`.
