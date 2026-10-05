# Reporte de resultados — Modelos de predicción ERCE 2019 (Cuba, 3er grado)

Fecha: 29 de septiembre de 2026
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

**Ninguno de los cinco modelos sirve para predecir el desempeño en Matemática
a partir de este módulo.** No es un resultado provisional ni un problema de
ajuste: es que la información del questionnaire no contiene la señal.

Para demostrarlo hay que comparar contra lo más simple posible: **adivinar
siempre lo más frecuente**. Si un modelo no supera esa respuesta trivial, no
está prediciendo nada.

| Modelo | Lo que consigue | Adivinar lo más frecuente | ¿Mejora? |
|--------|------------------|---------------------------|-----------|
| `clf_nivel` | 0.137 | 0.124 | apenas (+0.013) |
| `clf_riesgo` | 0.494 | 0.432 | apenas (+0.062) |
| `clf_superacion` | 0.342 | 0.342 | **no** (0.000) |
| `clf_repitencia` | 0.487 | 0.487 | **no** (0.000) |
| `reg_mat` | R² = −0.099 | R² = 0.000 | **peor** (−0.099) |

Las cifras son el F1 promedio de las 4 tareas de clasificación y el R² de la
regresión, ambas medidas sobre un 20 % de estudiantes reservado que el modelo
nunca vio en el entrenamiento.

Tres lecturas importantes de esa tabla:

1. **Dos modelos no hacen nada.** En la práctica, `clf_superacion` responde
   "no alcanza alto desempeño" a todos los estudiantes, y `clf_repitencia`
   responde "no repitió" a todos. Su puntaje no es desempeño: es el promedio de
   acertar siempre a la clase mayoritaria.

2. **Uno hace peor que no hacer nada.** En `reg_mat`, predecir siempre el
   puntaje promedio del grupo es mejor que usar el modelo entrenado (R²
   negativo). El modelo aprendió patrones que no se sostienen en estudiantes
   nuevos.

3. **El F1 de ~0.49 no es un 49 % de acierto.** Es la confusión habitual de
   promediar sobre clases desbalanceadas: en `clf_repitencia`, donde el 95 %
   no repitió, un modelo que siempre dice "no repitió" saca exactamente el
   mismo 0.487. Por eso este reporte siempre muestra la línea base al lado.

### 1.3 Qué significa

El módulo mide **percepciones**: cómo se siente el estudiante, cómo percibe al
docente y al directivo, qué tan bien cree que va la escuela. La prueba de
Matemática mide **rendimiento**. Resulta que las percepciones no anticipan el
rendimiento en este caso.

Esto no dice que las condiciones de vida o la calidad percibida sean irrelevantes
para el aprendizaje: dice que, **medidas con este cuestionario y para este
colectivo, no permiten distinguir anticipadamente a un estudiante de otro**. Es
una diferencia importante y esperable, porque se han documentado relaciones
reales entre contexto socioeconómico y aprendizaje con otro tipo de evidencia
—inspecciones en el aula y evaluaciones de proceso— que aquí no están.

Un detalle que sí llamó la atención: en las cuatro tareas de clasificación, el
método de selección de variables eligió **exactamente las mismas 8 variables, y
todas provenían del cuestionario de los directivos**. Es decir, lo único que
parecía relacionarse con el desempeño era la valoración que hace la dirección
de la escuela, y aun así esa relación es indistinguible del azar.

### 1.4 Qué se recomienda

1. **No desplegar estos modelos** para ninguna de las cinco tareas. Un modelo
   que devuelve la misma respuesta para todos no aporta en una alerta temprana
   ni en una distribuição de grupos: solo ocupa espacio y puede dar una falsa
   sensación de cobertura.

2. **Reorientar la pregunta.** Si el objetivo real es advertir sobre
   estudiantes en riesgo, la vía con respaldo es la que ya existe en la base:
   historial de asistencia, rezago escolar previo y el resultado de la propia
   evaluación del año anterior. La prueba de Matemática *sí* está disponible;
   lo que falta es poder usarla antes de que ocurra, no predecirla con
   percepciones.

3. **Reutilizar el módulo para lo que sí mide.** Las percepciones de clima
   escolar, la relación familia-escuela y la percepción de la gestión son
   insumos válidos para caracterizar centros, focalizar recursos y diagnosticar
   instituciones. El error sería usarlas como si midieran desempeño.

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
| `clf_nivel` | SMO (chi², none) | 2.21 | 0.137 | 0.124 | +0.013 | 0.510 | underfitting |
| `clf_riesgo` | IBk (chi², smote) | 3.16 | 0.494 | 0.432 | +0.062 | 0.513 | brecha controlada |
| `clf_superacion` | SMO (chi², none) | 1.08 | 0.342 | 0.342 | 0.000 | 0.504 | underfitting |
| `clf_repitencia` | IBk (chi², smoteenn) | 18.64 | 0.487 | 0.487 | 0.000 | 0.500 | brecha controlada |
| `reg_mat` | M5P (CFS, none) | — | R² −0.099 | R² 0.000 | −0.099 | r 0.314 | sobreajuste |

Ningún AUC supera 0.52. Una discrimination de 0.50 es indistinguible de lanzar
una moneda.

**Variables seleccionadas:** las cuatro clasificaciones eligen **las mismas 8**,
todas del cuestionario de directivos — `PPIT46_02_N`, `PPIT46_03_N`,
`PPIT46_04_N`, `PPIT46_05_N`, `DDIT41_01_N`, `DDIT41_02_N`, `DDIT41_03_N`,
`DDIT41_04_N`. La regresión elige 10 y sí mezcla bloques: `Indice_global`,
`E3IT21_04_N`, `E3IT21_05_N`, `PPIT46_01_N`, `ISECF`, `EDU`, `AUSE`, `ATRE`,
`LIBH`, `TSTU`.

**Friedman + Nemenyi:** en las cinco tareas la diferencia entre algoritmos es
estadísticamente significativa (p entre 0.0017 y 0.0426, CD ≈ 3.4–4.0), pero
eso compara algoritmos **con este conjunto de variables**, no calidad
predictiva: en `clf_nivel` el ganador por ranking es NaiveBayes y el
desplegado SMO, y ambos están en la línea base.

### 2.5 Diagnóstico de `reg_mat` (R² negativo)

No es un error de cálculo. Verificado:

- En entrenamiento: R² 0.495, r 0.704, RMSE 55.9, sesgo 0.
- En hold-out: r 0.314, RMSE 79.6, R² −0.099, sesgo **2.77 puntos (0.04 DE)**.
- El sesgo es despreciable; lo que hunde el R² es la **sobredispersión**: las
  predicciones se mueven más que el objetivo real (DE 84.0 frente a 75.9),
  patrón clásico de sobreajuste. Se confirma en la curva de aprendizaje
  (entrenamiento 56.2 contra validación 79.6) y en la CV 10-fold (R² −0.093).

### 2.6 Corrección aplicada durante este trabajo

`modelos/metricas.py` construía `clases` con `str()` mientras el objetivo
conservaba su tipo numérico en las tareas derivadas. `labels=clases` no casaba
con `y_true`, así que `confusion_matrix` devolvía una matriz de ceros y se
arrastraba a `TP_Rate`, `FP_Rate`, `MCC`, `ROC_Area`, `PRC_Area` y a las
gráficas de confusión, ROC y por clase de `clf_riesgo`, `clf_superacion` y
`clf_repitencia`. `exactitud`, `kappa`, `f1_macro` y `auc_macro` no se veían
afectadas porque no usan `labels`.

Se normalizan ambos lados a `str` y se añadió cobertura de prueba
(`tests/test_metricas.py`). Con las métricas ya correctas apareció lo que
expone el §1.2: los dos modelos que colapsan a la clase mayoritaria y el R²
negativo. **Los artefactos versionados en `artefactos/` provienen de una
corrida posterior a la corrección.**

También se corrigió la propagación de `NaN` en las derivadas de riesgo y
superación: `NL_VP1` ausente se convertía en clase 0, lo que inflaba
`clf_riesgo` y `clf_superacion` a 4 899 filas frente a 4 879 de `clf_nivel`.

### 2.7 Reproducir

```bash
entorno/bin/python -m modelos.datos       # regenera modelos/datos/*.csv
entorno/bin/python -m modelos.entrenar --tareas todas
entorno/bin/python scripts/verificar_artefactos.py
entorno/bin/python -m pytest -q            # 35 tests
```

Artefactos por modelo en `artefactos/<tarea>/`: `pipeline.joblib`,
`metadata.json`, tablas y gráficas; el resumen comparativo en
`artefactos/manifiesto.json`. El detalle del proceso está en
`docs/metodologia.md`.
