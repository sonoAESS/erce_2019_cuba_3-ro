# Análisis estadístico ERCE 2019 – Cuba, 3er grado

Estudio estadístico interactivo sobre el rendimiento en **Lectura** y **Matemática** de los
estudiantes cubanos de **3er grado** en el **ERCE 2019** (UNESCO-LLECE), con gráficos
profesionales (Plotly), análisis inferencial e interpretaciones automáticas, presentado en
una app **Streamlit** y en un cuaderno **Jupyter**.

## Preguntas que responde el estudio

1. **Rendimiento frente a la región**: posición de Cuba en el ERCE 2019.
2. **Territorio**: diferencias entre provincias (ANOVA) y entre zona urbana/rural (t de Welch).
3. **Socioeconomía**: gradiente del puntaje según el índice socioeconómico de la familia (ISECF) y la educación de los padres.
4. **Género**: brechas niño/niña en cada asignatura.
5. **Factores del estudiante**: correlaciones con edad, apoyo al aprendizaje, violencia escolar, involucramiento parental, entre otros.
6. **Calidad escolar / docente**: índices del Módulo Nacional (global, preguntas, refuerzo, calidad de clases) y su vínculo con los puntajes.
7. **Distribución geográfica**: puntaje medio por escuela sobre el mapa de Cuba.
8. **Contexto regional**: Cuba frente al promedio de la región.

## Conclusiones principales

Basadas en el análisis de las bases oficiales de Cuba, 3er grado, con medias ponderadas
(`WSEN`), errores estándar por réplicas BRR y pruebas inferenciales (t, ANOVA, χ², Pearson).

### Nacional y regional
- Cuba promedia **730 puntos en Lectura** y **751 en Matemática**, por encima de la media
  regional de referencia (700) y del promedio regional del ERCE 2019 (697 y 698). En
  **Matemática de 3.er grado Cuba registró el puntaje más alto de la región**.
- La dispersión interna es alta: las diferencias **entre estudiantes dentro de Cuba**
  superan a las diferencias entre países de la región, por lo que el desafío principal es la
  **equidad interna** más que el nivel medio.

### Socioeconomía (el hallazgo más consistente)
- El ISECF es el factor individual **más asociado** al rendimiento en ambas asignaturas:
  el cuartil más alto supera al más bajo por ≈ **60 puntos** (≈0.6 DE).
- Correlación **r ≈ 0.26** (≈7% de la varianza explicada), monótona y sistemática.

### Territorio
- Las diferencias entre **provincias son significativas** pero de efecto pequeño
  (η² ≈ 0.01): la provincia explica apenas ~1% de la varianza; la mayor parte de la
  desigualdad ocurre **dentro de cada provincia**.
- La brecha **urbano-rural** es significativa pero pequeña (d < 0.3) en ambas asignaturas;
  es menor que la brecha socioeconómica, lo que sugiere que la desventaja rural se explica
  más por condiciones socioeconómicas que por la ruralidad en sí.

### Género
- En **Lectura**, las niñas superan a los niños por ≈ **22 puntos** (d ≈ 0.24, significativo).
- En **Matemática**, la brecha es mínima y **no significativa**.

### Factores del estudiante
- Se asocian positivamente con el rendimiento: apoyo al aprendizaje (r ≈ 0.24),
  involucramiento parental (r ≈ 0.24) y organización de la enseñanza (r ≈ 0.23).
- La **violencia escolar** se asocia negativamente (r ≈ −0.24).
- La edad y la autoeficacia en matemática prácticamente no se asocian.

### Calidad escolar y docente
- El **índice global de calidad** del Módulo Nacional es el más asociado (r ≈ 0.30),
  seguido del refuerzo pedagógico (r ≈ 0.27) y del diálogo pedagógico (r ≈ 0.23).
- Asociaciones de magnitud **pequeña a moderada**: la calidad pedagógica percibida
  acompaña a mejores resultados, pero no es el único determinante.

## Estructura

```
├── app.py                 # App Streamlit (secciones + conclusiones por pregunta)
├── cuadernos/             # Cuadernos Jupyter (ver sección siguiente)
├── docs/                  # Metodología y reportes en Markdown
├── modelos/               # Biblioteca de modelado: config, preprocesamiento,
│                          # selectores de atributos, métricas, plots, experimento
├── artefactos/            # Por modelo: pipeline.joblib, metadata.json, metricas,
│                          # importancias, figuras de reporte y manifiesto.json
├── erce/
│   ├── config.py          # Rutas, constantes, etiquetas
│   ├── data.py            # Carga y limpieza de datos (LLECE + Módulo Nacional)
│   ├── analysis.py        # Estadística descriptiva e inferencial (medias ponderadas, BRR, t, ANOVA, chi2, Pearson)
│   ├── plots.py           # Figuras Plotly profesionales
│   └── interpret.py       # Interpretaciones automáticas y conclusiones
├── pyproject.toml         # Dependencias declaradas (uv)
├── requirements.txt       # Dependencias pinneadas (pip)
├── uv.lock                # Lockfile de uv
└── README.md
```

> Los directorios `07.ERCE-2019-FINAL/` (bases de datos) y `2. Libros de código/`
> (codebooks) **no se versionan** en el repositorio; deben copiarse aparte en la raíz
> del proyecto para que la app y el cuaderno funcionen.

## Módulo Nacional: qué se puede predecir con estos datos

Dos cuadernos, que deben ejecutarse **en este orden**. El primero produce los datos que
el segundo consume, de modo que el flujo completo es reproducible con dos comandos.

| Cuaderno | Qué hace | Qué escribe |
|---|---|---|
| `cuadernos/exploracion.ipynb` | Carga y cura `CUBA3_MAT_MODULO_NACIONAL.csv`: 5 595 filas → 4 899 estudiantes únicos, resolución de duplicados por docente, tipado de 28 variables (25 numéricas + `provincia`, `género`, `area_esc`), índices derivados y selección Weka fold a fold | `modelos/datos/reg_indice_global.csv`, `clf_indice_bajo.csv`, `clf_indice_nivel.csv` y `seleccion_indice_global.csv` |
| `cuadernos/estimacion.ipynb` | Reproduce celda por celda el proceso de `modelos/experimento.py`: hold-out 20 %, CV 5 con los cuatro `AttributeSelection` de Weka (`CfsSubsetEval`, `ChiSquared`, `InfoGain`, `WrapperSubsetEval`), balanceo por índice de desbalance, Friedman/Nemenyi, CV 10, estabilidad 3×5, brecha train–test, curva de aprendizaje e importancia de variables | `artefactos/indice_global/` con `pipeline.joblib`, `metadata.json`, CSV de métricas y 7–9 figuras por tarea (PNG/SVG/PDF) |

```bash
entorno/bin/python -m jupyter nbconvert --to notebook --execute --inplace cuadernos/exploracion.ipynb
entorno/bin/python -m jupyter nbconvert --to notebook --execute --inplace cuadernos/estimacion.ipynb
```

> `cuadernos/estimacion.ipynb` recorre la rejilla completa (≈ 800 evaluaciones de modelo) y tarda
> del orden de 40 minutos. `modelos/experimento.py` ejecuta exactamente el mismo
> proceso; el cuaderno existe para que cada paso sea auditable y modificable.

### Resultado

Las variables del Módulo Nacional **no explican** el índice global de percepción del
estudiante: la correlación máxima con cualquier variable disponible es `|r| ≤ 0,06`
(`cuadernos/exploracion.ipynb`, §7). Aplicando el proceso completo se obtiene:

| Tarea | Ganador (CV 5) | Hold-out | Línea base | Veredicto |
|---|---|---|---|---|
| `reg_indice_global` (RMSE) | IBk + ChiSquaredAttributeEval | RMSE 39,46 · r² −0,226 | RMSE 35,65 · r² 0 | **nulo**: peor que la media |
| `clf_indice_bajo` (F1 macro) | SMO + ChiSquaredAttributeEval | 0,475 (+0,135) | 0,339 | supera la base, pero CV 10 = 0,394 ± 0,055 y AUC 0,546: apenas distingue |
| `clf_indice_nivel` (F1 macro) | NaiveBayes + WrapperSubsetEval | 0,109 (−0,118) | 0,226 | **nulo**: peor que la clase mayoritaria |

El diagnóstico de brecha train–test coincide en los dos casos de clasificación
(`ok` y `nulo`) y las curvas de aprendizaje no mejoran al agregar datos
(`validación` empeora levemente en las tres tareas). Friedman detecta diferencias
entre algoritmos (`p ≤ 0,0006`), pero se trata de diferencias entre **modelos que no
aprenden**: elegir el mejor hiperparámetro no crea señal donde no la hay.

La conclusión es sobre el instrumento, no sobre el algoritmo: para estimar el
desempeño del 3er grado hay que unir el puntaje de mathematics (`PM3.csv`), como ya
hace el bloque de modelos de `artefactos/`. Estos tres pipelines quedan persistidos
como línea base documentada, no como sistema de puntuación.

## Ejecución

El entorno virtual ya creado es `entorno/` (Python 3.12). Para instalarlo/actualizarlo
desde cero o sobre un entorno nuevo:

```bash
# con uv (recomendado)
uv pip install --python entorno/bin/python -r <(uv export)

# o con pip
python -m pip install -r requirements.txt
```

Ejecutar la app:

```bash
entorno/bin/python -m streamlit run app.py
```

La app quedará en `http://localhost:8501`.

Abrir el cuaderno Jupyter:

```bash
entorno/bin/python -m jupyter lab cuadernos/analisis_preguntas.ipynb
```

> El kernel del cuaderno está registrado como `erce` (se instala con
> `entorno/bin/python -m ipykernel install --user --name erce`).

## Metodología

- **Puntajes**: media de los 5 valores plausibles (VP1–VP5) por estudiante.
- **Medias ponderadas** con el peso de muestreo `WSEN`.
- **Errores estándar** mediante réplicas BRR (factor de Fay 0.5), combinando varianza de
  muestreo y varianza de imputación (valores plausibles).
- **Pruebas**: t de Student (Welch) con d de Cohen, ANOVA de una vía con η², chi-cuadrado
  con V de Cramér, y correlación de Pearson.
- **Escala**: media regional 700, DE 100.

> **Nota**: Cuba no participó en el TERCE 2013, por lo que la comparación de evolución se
> realiza frente al promedio regional del ERCE 2019 (fuente: Informe Ejecutivo Regional).
>
> Las correlaciones e índices se interpretan como **asociaciones observacionales**, no
> como relaciones causales.