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
├── analisis_preguntas.ipynb  # Cuaderno Jupyter: 8 preguntas con proceder y gráficos
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
entorno/bin/python -m jupyter lab analisis_preguntas.ipynb
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