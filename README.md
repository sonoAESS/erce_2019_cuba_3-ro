# Análisis estadístico ERCE 2019 – Cuba, 3er grado

Estudio estadístico interactivo sobre el rendimiento en **Lectura** y **Matemática** de los
estudiantes cubanos de **3er grado** en el **ERCE 2019** (UNESCO-LLECE), con gráficos
profesionales (Plotly), análisis inferencial e interpretaciones automáticas, presentado en
una app **Streamlit**.

## Preguntas que responde el estudio

- **Rendimiento por territorio**: diferencias entre provincias (ANOVA) y entre zona urbana/rural (t de Welch).
- **Brechas socioeconómicas**: gradiente del puntaje según el índice socioeconómico de la familia (ISECF) y la educación de los padres.
- **Género y factores del estudiante**: brechas niño/niña y correlaciones con edad, apoyo al aprendizaje, violencia escolar, involucramiento parental, entre otros.
- **Calidad escolar / docente**: índices del Módulo Nacional (global, preguntas, refuerzo, calidad de clases) y su vínculo con los puntajes.
- **Mapa geográfico**: puntaje medio por escuela sobre el mapa de Cuba.
- **Contexto regional**: Cuba frente al promedio de la región en el ERCE 2019.

## Estructura

```
├── app.py                 # App Streamlit (secciones + conclusiones por pregunta)
├── analisis_preguntas.ipynb  # Cuaderno Jupyter: 8 preguntas de interés con proceder y gráficos
├── erce/
│   ├── config.py          # Rutas, constantes, etiquetas
│   ├── data.py            # Carga y limpieza de datos (LLECE + Módulo Nacional)
│   ├── analysis.py        # Estadística descriptiva e inferencial (medias ponderadas, BRR, t, ANOVA, chi2, Pearson)
│   ├── plots.py           # Figuras Plotly profesionales
│   └── interpret.py       # Interpretaciones automáticas y conclusiones
├── pyproject.toml         # Dependencias (uv)
└── uv.lock                # Lockfile
```

> Los directorios `07.ERCE-2019-FINAL/` (bases de datos) y `2. Libros de código/`
> (codebooks) **no se versionan** en el repositorio; deben copiarse aparte en la raíz
> del proyecto para que la app y el cuaderno funcionen.

## Ejecución

El entorno virtual ya creado es `entorno/` (Python 3.12). Para instalarlo/actualizarlo
desde cero o sobre un entorno nuevo:

```bash
uv pip install --python entorno/bin/python -r <(uv export)
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
