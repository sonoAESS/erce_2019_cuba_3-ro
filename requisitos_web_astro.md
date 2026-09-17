# Requisitos y arquitectura — Web Astro estática "ERCE 2019 · Cuba – 3er grado"

Documento de requisitos funcionales y no funcionales, ingeniería, arquitectura y
tecnologías para reproducir la app Streamlit actual (`app.py` + paquete `erce/`)
como un sitio 100 % estático con Astro, alimentado por un libro Excel.

---

## 1. Requisitos funcionales (RF)

### RF-1 · Datos y pipeline de generación (build-time)

| ID | Descripción | Criterio de aceptación |
|----|-------------|------------------------|
| RF-1.1 | Consolidar las fuentes CSV del proyecto (`07.ERCE-2019-FINAL/`) en un libro Excel `src/data/erce.xlsx` con hojas `estudiantes`, `escuelas`, `modulo` y `regional`, aplicando la misma limpieza y convenciones de `erce/data.py`. | El script genera el Excel de forma reproducible desde la raíz del proyecto. |
| RF-1.2 | Script que calcula todos los resultados que hoy produce la app en runtime y los exporta a JSON: KPIs nacionales, tabla por provincia (n, Media, EE, IC95, DE), ANOVA (F, p, η²), t de Welch por zona y género (t, p, d de Cohen), correlaciones de Pearson (r, p, n), cuartiles ISECF, χ² con V de Cramér, promedios de índices del Módulo Nacional y puntaje por quintil de cada índice, tabla regional. | Cada JSON tiene esquema y tipos definidos; todo se regenera con un solo comando. |
| RF-1.3 | Los textos interpretativos de `erce/interpret.py` se pre-renderizan a JSON (no se portan a JavaScript). | El texto mostrado es idéntico al de la app Streamlit para los mismos números. |
| RF-1.4 | El spec JSON de cada figura Plotly se exporta desde `erce/plots.py`. | Al renderizar, la figura es idéntica en apariencia, hover y zoom a la generada por Plotly en vivo. |
| RF-1.5 | La submuestra fija de la dispersión ISECF (`sample(1800, random_state=42)`) se precalcula y empaqueta. | Los puntos de la dispersión son estables entre builds e idénticos a la app actual. |
| RF-1.6 | **El Excel es la fuente de verdad editable por el host.** El script de exportación lee solo `erce.xlsx` (hojas y columnas con esquema validado) y regenera todos los JSON a partir de él; la consolidación CSV→Excel es un script de arranque independiente que **nunca** se ejecuta en CI (para no sobrescribir ediciones del host). | Un cambio en `erce.xlsx` se refleja en el sitio tras el siguiente build, sin tocar código. |
| RF-1.7 | El build genera `metadata.json` con la **fecha de última actualización** (derivada de la fecha de modificación del Excel o de un campo configurado) y se muestra en el pie/portada del sitio. | El sitio informa de cuándo se actualizaron los datos. |
| RF-1.8 | Validación del Excel antes de exportar: comprobación de hojas, columnas obligatorias, tipos numéricos y ausencia de valores en blanco en columnas críticas. | Ante un Excel mal formado, el build **falla con mensaje claro en español** indicando qué hoja/columna corregir; no publica datos rotos. |

### RF-2 · Navegación y estructura

- **RF-2.1** Ocho páginas/rutas 1:1 con las secciones de `app.py`:
  `Inicio/Resumen`, `Conclusiones por pregunta`, `Rendimiento por territorio`,
  `Brechas socioeconómicas`, `Género y factores`, `Calidad escolar / docente`,
  `Mapa geográfico` y `Contexto regional`.
- **RF-2.2** Navegación lateral persistente (sidebar en escritorio, menú superior en
  móvil) con título, subtítulo y nota de fuente idénticos a la app original.
- **RF-2.3** En "Conclusiones por pregunta", los botones "Ir a…" funcionan como
  enlaces anclados entre secciones.

### RF-3 · Portada (Inicio/Resumen)

- Hero con degradado azul institucional (texto, leyenda y copetes idénticos).
- Cuatro tarjetas KPI: estudiantes evaluados, Lectura media (DE/EE), Matemática
  media (DE/EE) y número de escuelas.
- Resumen ejecutivo (bloque análisis + conclusión) y dos histogramas de
  distribución con línea de mediana.
- Nota metodológica colapsable.

### RF-4 · Secciones analíticas

- **RF-4.1** Toggle **Lectura/Matemática** (island) en las secciones de territorio,
  socioeconómico, género y calidad, que conmuta: datasets JSON, color azul/rojo,
  textos interpretativos y figuras.
- **RF-4.2 Territorio**: barras por provincia con IC95 + referencia 700, boxplot
  urbano/rural, interpretaciones, tabla completa en acordeón y nota metodológica.
- **RF-4.3 Socioeconómico**: medias por cuartil de ISECF, dispersión ISECF vs
  puntaje, barras apiladas de educación por nivel de desempeño y bloque χ²
  (educación × zona).
- **RF-4.4 Género**: boxplot por género, texto de brecha con d de Cohen y tabla de
  correlaciones de factores del estudiante con etiquetas traducidas.
- **RF-4.5 Calidad**: promedios de índices (0–100), puntaje medio por quintil de
  cada índice y correlaciones detalladas en acordeón.
- **RF-4.6 Regional**: barras agrupadas Cuba vs Región, interpretación y nota
  histórica (TERCE 2013).
- **RF-4.7** Todos los acordeones (tabla completa, correlaciones detalladas, notas
  metodológicas) implementados con `<details>` o componente accordion.

### RF-5 · Mapa geográfico

- Burbujas georreferenciadas por escuela, coloreadas por puntaje (escala Reds),
  con hover de nombre/provincia/municipio/puntaje, centrado en Cuba y zoom ~5.2.

---

## 2. Requisitos no funcionales (RNF)

### RNF-1 · Rendimiento

- Tiempo de carga inicial < 2 s (red 4G media) por página.
- Lazy-load y code-splitting: plotly.js (~1.5 MB) se carga solo donde hay gráficos,
  tras el primer *idle*.
- Assets servidos con compresión gzip/brotli; sin dependencia de Plotly en páginas
  que no lo usen.

### RNF-2 · Fiabilidad y consistencia

- Verificación por snapshot: script que compara los JSON exportados contra los
  valores que produce la app Streamlit actual; el build falla si difieren.
- El build falla si falta cualquier JSON requerido o si su esquema no valida.
- El build falla si la validación del Excel (RF-1.8) detecta errores estructurales.
- Reproducibilidad: el Excel y los JSON se regeneran con el mismo resultado desde
  las mismas fuentes.

### RNF-3 · Seguridad y privacidad

- Al cliente solo viajan agregados (JSON); nunca microdato por estudiante.
- Cero secretos o tokens: el mapa usa tiles públicos de OpenStreetMap (sin clave).
- Sin servidor ni base de datos: superficie de ataque mínima.

### RNF-4 · Accesibilidad y usabilidad

- Contraste de la paleta institucional que cumpla al menos AA en textos.
- Responsive: layout "wide" en escritorio y reflujo correcto en tablet y móvil.

### RNF-5 · Mantenibilidad

- Un solo comando regenera todo: `python scripts/exportar_datos.py && npm run build`.
- Separación clara datos/tipos/UI: JSON tipados en TypeScript y componentes aislados.
- El proyecto web convive con `erce/` sin modificarlo; `erce/` sigue siendo la
  fuente de verdad analítica.
- Dos flujos claramente separados: **arranque** (CSV→Excel, solo desarrollador) y
  **actualización** (Excel→JSON→sitio, lo único que usa el host).

### RNF-6 · Despliegue y portabilidad

- Salida 100 % estática (`dist/`), desplegable en GitHub Pages, Netlify, Vercel o
  cualquier hosting estático.
- No requiere Python ni Node en el servidor.
- La regeneración de datos y el build corren en **CI** (GitHub Actions/Netlify/
  Vercel), de modo que el host no necesita instalar nada en su máquina.

### RNF-7 · Compatibilidad

- Navegadores modernos (dos últimas versiones de Chrome, Firefox, Safari, Edge).
- Tiles OSM con fallback: si se bloquean, el mapa muestra mensaje de carga y los
  datos siguen disponibles en las tablas.

---

## 3. Ingeniería del proyecto (proceso)

**Fases:**

1. **Preparación de datos** — script de arranque `importar_csv.py` (CSVs → `erce.xlsx`)
   y script de actualización `exportar_datos.py` (Excel → JSON) con validación de
   esquema (RF-1.8). Paso de mayor riesgo; primero a validar.
2. **Snapshot de referencia** — capturar los valores actuales de la app Streamlit
   como *golden files*.
3. **Esqueleto Astro** — ocho páginas, layout, sidebar y estilos base con la paleta
   institucional.
4. **Islands** — `PlotlyChart`, `MapLibreMap`, `TogglePrueba`, `Accordion`, `KPI`,
   `BlocInterpretacion`.
5. **Enchufe de datos** — cada página consume sus JSON; verificación visual contra
   la app Streamlit.
6. **Mapa** — migrar `Scattermapbox` (requiere token) a MapLibre + OSM.
7. **QA y validación final** — snapshot completo, build limpio y prueba manual de
   las ocho secciones y del toggle; simular la edición del Excel por el host y
   comprobar que el CI publica el sitio actualizado.
8. **Deploy** — publicación estática.

**Verificación en cada fase:** comparación JSON ↔ Streamlit, `astro build` y
revisión manual.

---

## 4. Arquitectura

```
┌─ Arranque (solo desarrollador, fuera de CI) ─────────────────────┐
│  CSVs raw (07.ERCE-2019-FINAL)                                   │
│     └─ scripts/importar_csv.py  (consolidación única)            │
│           └─ src/data/erce.xlsx   ← fuente verdad EDITABLE       │
└──────────────────────────────────────────────────────────────────┘
┌─ Build time (CI: GitHub Actions / Netlify / Vercel) ─────────────┐
│  el host sube/reemplaza src/data/erce.xlsx y hace commit         │
│     └─ scripts/exportar_datos.py  (lee SOLO el Excel; valida)    │
│           ├─ public/data/stats/*.json  (tablas + textos)         │
│           ├─ public/data/figures/*.json (specs Plotly)           │
│           └─ public/data/metadata.json (fecha de actualización)  │
│                                                                    │
│  Astro build ──> src/pages/*.astro + islands ──> dist/ (estático)│
└────────────────────────────────────────────────────────────────────┘
┌─ Runtime (navegador) ─────────────────────────────────────────────┐
│  HTML/CSS + islands hidratadas:                                   │
│   • PlotlyChart  → plotly.js renderiza fig.json                   │
│   • MapLibreMap  → burbujas sobre tiles OSM (sin token)           │
│   • TogglePrueba → conmuta Lectura/Matemática (JSON local)        │
│   • Accordion/KPI→ sin JS o hidratación mínima                    │
│  Import de JSON estáticos + interpolación de textos               │
└────────────────────────────────────────────────────────────────────┘
```

**Principio clave:**

> Todo lo computacional vive en **build-time** (Python) y todo lo interactivo vive
> en **client-side** (islands) sobre datos ya calculados.

---

## 5. Tecnologías

| Capa | Tecnología | Justificación |
|------|-----------|---------------|
| Framework estático | **Astro 5** | HTML-first, cero JS por defecto, islands multi-framework, datos estáticos y `getStaticPaths`. |
| Islands | **React 19** (alternativas: Vue, Svelte) | Charting y mapa interactivos solo donde hace falta. |
| Generación de datos | **Python 3.12** (reutiliza `erce/`; pandas, numpy, scipy, openpyxl) | Sin reescritura de la lógica estadística; valida contra la app actual. |
| Libro Excel | **openpyxl** (hojas por dataset + validación de columnas) | Cumple RF-1.1 con esquema controlado. |
| Gráficos | **plotly.js** (`newPlot` con `fig.json`) o build `plotly-basic` | Fidelidad 1:1 con la app Streamlit. |
| Mapa | **MapLibre GL JS** + tiles OSM raster | Interactivo sin claves/secretos. |
| Estilos | **CSS/Tailwind** + variables de diseño (paleta institucional) | Contraste, reuso y tipografía Segoe UI/Helvetica. |
| Tipos | **TypeScript** con tipos generados desde los JSON | Evita desajustes y valida esquemas en build. |
| Verificación | Script Python de **snapshot** (JSON vs Streamlit) + `astro check` | RNF-2. |
| CI/Deploy | GitHub Actions → `astro build` + publicación estática | Reproducible y sin servidor. (Hosting a definir.) |
| Bundling | Vite (incluido en Astro), code-splitting por página/isla | Controla el peso de plotly.js. |

---

## 6. Decisiones de diseño adoptadas

- Los gráficos se reproducen **exportando el spec JSON de cada figura Plotly** desde
  Python y renderizándolos con plotly.js (máxima fidelidad).
- El mapa es **interactivo con MapLibre + tiles OSM** (sin token de Mapbox).
- El Excel es la **piedra angular editable**: el host solo toca `erce.xlsx` y el CI
  lo convierte en el sitio. La consolidación CSV→Excel es un paso de arranque único
  que corre fuera de CI para no pisar las ediciones del host.

---

## 7. Actualización de datos por el host

El mantenedor del sitio **no necesita saber de Python ni de Node**. Actualizar los
datos es solo sustituir el archivo `src/data/erce.xlsx` y republicar.

### 7.1 Flujo general (con repositorio + CI)

1. El host descarga `src/data/erce.xlsx` del repositorio (o mantiene una copia local).
2. Abre el archivo en Excel/LibreOffice y edita las hojas `estudiantes`, `escuelas`,
   `modulo` y `regional`. **No debe** renombrar hojas, borrar columnas esperadas ni
   cambiar tipos numéricos a texto.
3. Sobrescribe el archivo en el repositorio y hace commit (o lo sube por la interfaz
   web de GitHub).
4. La CI (GitHub Actions, Netlify o Vercel) detecta el cambio y ejecuta
   `exportar_datos.py` + `astro build` automáticamente.
   - Si el Excel es válido → publica el nuevo sitio.
   - Si el Excel falla la validación → el build se detiene y se muestra el error
     indicando la hoja/columna a corregir. El sitio publicado **no se rompe**: sigue
     sirviendo la última versión buena.

### 7.2 Variantes según hosting

- **GitHub Pages / Netlify / Vercel conectados a un repo:** flujo 7.1. Es la opción
  recomendada: totalmente automática y con historial/restauración de versiones del Excel.
- **Netlify Drop / Vercel CLI (sin repo):** el flujo 7.1 no aplica. En su lugar, se
  provee un comando local que el host ejecuta en su equipo
  (`python scripts/exportar_datos.py && npm run build`) y arrastra la carpeta `dist/`.
  Requiere tener Python y Node instalados; documentado en el README del proyecto web.
- **Editor sencillo opcional:** si el perfil del host es no técnico, se puede añadir
  un pequeño *worksheet* o script de ayuda que rellene una plantilla Excel desde un
  formulario y genere `erce.xlsx`; queda fuera del alcance inicial como mejora.

### 7.3 Garantías para el host

- **Validación previa a publicar** (RF-1.8) con mensajes en español ("Falta la
  columna `WSEN` en la hoja `estudiantes`"), para no publicar datos incorrectos.
- **Fecha de última actualización** visible en el sitio (RF-1.7), para que el host
  confirme que su edición se reflejó.
- El CSV original nunca interviene en la actualización: solo el Excel. Así, las
  ediciones hechas directamente sobre `erce.xlsx` no se pierden en ningún momento.