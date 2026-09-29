# -*- coding: utf-8 -*-
"""Configuración del bloque de construcción de modelos ERCE 2019 – Cuba, 3er grado.

Define rutas, semilla, umbrales del proceso fiel a Weka (IR, balanceo, CV),
las variables objetivo ("tareas") y la paleta de los gráficos publicables.

Metodología (Weka):
- IR = total instancias mayoritarias / total instancias minoritarias de la clase.
- SI IR <= 1.2  -> sin balancear.
- SI 1.2 < IR <= 5  -> SMOTE.
- SI 5   < IR <= 9  -> SMOTE + submuestreo (RandomUnderSampler).
- SI IR > 9         -> SMOTEENN (SMOTE + Edited Nearest Neighbours).
- Siempre se compara contra el modelo sin balancear y se conserva el mejor por F1.
- CV de selección: k-fold estratificada. Evaluación final: 10-fold (default Weka)
  + holdout reservado (estimación imparcial; nunca se usa para seleccionar).
"""
import os

# ---------------------------------------------------------------- rutas
MODELOS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODELOS_DIR)
DATOS_DIR = os.path.join(MODELOS_DIR, "datos")
ARTEFACTOS_DIR = os.path.join(BASE_DIR, "artefactos")
REPORTE_DIR = os.path.join(ARTEFACTOS_DIR, "reporte")

# ---------------------------------------------------------------- proceso
SEED = 42
CV_FOLDS_SELECCION = 5           # CV interna para selección y Friedman
CV_FOLDS_FINAL = 10              # CV final fiel a Weka (default)
REPETICIONES_VALIDACION = 3      # 3x5 CV del ganador: media +/- DE
HOLDOUT_FRACCION = 0.2           # reserva de test imparcial (estratificada si clasificación)
UMBRAL_LOG = {1.2, 5.0, 9.0}     # umbrales de IR
IR_BALANCEADO = 1.2
IR_SMOTE_MAX = 5.0
IR_SMOTE_RUS_MAX = 9.0
K_SELECTOR = 8                   # número de atributos por defecto en Ranker/SelectKBest
MAX_WRAPPER_FEATURES = 8         # tope de atributos del WrapperSubsetEval
PRESELECCION_MAX = 20            # tope de candidatos del ranking previo (CFS)
PRESELECCION_WRAPPER = 12        # tope de candidatos del wrapper (más caro)
BREAK_OVERFIT = 0.08             # brecha train-test (abs) que dispara alerta de sobreajuste
MUESTRA_PERMUTACION = 1500       # límite de filas para permutation importance

# ---------------------------------------------------------------- features
# Fuente: Módulo Nacional de Matemática (CSV adicionado) + contexto
# sociodemográfico de la base ERCE (FA_A3). La variable objetivo reportada en
# últimos columnas de los datasets: PM3 / NL_VP1 / REPC.
_FILTRO_SOLO_NUMERICAS = (
    # Estudiante y su contexto escolar (CSV del módulo)
    "edad", "sec_3", "can_prof_3", "prof_hom_3", "prof_muj_3", "num_hab_esc",
    # Índices del estudiante, del docente y del directivo (CSV del módulo)
    "Indice_global", "Indice_preg", "Indice_refuerzo",
    "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
    "Indice_calidad_clases_dir",
    # Ítems del Módulo Nacional (estudiante)
    "E3IT21_01_N", "E3IT21_02_N", "E3IT21_03_N", "E3IT21_04_N", "E3IT21_05_N",
    # Ítems del Módulo Nacional (docente)
    "PPIT44_01_N", "PPIT44_02_N", "PPIT44_03_N", "PPIT44_04_N", "PPIT44_05_N",
    "PPIT45_N",
    "PPIT46_01_N", "PPIT46_02_N", "PPIT46_03_N", "PPIT46_04_N", "PPIT46_05_N",
    # Ítems del Módulo Nacional (directivo)
    "DDIT41_01_N", "DDIT41_02_N", "DDIT41_03_N", "DDIT41_04_N", "DDIT41_05_N",
    # Contexto sociodemográfico (base ERCE)
    "ISECF", "EDU", "PREE", "AUSE", "ATRE", "LIBH", "TSTU",
)

FEATURES_NUMERICOS = list(_FILTRO_SOLO_NUMERICAS)
FEATURES_CATEGORICOS = ["provincia", "género", "area_esc"]
FEATURES = FEATURES_NUMERICOS + FEATURES_CATEGORICOS

# Agrupación para el relato del artículo/cliente
GRUPOS_FEATURES = {
    "Contexto sociodemográfico": ["ISECF", "EDU", "PREE", "AUSE", "ATRE",
                                  "LIBH", "TSTU", "provincia"],
    "Índices de calidad": ["Indice_global", "Indice_preg", "Indice_refuerzo",
                           "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
                           "Indice_calidad_clases_dir"],
    "Ítems del Módulo Nacional": [
        "E3IT21_01_N", "E3IT21_02_N", "E3IT21_03_N", "E3IT21_04_N", "E3IT21_05_N",
        "PPIT44_01_N", "PPIT44_02_N", "PPIT44_03_N", "PPIT44_04_N", "PPIT44_05_N",
        "PPIT45_N",
        "PPIT46_01_N", "PPIT46_02_N", "PPIT46_03_N", "PPIT46_04_N", "PPIT46_05_N",
        "DDIT41_01_N", "DDIT41_02_N", "DDIT41_03_N", "DDIT41_04_N", "DDIT41_05_N",
    ],
    "Contexto escolar": ["sec_3", "can_prof_3", "prof_hom_3", "prof_muj_3",
                         "num_hab_esc"],
    "Estudiante": ["edad", "género", "area_esc"],
}

# ---------------------------------------------------------------- tareas
# Cada tarea tiene su objetivo como ÚLTIMA columna del dataset; `columna_origen`
# es la variable del modelo maestro y `derivada` expresa cómo construir una binaria.
TAREAS_CLASIFICACION = {
    "clf_nivel": {
        "columna": "NL_VP1", "descripcion": "Nivel de desempeño Matemática ERCE (I–IV)",
        "tipo": "clasificacion", "grupo": "Nivel de desempeño",
    },
    "clf_riesgo": {
        "columna": None, "descripcion": "En riesgo: Nivel I vs resto",
        "derivada": lambda s: s["NL_VP1"].eq("I").astype(float).where(s["NL_VP1"].notna()),
        "tipo": "clasificacion", "grupo": "Riesgo educativo",
    },
    "clf_superacion": {
        "columna": None, "descripcion": "Alto desempeño: Nivel III/IV vs resto",
        "derivada": lambda s: s["NL_VP1"].isin(["III", "IV"]).astype(float).where(s["NL_VP1"].notna()),
        "tipo": "clasificacion", "grupo": "Alto desempeño",
    },
    "clf_repitencia": {
        "columna": "REPC", "descripcion": "Reptencia: repitió el año (caso desbalanceado)",
        "tipo": "clasificacion", "grupo": "Trayectoria educativa",
    },
}
TAREAS_REGRESION = {
    "reg_mat": {
        "columna": "PM3", "descripcion": "Puntaje Matemática según módulo y contexto",
        "tipo": "regresion", "grupo": "Puntaje Matemática",
    },
}
TAREAS = {**TAREAS_CLASIFICACION, **TAREAS_REGRESION}

# Nombres de los 4 selectores (evaluadores de atributos estilo Weka)
SELECTORES_NOMBRES = ["cfs", "chi2", "infogain", "wrapper"]

# ---------------------------------------------------------------- paleta
PALETA = {
    "azul": "#14448c",
    "azul_claro": "#4a9be8",
    "azul_muy_claro": "#eaf2fb",
    "marino": "#12223b",
    "rojo": "#c0392b",
    "ambar": "#b8860b",
    "verde": "#1c7c4d",
    "gris": "#5b6168",
    "gris_claro": "#d7dbe2",
    "fondo": "#f6f8fc",
    "blanco": "#ffffff",
}