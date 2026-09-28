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
_FILTRO_SOLO_NUMERICAS = (
    "ISECF", "EDAD", "AAEG3", "ORGEN", "EFMAT", "VIOES", "INVAP",
    "PREE", "AUSE", "ATRE", "LIBH", "TSTU",
    "Indice_global", "Indice_preg", "Indice_refuerzo",
    "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
    "Indice_calidad_clases_dir",
    # Ítems del Módulo Nacional
    "E3IT21_01_N", "E3IT21_02_N", "E3IT21_03_N", "E3IT21_04_N", "E3IT21_05_N",
    "PPIT44_01_N", "PPIT44_02_N", "PPIT44_03_N", "PPIT44_04_N", "PPIT44_05_N",
    "PPIT45_N",
    "PPIT46_01_N", "PPIT46_02_N", "PPIT46_03_N", "PPIT46_04_N", "PPIT46_05_N",
    "DDIT41_01_N", "DDIT41_02_N", "DDIT41_03_N", "DDIT41_04_N", "DDIT41_05_N",
    # Contexto escolar del Módulo Nacional
    "sec_3", "can_prof_3", "prof_hom_3", "prof_muj_3", "num_hab_esc",
    # Ítems del cuestionario de estudiantes (E3IT*) — se cargan dinámicamente
    # desde FA_A3, excluyendo E3IT21_01/02 (subsumidos por el módulo).
)

# Ítems E3IT* del cuestionario de estudiantes (el módulo aporta E3IT21_*_N).
_ITEMS_FA = [
    "E3IT01", "E3IT02", "E3IT03", "E3IT04", "E3IT05", "E3IT06", "E3IT07",
    "E3IT08_01", "E3IT08_02", "E3IT08_03", "E3IT08_04",
    "E3IT09_01", "E3IT09_02", "E3IT09_03", "E3IT09_04", "E3IT09_05", "E3IT09_06",
    "E3IT10_01", "E3IT10_02", "E3IT10_03",
    "E3IT11_01", "E3IT11_02", "E3IT11_03", "E3IT11_04",
    "E3IT12_01", "E3IT12_02", "E3IT12_03", "E3IT12_04", "E3IT12_05",
    "E3IT13_01", "E3IT13_02", "E3IT13_03", "E3IT13_04", "E3IT13_05",
    "E3IT13_06", "E3IT13_07",
    "E3IT14_01", "E3IT14_02", "E3IT14_03",
    "E3IT15_01", "E3IT15_02", "E3IT15_03", "E3IT15_04", "E3IT15_05",
    "E3IT15_06", "E3IT15_07",
    "E3IT16_01", "E3IT16_02", "E3IT16_03", "E3IT16_04", "E3IT16_05",
    "E3IT17",
    "E3IT18_01", "E3IT18_02", "E3IT18_03", "E3IT18_04",
    "E3IT19", "E3IT19A", "E3IT19B", "E3IT20",
]

FEATURES_NUMERICOS = list(_FILTRO_SOLO_NUMERICAS) + _ITEMS_FA
FEATURES_CATEGORICOS = ["provincia", "SEX", "RURAL", "EDU"]
FEATURES = FEATURES_NUMERICOS + FEATURES_CATEGORICOS

# Agrupación para el relato del artículo/cliente
GRUPOS_FEATURES = {
    "Condiciones de vida": ["ISECF", "LIBH", "TSTU", "ATRE", "AUSE",
                            "PREE", "EDAD", "EDU", "provincia"],
    "Contexto del estudiante": ["AAEG3", "ORGEN", "EFMAT", "VIOES", "INVAP",
                                "SEX", "RURAL"],
    "Calidad escolar percibida": ["Indice_global", "Indice_preg", "Indice_refuerzo",
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
    "Ítems del cuestionario de estudiantes": _ITEMS_FA,
}

# ---------------------------------------------------------------- tareas
# Cada tarea tiene su objetivo como ÚLTIMA columna del dataset; `columna_origen`
# es la variable del modelo maestro y `derivada` expresa cómo construir una binaria.
TAREAS_CLASIFICACION = {
    "clf_nivel": {
        "columna": "NL_VP1", "descripcion": "Nivel de desempeño ERCE (I–IV)",
        "tipo": "clasificacion", "grupo": "Nivel de desempeño",
    },
    "clf_riesgo": {
        "columna": None, "descripcion": "En riesgo: Nivel I vs resto",
        "derivada": lambda s: (s["NL_VP1"] == "I").astype(int),
        "tipo": "clasificacion", "grupo": "Riesgo educativo",
    },
    "clf_superacion": {
        "columna": None, "descripcion": "Alto desempeño: Nivel III/IV vs resto",
        "derivada": lambda s: s["NL_VP1"].isin(["III", "IV"]).astype(int),
        "tipo": "clasificacion", "grupo": "Alto desempeño",
    },
    "clf_repitencia": {
        "columna": "REPC", "descripcion": "Reptencia: repitió el año (caso desbalanceado)",
        "tipo": "clasificacion", "grupo": "Trayectoria educativa",
    },
}
TAREAS_REGRESION = {
    "reg_lect": {
        "columna": "PL3", "descripcion": "Puntaje Lectura según condiciones de vida",
        "tipo": "regresion", "grupo": "Puntaje Lectura",
    },
    "reg_mat": {
        "columna": "PM3", "descripcion": "Puntaje Matemática según condiciones de vida",
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