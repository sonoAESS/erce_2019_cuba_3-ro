# -*- coding: utf-8 -*-
"""App Streamlit: Estudio estadístico ERCE 2019 – Cuba, 3er grado."""
import streamlit as st

from erce import analysis as A
from erce import data
from erce import interpret as I
from erce import plots as P

st.set_page_config(
    page_title="ERCE 2019 · Cuba – 3er grado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Cargando datos ERCE 2019...")
def _cargar_datos():
    return (
        data.dataframe_estudiantes(),
        data.dataframe_escuelas(),
        data.dataframe_modulo(),
        data.dataframe_regional(),
    )


@st.cache_data(show_spinner="Calculando estadísticas...")
def _estadisticas():
    df, esc, mod, reg = _cargar_datos()
    res = A.resumen_nacional(df)
    return {
        "df": df, "esc": esc, "mod": mod, "reg": reg, "resumen": res,
        "tab_lect_prov": A.tabla_resumen(df, "PL3", "ubi1_cen_esc"),
        "tab_mat_prov": A.tabla_resumen(df, "PM3", "ubi1_cen_esc"),
        "anova_lect_prov": A.prueba_anova(df, "PL3", "ubi1_cen_esc"),
        "anova_mat_prov": A.prueba_anova(df, "PM3", "ubi1_cen_esc"),
        "t_lect_genero": A.prueba_t_grupos(df, "PL3", "genero"),
        "t_mat_genero": A.prueba_t_grupos(df, "PM3", "genero"),
        "t_lect_zona": A.prueba_t_grupos(df, "PL3", "zona"),
        "t_mat_zona": A.prueba_t_grupos(df, "PM3", "zona"),
        "corr_lect": A.correlaciones(df, ["ISECF", "EDAD", "AAEG3", "ORGEN", "EFMAT", "VIOES", "INVAP"], "PL3"),
        "corr_mat": A.correlaciones(df, ["ISECF", "EDAD", "AAEG3", "ORGEN", "EFMAT", "VIOES", "INVAP"], "PM3"),
        "cuartiles_lect": A.cuartiles_isect(df, "PL3"),
        "cuartiles_mat": A.cuartiles_isect(df, "PM3"),
        "corr_ind_lect": A.correlaciones(mod, ["Indice_global", "Indice_preg", "Indice_refuerzo",
                                               "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
                                               "Indice_calidad_clases_dir"], "PL3"),
        "corr_ind_mat": A.correlaciones(mod, ["Indice_global", "Indice_preg", "Indice_refuerzo",
                                              "Indice_prep_didact_prof", "Indice_calidad_clases_prof",
                                              "Indice_calidad_clases_dir"], "PM3"),
        "chi2_edu_zona": A.prueba_chi2(df, "edu_padres", "zona"),
        "chi2_gen_zona": A.prueba_chi2(df, "genero", "zona"),
    }


# ---------------------------------------------------------------- utilidades
def _html(texto, color="#1f3a93"):
    st.markdown(f'<div style="border-left:4px solid {color}; padding:8px 14px; '
                f'background:#f6f8fb; border-radius:4px; margin:6px 0;">{texto}</div>',
                unsafe_allow_html=True)


def _html_interpretacion(interp, color="#1f3a93"):
    """Renderiza un dict {"analisis": str, "conclusion": str} en dos bloques."""
    analisis = interp.get("analisis", "")
    if analisis:
        _html(analisis, color)
    conclusion = interp.get("conclusion", "")
    if conclusion:
        st.markdown(
            f'<div style="border-left:4px solid #1c7c4d; background:#eef7f0; '
            f'padding:8px 14px; border-radius:4px; margin:2px 0 10px 0;">{conclusion}</div>',
            unsafe_allow_html=True,
        )


def _tarjeta_kpi(titulo, valor, sub, color):
    st.markdown(
        f"""<div style="background:white;border-radius:10px;padding:14px 18px;
        box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:4px solid {color};height:100%">
        <div style="color:#666;font-size:.85rem">{titulo}</div>
        <div style="font-size:1.9rem;font-weight:700;color:{color}">{valor}</div>
        <div style="color:#888;font-size:.8rem">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def _titulo_seccion(icono, titulo, sub):
    st.markdown(f"### {icono} {titulo}")
    st.caption(sub)


def _nota_metodologica(texto):
    with st.expander("ℹ️ Nota metodológica", expanded=False):
        st.markdown(texto)


# ================================================================= PORTADA
def seccion_portada(S):
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1f3a93 0%,#2e5bb0 60%,#4aa3df 100%);
        padding:34px 36px;border-radius:14px;color:white;margin-bottom:20px">
        <div style="font-size:.95rem;letter-spacing:2px;opacity:.85">ESTUDIO ESTADÍSTICO · ERCE 2019</div>
        <div style="font-size:2.3rem;font-weight:800;margin:6px 0">Rendimiento en <span style="color:#ffd166">Lectura</span> y
        <span style="color:#ff8a80">Matemática</span> · 3er grado</div>
        <div style="font-size:1.05rem;opacity:.92">Resultados de Cuba en el Estudio Regional Comparativo y Explicativo (UNESCO – LLECE),
        con análisis territorial, socioeconómico, de género y de calidad escolar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    res = S["resumen"]
    n = res["Lectura"]["n"]
    with c1:
        _tarjeta_kpi("Estudiantes evaluados", f"{n:,}", "3er grado, ambos sexos", P.COLORES["azul"])
    with c2:
        _tarjeta_kpi("Lectura · media", res["Lectura"]["media"], f"DE {res['Lectura']['de']} · EE {res['Lectura']['ee']}", P.COLORES["azul"])
    with c3:
        _tarjeta_kpi("Matemática · media", res["Matemática"]["media"], f"DE {res['Matemática']['de']} · EE {res['Matemática']['ee']}", P.COLORES["rojo"])
    with c4:
        esc = S["esc"]
        _tarjeta_kpi("Escuelas", f"{esc['IDSCHOOL'].nunique():,}", "con coordenadas geográficas", P.COLORES["verde"])

    st.markdown("---")
    st.markdown("#### Resumen ejecutivo")
    _html_interpretacion(I.interpretar_nacional(res), P.COLORES["azul"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(P.grafico_distribucion(S["df"], "PL3", "Distribución de puntajes · Lectura", P.COLORES["azul"]), width='stretch')
    with c2:
        st.plotly_chart(P.grafico_distribucion(S["df"], "PM3", "Distribución de puntajes · Matemática", P.COLORES["rojo"]), width='stretch')

    _nota_metodologica(
        "Los puntajes corresponden a la media de los 5 valores plausibles por estudiante. "
        "Las medias y errores estándar usan el peso de muestreo WSEN y el método de réplicas "
        "BRR (factor de Fay 0.5), combinando varianza de muestreo e imputación. "
        "La escala regional tiene media 700 y desviación estándar 100."
    )


# ============================================================== TERRITORIO
def seccion_territorio(S):
    _titulo_seccion("🗺️", "Rendimiento por territorio",
                    "Diferencias entre provincias y entre zonas urbana y rural.")
    sel = st.radio("Prueba", ["Lectura", "Matemática"], horizontal=True, key="ter_sel")
    score = "PL3" if sel == "Lectura" else "PM3"
    color = P.PALETA_PRUEBAS[sel]

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(P.grafico_barras_medias(
            S[f"tab_{'lect' if sel=='Lectura' else 'mat'}_prov"], "ubi1_cen_esc", sel, color,
            f"Puntaje medio por provincia · {sel}"), width='stretch')
    with c2:
        st.plotly_chart(P.grafico_caja(S["df"], "zona", score, color,
                                       f"Zona urbana vs rural · {sel}"), width='stretch')

    anova = S[f"anova_{'lect' if sel=='Lectura' else 'mat'}_prov"]
    t_zona = S[f"t_{'lect' if sel=='Lectura' else 'mat'}_zona"]
    de = S["resumen"][sel]["de"]
    _html_interpretacion(
        I.interpretar_territorio(S[f"tab_{'lect' if sel=='Lectura' else 'mat'}_prov"], anova, sel, de),
        P.COLORES["azul"])
    _html_interpretacion(I.interpretar_zona(t_zona, sel, de), P.COLORES["verde"])

    with st.expander("Tabla completa por provincia"):
        st.dataframe(S[f"tab_{'lect' if sel=='Lectura' else 'mat'}_prov"], width='stretch', hide_index=True)

    _nota_metodologica(
        "Las medias por provincia son ponderadas (WSEN) con error estándar BRR. "
        "La prueba ANOVA usa el puntaje medio por estudiante sin ponderar, "
        "por lo que es una aproximación conservadora del efecto de la provincia."
    )


# ============================================================= SOCIOECONOMICO
def seccion_socioeconomico(S):
    _titulo_seccion("🏠", "Brechas socioeconómicas",
                    "Relación entre el índice socioeconómico de la familia (ISECF) y el rendimiento.")
    sel = st.radio("Prueba", ["Lectura", "Matemática"], horizontal=True, key="soc_sel")
    score = "PL3" if sel == "Lectura" else "PM3"
    color = P.PALETA_PRUEBAS[sel]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(P.grafico_isect_cuartiles(
            S[f"cuartiles_{'lect' if sel=='Lectura' else 'mat'}"], color,
            f"Puntaje medio por cuartil de ISECF · {sel}"), width='stretch')
    with c2:
        sub = S["df"].dropna(subset=["ISECF", score]).sample(1800, random_state=42)
        st.plotly_chart(P.grafico_correlacion(sub, "ISECF", score, color,
                                              f"ISECF vs puntaje · {sel}"), width='stretch')

    _html_interpretacion(I.interpretar_isect(
        S[f"cuartiles_{'lect' if sel=='Lectura' else 'mat'}"],
        S[f"corr_{'lect' if sel=='Lectura' else 'mat'}"], sel,
        S["resumen"][sel]["de"]), P.COLORES["azul"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(P.grafico_barras_apiladas_educacion(
            S["df"], "Educación de los padres por nivel de desempeño en Lectura"), width='stretch')
    with c2:
        chi = S["chi2_edu_zona"]
        st.markdown("##### Educación de los padres y zona")
        _html(
            f"La educación de los padres está **asociada a la zona** (urbana/rural): "
            f"chi² = {chi['chi2']}, p {I.fmt_p(chi['p'])}; V de Cramér = {chi['v_cramer']:.2f}.",
            P.COLORES["ambar"],
        )
        st.markdown("""
        - **Terciaria** = al menos un padre con estudios de nivel terciario.
        - **≤ Secundaria** = nivel educativo menor.
        - El ISECF (índice socioeconómico de la familia) está estandarizado (media 0, DE 1).
        """)

    _nota_metodologica(
        "El ISECF se construye a partir de bienes del hogar, educación y ocupación de los padres. "
        "Los cuartiles se calculan sobre los estudiantes con ISECF válido. "
        "La correlación es de Pearson sobre el puntaje medio (valores plausibles)."
    )


# ============================================================== GENERO
def seccion_genero(S):
    _titulo_seccion("👧👦", "Género y factores del estudiante",
                    "Brechas de género y vínculo de factores del estudiante con el rendimiento.")
    sel = st.radio("Prueba", ["Lectura", "Matemática"], horizontal=True, key="gen_sel")
    score = "PL3" if sel == "Lectura" else "PM3"
    color = P.PALETA_PRUEBAS[sel]
    t_gen = S[f"t_{'lect' if sel=='Lectura' else 'mat'}_genero"]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(P.grafico_caja(S["df"], "genero", score, color,
                                       f"Distribución por género · {sel}"), width='stretch')
    with c2:
        st.markdown("##### Brecha de género")
        _html_interpretacion(I.interpretar_genero(t_gen, sel, S["resumen"][sel]["de"]),
                             P.COLORES["azul"])
        chi = S["chi2_gen_zona"]
        st.markdown(f"""
        - **Niñas** se definen como SEX = 1; **Niños** SEX = 0 (cuestionario ERCE).
        - La distribución por género es equilibrada en la muestra.
        - Asociación género × zona: chi² = {chi['chi2']}, p {I.fmt_p(chi['p'])}.
        """)

    st.markdown("##### Factores del estudiante y su asociación con el puntaje")
    _html_interpretacion(I.interpretar_correlaciones(S[f"corr_{'lect' if sel=='Lectura' else 'mat'}"], sel),
                         P.COLORES["verde"])

    corr = S[f"corr_{'lect' if sel=='Lectura' else 'mat'}"].copy()
    corr["variable"] = corr["variable"].map({
        "ISECF": "Índice socioeconómico (ISECF)",
        "EDAD": "Edad",
        "AAEG3": "Apoyo al aprendizaje",
        "ORGEN": "Organización de la enseñanza",
        "EFMAT": "Autoeficacia en Matemática",
        "VIOES": "Violencia en la escuela",
        "INVAP": "Involucramiento parental",
    }).fillna(corr["variable"])
    st.dataframe(corr, width='stretch', hide_index=True)

    _nota_metodologica(
        "La prueba t de Welch compara los puntajes medios de niñas y niños sobre los valores "
        "plausibles promediados; el d de Cohen cuantifica el tamaño de la brecha. "
        "Las correlaciones son de Pearson; un signo negativo indica menor puntaje asociado "
        "a mayores valores de la variable."
    )


# ============================================================== CALIDAD
def seccion_calidad(S):
    _titulo_seccion("🏫", "Calidad escolar y docente (Módulo Nacional)",
                    "Índices del Módulo Nacional de Cuba y su relación con el rendimiento.")
    mod = S["mod"]
    sel = st.radio("Prueba", ["Lectura", "Matemática"], horizontal=True, key="cal_sel")
    score = "PL3" if sel == "Lectura" else "PM3"
    color = P.PALETA_PRUEBAS[sel]
    idx_cols = ["Indice_global", "Indice_preg", "Indice_refuerzo",
                "Indice_prep_didact_prof", "Indice_calidad_clases_prof", "Indice_calidad_clases_dir"]

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.plotly_chart(P.grafico_barras_indices_promedio(
            mod, idx_cols, color, "Promedio de los índices de calidad (0–100)"), width='stretch')
    with c2:
        st.plotly_chart(P.grafico_indices(mod, score, color,
                                          f"Puntaje según quintil de cada índice · {sel}"), width='stretch')

    _html_interpretacion(I.interpretar_indices(S[f"corr_ind_{'lect' if sel=='Lectura' else 'mat'}"], sel),
                         P.COLORES["azul"])

    with st.expander("Correlaciones detalladas con los índices"):
        st.dataframe(S[f"corr_ind_{'lect' if sel=='Lectura' else 'mat'}"], width='stretch', hide_index=True)

    _nota_metodologica(
        "Los índices del Módulo Nacional (0–100) provienen de las respuestas de docentes y "
        "directivos (Cuestionario del Profesor QP3 y del Director QD3). Miden percepciones "
        "sobre la calidad pedagógica; su relación con los puntajes es observacional."
    )


# ============================================================== MAPA
def seccion_mapa(S):
    _titulo_seccion("📍", "Distribución geográfica del rendimiento",
                    "Puntaje medio por escuela sobre el mapa de Cuba.")
    esc = S["esc"]
    sel = st.radio("Prueba", ["Lectura", "Matemática"], horizontal=True, key="map_sel")
    score = "PL3" if sel == "Lectura" else "PM3"
    st.plotly_chart(P.grafico_mapa(esc, score, f"Puntaje medio por escuela · {sel}"),
                    width='stretch')
    _html(
        f"Se representan **{esc[score].notna().sum()} escuelas**. Las burbujas más oscuras "
        f"corresponden a mayor rendimiento en {sel}. Las provincias de Pinar del Río, La Habana "
        f"y Matanzas concentran la mayor cantidad de escuelas de la muestra.",
        P.COLORES["verde"],
    )
    _nota_metodologica(
        "Coordenadas geográficas de las escuelas de la muestra cubana (archivo de coordenadas). "
        "El mapa usa estilos de OpenStreetMap a través de Plotly Mapbox."
    )


# ============================================================== CONCLUSIONES
def seccion_conclusiones(S):
    _titulo_seccion("✅", "Conclusiones por pregunta de interés",
                    "Respuestas a las preguntas del estudio, el método que las sustenta y dónde ver el detalle.")

    for i, c in enumerate(I.conclusiones_finales(S), start=1):
        with st.container():
            st.markdown(f"#### {i}. {c['pregunta']}")
            _html(c["respuesta"], P.COLORES["azul"])
            col_met, col_sec = st.columns([1.6, 1])
            with col_met:
                st.markdown(
                    f"<div style='background:#f3f4f6;border-radius:6px;padding:8px 12px;"
                    f"font-size:.85rem'><b>🔬 Método:</b> {c['metodo']}</div>",
                    unsafe_allow_html=True,
                )
            with col_sec:
                st.markdown(
                    f"<div style='background:#eef3ff;border-radius:6px;padding:8px 12px;"
                    f"font-size:.85rem'><b>📍 Detalle en:</b> {c['seccion']} "
                    f"<span style='color:#777'>— {c['detalle']}</span></div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"Ir a «{c['seccion']}»", key=f"ir_{i}"):
                    st.session_state["ir_a"] = c["seccion"]
            st.markdown("---")


# ============================================================== REGIONAL
def seccion_regional(S):
    _titulo_seccion("🌎", "Cuba en el contexto regional",
                    "Resultados del ERCE 2019 frente al promedio de la región.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(P.grafico_regional(S["reg"]), width='stretch')
    with c2:
        st.markdown("##### Posición de Cuba")
        _html_interpretacion(I.interpretar_regional(S["reg"], S["resumen"]), P.COLORES["rojo"])
        st.info(
            "**Nota sobre evolución histórica:** Cuba no participó en el TERCE 2013, "
            "por lo que no es posible estimar directamente la mejora o deterioro respecto "
            "al ciclo anterior. El contraste se realiza frente al promedio regional del ERCE 2019."
        )
    st.markdown("""
    #### Interpretación adicional
    - En **Matemática 3°**, Cuba presentó el puntaje promedio **más alto entre los países participantes** (751) y el 75% de sus estudiantes quedó por encima del Nivel I.
    - En **Lectura 3°**, el puntaje medio (730) se ubicó también por encima del promedio regional (697).
    - La **dispersión interna** es alta: las diferencias entre estudiantes dentro de Cuba son mayores que entre países, lo que apunta a brechas territoriales y socioeconómicas internas.
    """)
    _nota_metodologica(
        "Cifras regionales extraídas del Informe Ejecutivo Regional ERCE 2019 (UNESCO/LLECE). "
        "Los promedios regionales corresponden al conjunto de países participantes."
    )


# ================================================================== MAIN
def main():
    df, esc, mod, reg = _cargar_datos()
    S = _estadisticas()

    if "ir_a" in st.session_state:
        st.session_state["nav_seccion"] = st.session_state.pop("ir_a")

    with st.sidebar:
        st.markdown("### 📊 ERCE 2019 · Cuba")
        st.markdown("**3er grado** · Lectura y Matemática")
        st.markdown("---")
        seccion = st.radio(
            "Secciones",
            [
                "Inicio / Resumen",
                "Conclusiones por pregunta",
                "Rendimiento por territorio",
                "Brechas socioeconómicas",
                "Género y factores",
                "Calidad escolar / docente",
                "Mapa geográfico",
                "Contexto regional",
            ],
            label_visibility="collapsed",
            key="nav_seccion",
        )
        st.markdown("---")
        st.caption("Fuente: UNESCO-LLECE, ERCE 2019. "
                   "Base de datos Cuba, 3er grado. Análisis propio.")

    if seccion == "Inicio / Resumen":
        seccion_portada(S)
    elif seccion == "Conclusiones por pregunta":
        seccion_conclusiones(S)
    elif seccion == "Rendimiento por territorio":
        seccion_territorio(S)
    elif seccion == "Brechas socioeconómicas":
        seccion_socioeconomico(S)
    elif seccion == "Género y factores":
        seccion_genero(S)
    elif seccion == "Calidad escolar / docente":
        seccion_calidad(S)
    elif seccion == "Mapa geográfico":
        seccion_mapa(S)
    elif seccion == "Contexto regional":
        seccion_regional(S)


if __name__ == "__main__":
    main()
