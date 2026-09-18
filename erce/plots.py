# -*- coding: utf-8 -*-
"""Figuras Plotly profesionales para el estudio ERCE 2019 – Cuba, 3er grado."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta institucional (azul Cuba / rojo bandera, con neutros)
COLORES = {
    "azul": "#1f3a93",
    "celeste": "#4aa3df",
    "rojo": "#c8102e",
    "rojo_suave": "#e5726a",
    "verde": "#2e8b57",
    "ambar": "#e6a817",
    "gris": "#8c8c8c",
    "fondo": "#fbfbfa",
}

PALETA_PRUEBAS = {"Lectura": COLORES["azul"], "Matemática": COLORES["rojo"]}


def _layout(fig, titulo=None, alto=520, xlab=None, ylab=None, leyenda=True,
            margen=None):
    m = {"l": 40, "r": 20, "t": 60, "b": 40}
    if margen:
        m.update(margen)
    fig.update_layout(
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font": {"size": 18, "color": "#222222"}},
        height=alto,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 13, "color": "#333333"},
        xaxis={"title": xlab, "gridcolor": "#e8e8e8", "zeroline": False},
        yaxis={"title": ylab, "gridcolor": "#e8e8e8", "zeroline": False},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"} if leyenda else {},
        margin=m,
    )
    return fig


def _anotar_referencia(fig, valor, texto, orientacion="v"):
    """Etiqueta de una línea de referencia, colocada fuera del área de datos."""
    if orientacion == "v":
        fig.add_annotation(
            x=valor, y=1.04, yref="paper", text=texto, showarrow=False,
            font={"size": 12, "color": COLORES["gris"]},
            xanchor="center", yanchor="bottom", bgcolor="rgba(255,255,255,0.85)",
            borderpad=3, bordercolor="#e8e8e8",
        )
    else:
        fig.add_annotation(
            y=valor, x=1.02, xref="paper", text=texto, showarrow=False,
            font={"size": 12, "color": COLORES["gris"]},
            yanchor="middle", xanchor="left", bgcolor="rgba(255,255,255,0.85)",
            borderpad=3, bordercolor="#e8e8e8",
        )


def grafico_distribucion(df, score, titulo, color):
    """Histograma + KDE de densidad del puntaje."""
    fig = px.histogram(
        df, x=score, nbins=45, color_discrete_sequence=[color],
        opacity=0.75, marginal="violin",
    )
    fig.update_traces(marker_line=dict(width=1, color="white"))
    med = df[score].median()
    fig.add_vline(x=med, line_dash="dash", line_color=COLORES["gris"])
    _anotar_referencia(fig, med, f"Mediana {med:.0f}")
    return _layout(fig, titulo, alto=460, xlab="Puntaje (escala ERCE, media regional 700)")


def grafico_barras_medias(tabla, col_grupo, score_label, color, titulo):
    """Barras horizontales con la media por grupo."""
    tabla = tabla.sort_values("Media")
    fig = go.Figure(go.Bar(
        x=tabla["Media"], y=tabla[col_grupo], orientation="h",
        marker_color=color, marker_line_width=0,
        text=tabla["Media"].round(0).astype(int),
        textposition="inside",
        textfont={"color": "white", "size": 11},
        insidetextanchor="middle",
        cliponaxis=False,
        hovertemplate="%{y}<br>Media: %{x:.1f}<extra></extra>",
    ))
    fig.add_vline(x=700, line_dash="dot", line_color=COLORES["gris"])
    _anotar_referencia(fig, 700, "Media regional 700")
    xmax = tabla["Media"].max()
    xmin = min(0, tabla["Media"].min())
    fig.update_xaxes(range=[xmin, xmax * 1.03])
    return _layout(fig, titulo, alto=120 + 34 * len(tabla), xlab=f"Puntaje medio ({score_label})")


def grafico_caja(df, x, y, color, titulo, puntos="suspectedoutliers"):
    """Diagrama de caja por grupo."""
    fig = px.box(df, x=x, y=y, color=x, color_discrete_sequence=[color], points=puntos)
    fig.add_hline(y=700, line_dash="dot", line_color=COLORES["gris"])
    fig.update_traces(line_width=1.6)
    return _layout(fig, titulo, alto=480, ylab="Puntaje")


def grafico_medias_grupo(tabla, col_grupo, color, titulo, xlab=None):
    """Barras verticales con la media ponderada por grupo, la n y la
    referencia regional (más informativo que la caja de puntajes).

    `tabla` es el resultado de `analysis.tabla_resumen(df, score, col_grupo)`.
    """
    filas = tabla.sort_values("Media", ascending=False)
    cats = filas[col_grupo].astype(str).tolist()
    medias = filas["Media"].astype(float).tolist()
    n = filas["n"].astype(int).tolist()

    fig = go.Figure(go.Bar(
        x=cats,
        y=medias,
        marker_color=color,
        text=[f"{m:.0f}" for m in medias],
        textposition="outside",
        cliponaxis=False,
        textfont={"color": "#222222", "size": 13},
        customdata=list(zip(n)),
        hovertemplate=(
            "<b>%{x}</b><br>Media: %{y:.1f}<br>n: %{customdata[0]}<extra></extra>"
        ),
    ))

    fig.add_hline(y=700, line_dash="dot", line_color=COLORES["gris"])
    _anotar_referencia(fig, 700, "Referencia regional 700", orientacion="h")

    if len(cats) == 2:
        dif = medias[0] - medias[1]
        fig.add_annotation(
            x=cats[0], y=max(medias) + 12,
            text=f"Δ {dif:+.0f} puntos",
            showarrow=False, font={"size": 12, "color": "#333333"},
            bgcolor="rgba(255,255,255,0.85)", borderpad=3, bordercolor="#e8e8e8",
        )

    fig.update_yaxes(range=[600, (max(medias) + 60 if len(cats) == 2 else max(medias) + 30)])
    return _layout(fig, titulo, alto=430, xlab=xlab, ylab="Puntaje medio")


def grafico_pareto_niveles(df, nivel_col, titulo):
    """Barras apiladas 100% de niveles de desempeño por grupo."""
    tabla = pd.crosstab(df[nivel_col].fillna("Sin dato"), df["genero"].fillna("Sin dato"),
                        normalize="columns") * 100
    orden = ["I", "II", "III", "IV"]
    colores = [COLORES["rojo_suave"], COLORES["ambar"], COLORES["celeste"], COLORES["azul"]]
    fig = go.Figure()
    for i, nivel in enumerate(orden):
        if nivel in tabla.index:
            vals = tabla.loc[nivel]
            textos = [f"{v:.1f}%" if v >= 8 else "" for v in vals]
            fig.add_trace(go.Bar(
                name=f"Nivel {nivel}", x=tabla.columns, y=vals,
                marker_color=colores[i], text=textos,
                textposition="inside", textfont={"color": "white", "size": 10},
                cliponaxis=False,
            ))
    fig.update_layout(barmode="stack")
    return _layout(fig, titulo, alto=460, ylab="% de estudiantes")


def grafico_isect_cuartiles(cuartiles, color, titulo):
    """Medias por cuartil del índice socioeconómico."""
    fig = go.Figure(go.Bar(
        x=cuartiles["Cuartil"].astype(str), y=cuartiles["Media"],
        marker_color=[COLORES["azul"], COLORES["celeste"], COLORES["rojo_suave"], COLORES["rojo"]],
        text=cuartiles["Media"].round(0).astype(int), textposition="outside",
        cliponaxis=False, textfont={"color": "#222222"},
    ))
    fig.add_hline(y=700, line_dash="dot", line_color=COLORES["gris"])
    _anotar_referencia(fig, 700, "Media regional 700", orientacion="h")
    ymax = cuartiles["Media"].max()
    fig.update_yaxes(range=[600, ymax * 1.08])
    return _layout(fig, titulo, alto=440, xlab="Cuartil del índice socioeconómico (ISECF)", ylab="Puntaje medio")


def grafico_correlacion(df, x, y, color, titulo):
    """Dispersión con regresión lineal y banda de confianza."""
    fig = px.scatter(
        df, x=x, y=y, opacity=0.25, color_discrete_sequence=[color], trendline="ols",
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.4, color="white")))
    return _layout(fig, titulo, alto=440, xlab=x, ylab=y)


def grafico_barras_apiladas_educacion(df, titulo):
    """Distribución de la educación de los padres por nivel de logro (Lectura)."""
    tabla = pd.crosstab(df["NL_VP1"].fillna("Sin dato"), df["edu_padres"].fillna("Sin dato"), normalize="index") * 100
    fig = go.Figure()
    orden = ["≤ Secundaria", "Terciaria"]
    for cat in orden:
        if cat in tabla.columns:
            vals = tabla[cat]
            textos = [f"{v:.0f}%" if v >= 10 else "" for v in vals]
            fig.add_trace(go.Bar(
                name=cat, x=tabla.index, y=vals,
                marker_color=COLORES["rojo_suave"] if "Secundaria" in cat else COLORES["azul"],
                text=textos, textposition="inside", textfont={"color": "white", "size": 10},
                cliponaxis=False,
            ))
    fig.update_layout(barmode="stack")
    return _layout(fig, titulo, alto=460, xlab="Nivel de desempeño en Lectura", ylab="% de estudiantes")


def grafico_indices(df, score, color, titulo):
    """Puntaje medio por decil de cada índice de calidad del Módulo Nacional."""
    fig = go.Figure()
    for idx in ["Indice_global", "Indice_preg", "Indice_refuerzo",
                "Indice_prep_didact_prof", "Indice_calidad_clases_prof", "Indice_calidad_clases_dir"]:
        sub = df[df[idx].notna() & df[score].notna()].copy()
        if len(sub) < 20:
            continue
        sub["decil"] = pd.qcut(sub[idx], 5, labels=False, duplicates="drop") + 1
        medias = sub.groupby("decil")[score].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=medias["decil"], y=medias[score], mode="lines+markers",
            name=idx.replace("Indice_", "Índice "), line=dict(width=2.4),
        ))
    return _layout(fig, titulo, alto=480, xlab="Quintil del índice (1 = menor, 5 = mayor)", ylab="Puntaje medio")


def grafico_regional(df_regional):
    """Barras agrupadas Cuba vs Región por área."""
    fig = go.Figure()
    for col, color in [("Cuba", COLORES["azul"]), ("Región", COLORES["gris"])]:
        fig.add_trace(go.Bar(
            name=col, x=df_regional["Área"], y=df_regional[col],
            marker_color=color,
            text=df_regional[col].astype(int), textposition="outside",
            cliponaxis=False, textfont={"color": "#222222"},
        ))
    fig.update_layout(barmode="group")
    ymax = df_regional[["Cuba", "Región"]].max().max()
    fig.update_yaxes(range=[0, ymax * 1.12])
    return _layout(fig, "Cuba frente al promedio regional – ERCE 2019, 3er grado", alto=440, ylab="Puntaje medio")


def grafico_mapa(df_esc, score, titulo):
    """Mapa de burbujas con coordenadas de escuelas (coloreadas por puntaje)."""
    fig = go.Figure(go.Scattermapbox(
        lat=df_esc["geo_esc_x"], lon=df_esc["geo_esc_y"],
        mode="markers", text=df_esc["nom_esc"],
        customdata=np.stack([df_esc[score], df_esc["ubi1_cen_esc"], df_esc["ubi2_cen_esc"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>Provincia: %{customdata[1]}<br>Municipio: %{customdata[2]}"
            f"<br>{score}: %{{customdata[0]:.1f}}<extra></extra>"
        ),
        marker=dict(
            size=13, sizemode="diameter",
            color=df_esc[score], colorscale="Reds",
            cmin=df_esc[score].min(), cmax=df_esc[score].max(),
            colorbar=dict(title=score, thickness=14, len=0.7),
        ),
    ))
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": df_esc["geo_esc_x"].mean(), "lon": df_esc["geo_esc_y"].mean()},
            zoom=5.2,
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=640,
    )
    return fig


def grafico_barras_indices_promedio(df, idx_cols, color, titulo):
    """Promedio (0-100) de los índices de calidad del Módulo Nacional."""
    medias = df[idx_cols].mean().sort_values()
    etiquetas = {
        "Indice_global": "Índice global de calidad",
        "Indice_preg": "Preguntas y diálogo pedagógico",
        "Indice_refuerzo": "Refuerzo pedagógico",
        "Indice_prep_didact_prof": "Preparación didáctica docente",
        "Indice_calidad_clases_prof": "Calidad de clases (docente)",
        "Indice_calidad_clases_dir": "Calidad de clases (directivo)",
    }
    fig = go.Figure(go.Bar(
        x=medias.values, y=[etiquetas.get(i, i) for i in medias.index], orientation="h",
        marker_color=color, text=medias.round(1), textposition="outside",
        cliponaxis=False, textfont={"color": "#222222"},
    ))
    fig.add_vline(x=50, line_dash="dot", line_color=COLORES["gris"])
    _anotar_referencia(fig, 50, "Punto medio (50)")
    fig.update_xaxes(range=[0, medias.max() * 1.12])
    return _layout(fig, titulo, alto=160 + 36 * len(medias), xlab="Puntaje medio del índice (0-100)")
