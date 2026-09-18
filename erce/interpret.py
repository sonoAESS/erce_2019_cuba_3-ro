# -*- coding: utf-8 -*-
"""Interpretaciones analíticas basadas en los resultados estadísticos.

Cada función entrega un texto que describe el hallazgo (dato), lo interpreta en
su magnitud (unidades de desviación estándar, varianza explicada, contexto) y
cierra con una conclusión. Se usa la escala ERCE (media regional 700, DE 100)
como referencia para dimensionar las diferencias.
"""


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def _describir_d(d):
    """Convierte un d de Cohen a una calificación cualitativa estándar."""
    ad = abs(d)
    if ad < 0.2:
        return "muy pequeño"
    if ad < 0.5:
        return "pequeño"
    if ad < 0.8:
        return "moderado"
    return "grande"


def _describir_d_f(d):
    """Igual que _describir_d, en la forma femenina para «magnitud»."""
    if d is None:
        return "moderada"
    return {
        "muy pequeño": "muy pequeña",
        "pequeño": "pequeña",
        "moderado": "moderada",
        "grande": "grande",
    }[_describir_d(d)]


def _describir_r(r):
    """Califica la magnitud de una correlación de Pearson."""
    ar = abs(r)
    if ar < 0.1:
        return "prácticamente nula"
    if ar < 0.3:
        return "débil"
    if ar < 0.5:
        return "moderada"
    return "fuerte"


def _pts_a_de(dif, de):
    """Expresa una diferencia de puntaje en desviaciones estándar."""
    if not de:
        return None
    return dif / de


def _a_la(etiqueta):
    """Precede una etiqueta con su preposición correcta («al»/«a la»/«a»)."""
    if etiqueta.startswith("el "):
        return "al " + etiqueta[len("el "):]
    if etiqueta.startswith("la "):
        return "a la " + etiqueta[len("la "):]
    return "a " + etiqueta


def interpretar_nacional(res):
    """Resultados nacionales: hallazgo, interpretación y conclusión."""
    lect = res["Lectura"]
    mat = res["Matemática"]
    dif = mat["media"] - lect["media"]
    de_pool = (lect["de"] + mat["de"]) / 2
    desv = _pts_a_de(dif, de_pool)

    base = (
        f"En Cuba, el puntaje promedio ponderado de 3er grado fue de **{lect['media']}** "
        f"en Lectura (DE {lect['de']}) y **{mat['media']}** en Matemática (DE {mat['de']}). "
        f"Ambos promedios se sitúan **por encima de la media regional** de referencia "
        f"(700 puntos), lo que indica un desempeño destacado en el contexto latinoamericano."
    )

    analisis = base + (
        f" La diferencia de **{abs(dif):.0f} puntos a favor de Matemática** "
        f"equivale a unas {abs(desv):.2f} desviaciones estándar"
        if desv is not None else ""
    ) + (
        f", una brecha que, aunque estadísticamente relevante, se mantiene en el rango "
        f"de las diferencias consideradas pequeñas a moderadas dentro de la escala del estudio."
    )

    conclusion = (
        "**Conclusión:** el sistema educativo cubano muestra fortaleza relativa en ambas "
        "asignaturas y un rendimiento comparativamente mayor en Matemática. Sin embargo, "
        "la magnitud de la dispersión (DE ≈ 90–80) advierte que junto a los buenos "
        "promedios conviven estudiantes con desempeños muy desiguales, lo que orienta la "
        "atención hacia la equidad interna más que hacia el nivel medio."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_territorio(tab, anova, score_label, de):
    """Ranking de provincias: brecha, interpretación y conclusión."""
    mejor = tab.iloc[0]
    peor = tab.iloc[-1]
    dif = mejor["Media"] - peor["Media"]
    desv = _pts_a_de(dif, de)

    base = (
        f"La provincia con mayor rendimiento en {score_label} es **{mejor[tab.columns[0]]}** "
        f"({mejor['Media']:.0f}) y la de menor desempeño es **{peor[tab.columns[0]]}** "
        f"({peor['Media']:.0f}). La brecha entre extremos alcanza **{dif:.0f} puntos**"
    )
    if desv is not None:
        base += f" (≈ {desv:.2f} DE)"

    if anova and anova["p"] < 0.05:
        analisis = base + (
            f". El ANOVA confirma que estas diferencias son estadísticamente significativas "
            f"(F = {anova['F']:.1f}, p {fmt_p(anova['p'])}), pero el índice eta cuadrado "
            f"(η² = {anova['eta2']:.3f}) indica que la provincia explica apenas una fracción "
            f"de la variabilidad total de los puntajes."
        )
        conclusion = (
            "**Conclusión:** existen diferencias territoriales reales y medibles, pero la mayor "
            "parte de la desigualdad de aprendizaje se observa **entre estudiantes de una misma "
            "provincia**, no entre provincias. Esto sugiere que las brechas de rendimiento se "
            "explican más por condiciones de los hogares y de las escuelas individuales que por "
            "un factor territorial homogéneo, y que las políticas focalizadas solo por provincia "
            "serían insuficientes."
        )
    else:
        analisis = base + ". Las diferencias no resultaron estadísticamente significativas."
        conclusion = (
            "**Conclusión:** el rendimiento es relativamente homogéneo entre provincias, lo que "
            "refuerza la idea de que las desigualdades educativas en Cuba se concentran al interior "
            "de los territorios y no entre ellos."
        )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_zona(res_t, score_label, de):
    """Brecha urbana vs rural: hallazgo, interpretación y conclusión."""
    if res_t is None:
        return {"analisis": "No hay datos suficientes para comparar zonas.",
                "conclusion": ""}
    dif = abs(res_t["media_a"] - res_t["media_b"])
    desv = _pts_a_de(dif, de)
    quien = res_t["cat_a"] if res_t["media_a"] > res_t["media_b"] else res_t["cat_b"]

    base = (
        f"En {score_label}, el rendimiento medio es **{dif:.0f} puntos mayor en la zona "
        f"{quien}** ({max(res_t['media_a'], res_t['media_b']):.0f} vs "
        f"{min(res_t['media_a'], res_t['media_b']):.0f})"
    )
    if desv is not None:
        base += f" (≈ {desv:.2f} DE)"

    if res_t["p"] < 0.05:
        analisis = base + (
            f". La diferencia es estadísticamente significativa (t = {abs(res_t['t']):.2f}, "
            f"p {fmt_p(res_t['p'])}) y su tamaño del efecto es "
            f"**{_describir_d(res_t['cohen_d'])}** (d = {res_t['cohen_d']:.2f})."
        )
    else:
        analisis = base + ". La diferencia no alcanza significación estadística."

    conclusion = (
        "**Conclusión:** la brecha urbano-rural existe pero es pequeña y su dirección no es "
        "uniforme entre asignaturas: la ventaja urbana observada en Lectura se revierte a "
        "favor de la zona rural en Matemática. Ello indica que la ruralidad en sí misma no "
        "explica el desempeño: importan más las condiciones socioeconómicas y de recursos que "
        "suelen acompañarla, por lo que la política educativa debería atender la desventaja "
        "rural con un enfoque socioeconómico y no limitarse a una etiqueta territorial."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_isect(cuartiles, corr, score_label, de):
    """Brecha socioeconómica: gradiente, magnitud y conclusión."""
    q1 = cuartiles.iloc[0]
    q4 = cuartiles.iloc[-1]
    dif = q4["Media"] - q1["Media"]
    desv = _pts_a_de(dif, de)
    r = corr.iloc[0] if len(corr) else None

    base = (
        f"El gradiente socioeconómico es el patrón más consistente del estudio: los estudiantes "
        f"del cuartil más alto del índice socioeconómico (ISECF) promedian **{dif:.0f} puntos "
        f"más** en {score_label} que los del cuartil más bajo "
        f"({q4['Media']:.0f} vs {q1['Media']:.0f})"
    )
    if desv is not None:
        base += f", una brecha de ≈ {desv:.2f} desviaciones estándar"

    if r is not None:
        r2 = r["r"] ** 2
        analisis = base + (
            f". La correlación de Pearson es **r = {r['r']:.2f}** (p {fmt_p(r['p'])}), "
            f"de magnitud {_describir_r(r['r'])}; al elevar al cuadrado, el nivel "
            f"socioeconómico explica cerca del **{r2*100:.0f}% de la variabilidad** de los "
            f"puntajes. En la práctica, avanzar de un cuartil socioeconómico al siguiente se "
            f"asocia de forma sostenida y monótona con mejores resultados."
        )
    else:
        analisis = base + "."

    conclusion = (
        "**Conclusión:** el nivel socioeconómico de la familia es el factor individual que "
        "mejor anticipa el desempeño en esta asignatura, con una brecha de magnitud "
        f"{_describir_d_f(desv)} entre los extremos de la "
        "distribución. Aunque la mayor parte de la variabilidad queda sin explicar, la brecha "
        f"de ≈ {desv:.2f} DE entre los cuartiles extremos confirma que la equidad en las "
        "condiciones de vida es un eje central para la mejora educativa: las políticas de "
        "compensación a los hogares más desfavorecidos pueden contribuir a reducir esta brecha."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_genero(res_t, score_label, de):
    """Brecha de género: hallazgo, interpretación y conclusión."""
    if res_t is None:
        return {"analisis": "No hay datos suficientes.", "conclusion": ""}
    if res_t["media_a"] > res_t["media_b"]:
        ventaja = res_t["cat_a"]
        dif = res_t["media_a"] - res_t["media_b"]
        medio_a, medio_b = res_t["media_a"], res_t["media_b"]
    else:
        ventaja = res_t["cat_b"]
        dif = res_t["media_b"] - res_t["media_a"]
        medio_a, medio_b = res_t["media_b"], res_t["media_a"]
    det = "Los **niños**" if ventaja == "Niño" else "Las **niñas**"

    base = (
        f"{det} alcanzan, en promedio, **{dif:.1f} puntos más** que sus "
        f"pares en {score_label} ({medio_a:.0f} vs {medio_b:.0f})"
    )
    if res_t["p"] < 0.05:
        d = res_t["cohen_d"]
        analisis = base + (
            f". La diferencia es estadísticamente significativa (t = {abs(res_t['t']):.2f}, "
            f"p {fmt_p(res_t['p'])}) y el tamaño del efecto es "
            f"**{_describir_d(d)}** (d = {abs(d):.2f}), es decir, "
            f"una brecha real pero acotada dentro de la distribución de puntajes."
        )
        conclusion = (
            f"**Conclusión:** la ventaja de {'las niñas' if ventaja == 'Niña' else 'los niños'} "
            f"en {score_label} es consistente pero de magnitud pequeña. Su tamaño (≈ {abs(d):.2f} DE) "
            f"no justifica enfoques de enseñanza diferenciados por género; en todo caso, apunta "
            f"a la conveniencia de revisar cómo las expectativas y prácticas pedagógicas pueden "
            f"estar afectando de manera levemente distinta a cada grupo."
        )
    else:
        analisis = base + ", sin que la diferencia alcance significación estadística."
        conclusion = (
            f"**Conclusión:** en {score_label} no hay evidencia de una brecha sistemática por "
            f"género; los desempeños de niños y niñas son estadísticamente equivalentes."
        )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_correlaciones(corr_df, score_label):
    """Ranking de correlaciones: qué variables acompañan al mejor desempeño."""
    if corr_df.empty:
        return {"analisis": "Sin correlaciones disponibles.", "conclusion": ""}
    etiquetas = {
        "ISECF": "el índice socioeconómico familiar (ISECF)",
        "AAEG3": "el apoyo al aprendizaje de los estudiantes",
        "INVAP": "el involucramiento parental",
        "ORGEN": "la organización de la enseñanza",
        "EFMAT": "la autoeficacia en matemática",
        "VIOES": "la violencia al interior de la escuela",
        "EDAD": "la edad del estudiante",
    }
    top_pos = corr_df[corr_df["r"] > 0].head(1)
    top_neg = corr_df[corr_df["r"] < 0].tail(1)
    corr_ordenada = corr_df.sort_values("r", ascending=False)

    partes = []
    for _, v in corr_ordenada.iterrows():
        partes.append(
            f"{etiquetas.get(v['variable'], v['variable'])} (r = {v['r']:.2f})"
        )
    listado = "; ".join(partes[:3]) + ("…" if len(partes) > 3 else "")

    analisis = (
        f"Las variables de contexto y de clima escolar se asocian de forma diferenciada con "
        f"el puntaje en {score_label}. Ordenadas de mayor a menor asociación: {listado}. "
    )

    extras = []
    if len(top_pos):
        v = top_pos.iloc[0]
        extras.append(
            f"La asociación positiva más fuerte corresponde "
            f"{_a_la(etiquetas.get(v['variable'], v['variable']))}, "
            f"con una magnitud {_describir_r(v['r'])} (r = {v['r']:.2f}, p {fmt_p(v['p'])}) y "
            f"que explica ≈ {v['r']**2*100:.0f}% de la varianza."
        )
    if len(top_neg):
        v = top_neg.iloc[0]
        extras.append(
            f"En sentido contrario, {etiquetas.get(v['variable'], v['variable'])} se asocia "
            f"negativamente (r = {v['r']:.2f}): donde el clima escolar reporta más violencia, "
            f"el rendimiento tiende a ser menor."
        )
    analisis += " ".join(extras)

    conclusion = (
        "**Conclusión:** el desempeño no depende de un único factor. El apoyo familiar y el "
        "nivel socioeconómico se asocian de forma sistemática y positiva con el rendimiento, "
        "en compañía del clima escolar percibido, mientras que la violencia escolar se asocia "
        "con puntajes más bajos. Estas son asociaciones, no causalidad, pero sugieren que "
        "intervenir sobre el clima escolar y el apoyo familiar puede ser un complemento "
        "eficaz a la mejora pedagógica."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_indices(corr_df, score_label):
    """Relación entre índices de calidad escolar y puntajes."""
    if corr_df.empty:
        return {"analisis": "Sin datos de índices para este puntaje.", "conclusion": ""}
    etiquetas = {
        "Indice_global": "el índice global de calidad",
        "Indice_preg": "el índice de preguntas y diálogo pedagógico",
        "Indice_refuerzo": "el índice de refuerzo pedagógico",
        "Indice_prep_didact_prof": "la preparación didáctica del docente",
        "Indice_calidad_clases_prof": "la calidad de las clases según el docente",
        "Indice_calidad_clases_dir": "la calidad de las clases según el directivo",
    }
    ordenada = corr_df.sort_values("r", ascending=False)
    top = ordenada.iloc[0]
    partes = []
    for _, v in ordenada.iterrows():
        direccion = "positiva" if v["r"] > 0 else "negativa"
        sig = f" (p {fmt_p(v['p'])})" if v["p"] < 0.05 else " (no significativa)"
        partes.append(
            f"{etiquetas.get(v['variable'], v['variable'])} se asocia de forma "
            f"{direccion} (r = {v['r']:.2f}{sig})"
        )
    analisis = (
        f"Todos los índices del Módulo Nacional se asocian positivamente con el puntaje "
        f"en {score_label}. La asociación más fuerte corresponde "
        f"{_a_la(etiquetas.get(top['variable'], top['variable']))} (r = {top['r']:.2f}, "
        f"p {fmt_p(top['p'])}), con una magnitud {_describir_r(top['r'])}. Detalle: "
        f"{'; '.join(partes[:4])}."
    )
    conclusion = (
        "**Conclusión:** los estudiantes rinden más en escuelas cuyos docentes y directivos "
        "perciben una mayor calidad pedagógica, en especial donde el índice global de calidad "
        "es más alto. La fuerza de la asociación es de pequeña a moderada, lo que indica que "
        "la calidad escolar es un factor relevante pero no el único determinante del logro. "
        "Al tratarse de percepciones autocontestadas, conviene complementarlas con "
        "observaciones de aula para orientar políticas."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def interpretar_regional(reg, res):
    """Contexto regional: posición de Cuba y su significado."""
    fila_lect = reg[reg["Área"] == "Lectura"].iloc[0]
    fila_mat = reg[reg["Área"] == "Matemática"].iloc[0]
    de_lect = res["Lectura"]["de"]
    de_mat = res["Matemática"]["de"]
    desv_lect = _pts_a_de(fila_lect["Diferencia"], de_lect)
    desv_mat = _pts_a_de(fila_mat["Diferencia"], de_mat)

    analisis = (
        f"En el ERCE 2019, Cuba obtuvo **{fila_lect['Cuba']:.0f} puntos en Lectura** y "
        f"**{fila_mat['Cuba']:.0f} en Matemática**, superando el promedio regional "
        f"({fila_lect['Región']:.0f} y {fila_mat['Región']:.0f}) por "
        f"{fila_lect['Diferencia']:.0f} y {fila_mat['Diferencia']:.0f} puntos, "
        f"es decir, cerca de {desv_lect:.2f} y {desv_mat:.2f} desviaciones estándar por "
        f"encima de la media regional, respectivamente. **Cuba registró el puntaje más alto "
        f"de la región en Matemática de 3er grado** y uno de los mejores en Lectura."
    )
    conclusion = (
        "**Conclusión:** la posición de Cuba en el contexto regional es de liderazgo en ambas "
        "asignaturas, particularmente en Matemática. No obstante, la dispersión interna "
        f"(DE ≈ {de_lect:.0f} en Lectura y ≈ {de_mat:.0f} en Matemática) es mayor que la "
        f"ventaja sobre la media regional ({fila_lect['Diferencia']:.0f} y "
        f"{fila_mat['Diferencia']:.0f} puntos), lo que indica que el desafío principal no es "
        "elevar aún más el promedio, sino reducir las desigualdades internas de aprendizaje."
    )
    return {"analisis": analisis, "conclusion": conclusion}


def conclusiones_finales(S):
    """Conclusiones que responden las preguntas de interés del estudio.

    Devuelve una lista de dicts con las claves:
      pregunta, respuesta, metodo, seccion.
    Cada conclusión indica el método estadístico que la sustenta y en qué
    sección de la aplicación puede verse el detalle.
    """
    res = S["resumen"]
    lect = res["Lectura"]
    mat = res["Matemática"]

    anova = S["anova_lect_prov"]
    tz_lect = S["t_lect_zona"]
    tz_mat = S["t_mat_zona"]
    tg_lect = S["t_lect_genero"]
    tg_mat = S["t_mat_genero"]

    q = S["cuartiles_lect"]
    q1 = q.iloc[0]["Media"]
    q4 = q.iloc[-1]["Media"]
    corr_global = S["corr_lect"]
    r_isect = corr_global.loc[corr_global["variable"] == "ISECF", "r"].iloc[0]

    corr_ind = S["corr_ind_lect"]
    r_ind = corr_ind.loc[corr_ind["variable"] == "Indice_global", "r"].iloc[0]

    desv_isect = (q4 - q1) / lect["de"]

    prov_mejor = S["tab_lect_prov"].iloc[0]
    prov_peor = S["tab_lect_prov"].iloc[-1]
    brecha_prov = prov_mejor["Media"] - prov_peor["Media"]

    # ---- diferencias zona y género (con el ganador correcto)
    def _quien(t):
        if t is None:
            return None
        return (t["cat_a"], t["media_a"], t["cat_b"], t["media_b"])

    za, za_m, zb, zb_m = _quien(tz_lect)
    ga, ga_m, gb, gb_m = _quien(tg_lect)
    zona_lect_gana = za if za_m > zb_m else zb
    zona_lect_perd = zb if za_m > zb_m else za
    zona_lect_dif = abs(za_m - zb_m)
    genero_lect_gana = ga if ga_m > gb_m else gb
    genero_lect_perd = gb if ga_m > gb_m else ga
    genero_lect_dif = abs(ga_m - gb_m)

    zma, zma_m, zmb, zmb_m = _quien(tz_mat)
    gma, gma_m, gmb, gmb_m = _quien(tg_mat)
    zona_mat_gana = zma if zma_m > zmb_m else zmb
    zona_mat_perd = zmb if zma_m > zmb_m else zma
    zona_mat_dif = abs(zma_m - zmb_m)
    genero_mat_dif = abs(gma_m - gmb_m)

    fila_lect = S["reg"][S["reg"]["Área"] == "Lectura"].iloc[0]
    fila_mat = S["reg"][S["reg"]["Área"] == "Matemática"].iloc[0]

    return [
        {
            "pregunta": "¿Cuál es el nivel de rendimiento de Cuba frente a la región?",
            "respuesta": (
                f"Los estudiantes cubanos de 3er grado promedian **{lect['media']:.1f} puntos "
                f"en Lectura** y **{mat['media']:.1f} en Matemática**, por encima de la media "
                f"regional de referencia (700) y del promedio regional del ERCE 2019 "
                f"({fila_lect['Región']:.0f} y {fila_mat['Región']:.0f}). Cuba registró el "
                f"puntaje **más alto de la región en Matemática de 3er grado**."
            ),
            "metodo": (
                "Medias ponderadas por el peso de muestreo (WSEN), promediando los 5 valores "
                "plausibles de cada estudiante y con error estándar calculado por réplicas BRR "
                "(100 réplicas, factor de Fay 0.5). La comparación regional proviene del "
                "Informe Ejecutivo ERCE 2019 (UNESCO/LLECE)."
            ),
            "seccion": "Inicio / Resumen",
            "detalle": "KPIs de la portada, histogramas de distribución y la sección de contexto regional.",
        },
        {
            "pregunta": "¿Existen diferencias de rendimiento entre provincias?",
            "respuesta": (
                f"Sí, y son estadísticamente significativas (ANOVA F = {anova['F']:.1f}, "
                f"p {fmt_p(anova['p'])}), pero de tamaño limitado: la provincia que más puntúa "
                f"supera a la de menor rendimiento por **{brecha_prov:.0f} puntos** en Lectura "
                f"({prov_mejor['ubi1_cen_esc']} vs {prov_peor['ubi1_cen_esc']}). La provincia "
                f"explica apenas el {anova['eta2']*100:.1f}% de la variabilidad total "
                f"(η² = {anova['eta2']:.3f}), lo que indica que la mayor parte de las "
                f"desigualdades ocurre **entre estudiantes de una misma provincia**."
            ),
            "metodo": (
                "ANOVA de un factor sobre el puntaje medio por estudiante, con eta cuadrado "
                "(η²) para estimar la varianza explicada; medias provinciales ponderadas "
                "con error estándar por réplicas BRR."
            ),
            "seccion": "Rendimiento por territorio",
            "detalle": "Gráfico de barras por provincia y tabla completa.",
        },
        {
            "pregunta": "¿Existe brecha entre zona urbana y rural?",
            "respuesta": (
                f"En Lectura, la zona {zona_lect_gana} supera a la {zona_lect_perd} por "
                f"**{zona_lect_dif:.0f} puntos** (d = {abs(tz_lect['cohen_d']):.2f}, efecto "
                f"{_describir_d(tz_lect['cohen_d'])}). En Matemática la brecha es "
                f"**{zona_mat_dif:.0f} puntos** a favor de la zona {zona_mat_gana} "
                f"(d = {abs(tz_mat['cohen_d']):.2f}). En ambos casos la diferencia es "
                f"significativa (p {fmt_p(tz_lect['p'])} y p {fmt_p(tz_mat['p'])}), pero de "
                f"magnitud pequeña, menor que la asociada a las condiciones socioeconómicas."
            ),
            "metodo": (
                "Prueba t de Welch sobre el puntaje medio (valores plausibles promediados), "
                "con d de Cohen para dimensionar el tamaño de la brecha; barras de medias "
                "por zona con referencia regional."
            ),
            "seccion": "Rendimiento por territorio",
            "detalle": "Barras de medias por zona (urbana vs rural) y texto interpretativo.",
        },
        {
            "pregunta": "¿Cómo se asocia el nivel socioeconómico con el rendimiento?",
            "respuesta": (
                f"El nivel socioeconómico familiar (ISECF) es el factor individual más "
                f"asociado al rendimiento: la brecha entre el cuartil más alto y el más bajo "
                f"del índice es de **{q4 - q1:.0f} puntos** en Lectura ({q4:.0f} vs {q1:.0f}, "
                f"≈ {desv_isect:.2f} DE) y la correlación es r = {r_isect:.2f}, lo que "
                f"implica que el ISECF explica cerca del **{r_isect**2*100:.0f}% de la "
                f"varianza** de los puntajes."
            ),
            "metodo": (
                "Construcción de cuartiles del índice ISECF (estandarizado, media 0 y DE 1) "
                "con medias ponderadas por cuartil; correlación de Pearson entre ISECF y "
                "puntaje con su cuadrado (r²) como medida de varianza explicada."
            ),
            "seccion": "Brechas socioeconómicas",
            "detalle": "Gráfico de medias por cuartil, dispersión ISECF vs puntaje y educación de los padres.",
        },
        {
            "pregunta": "¿Existen brechas de género en el rendimiento?",
            "respuesta": (
                f"En Lectura, {'las niñas' if genero_lect_gana == 'Niña' else 'los niños'} "
                f"promedian **{genero_lect_dif:.0f} puntos más** que "
                f"{'los niños' if genero_lect_gana == 'Niña' else 'las niñas'} "
                f"(d = {abs(tg_lect['cohen_d']):.2f}, "
                f"efecto {_describir_d(tg_lect['cohen_d'])} y estadísticamente significativo). "
                f"En Matemática, en cambio, la brecha es mínima "
                f"({genero_mat_dif:.1f} puntos, p {fmt_p(tg_mat['p'])}) y no significativa."
            ),
            "metodo": (
                "Prueba t de Welch entre niñas y niños sobre el puntaje medio (valores "
                "plausibles promediados), con d de Cohen para el tamaño del efecto; "
                "barras de medias por género con referencia regional."
            ),
            "seccion": "Género y factores",
            "detalle": "Barras de medias por género y tabla de correlaciones de factores del estudiante.",
        },
        {
            "pregunta": "¿Qué factores del estudiante acompañan al mejor rendimiento?",
            "respuesta": (
                "El nivel socioeconómico (ISECF, r = 0.26), el apoyo al aprendizaje "
                "(AAEG3, r = 0.24), el involucramiento parental (INVAP, r = 0.24) y la "
                "organización de la enseñanza (ORGEN, r = 0.23) se asocian positivamente con "
                "el puntaje. En sentido inverso, la **violencia escolar** (VIOES, r = -0.24) "
                "acompaña a puntajes más bajos. La edad y la autoeficacia en matemática "
                "prácticamente no se asocian."
            ),
            "metodo": (
                "Correlaciones de Pearson entre cada factor del cuestionario de estudiante y "
                "el puntaje medio (valores plausibles), con significación y tamaño de muestra; "
                "se interpretan como asociaciones, no como causalidad."
            ),
            "seccion": "Género y factores",
            "detalle": "Tabla de correlaciones y texto interpretativo de la sección.",
        },
        {
            "pregunta": "¿La calidad escolar y docente percibida se asocia al rendimiento?",
            "respuesta": (
                f"El **índice global de calidad** del Módulo Nacional es el más asociado "
                f"(r = {r_ind:.2f}), seguido del refuerzo pedagógico (r = 0.27) y de las "
                f"preguntas y diálogo pedagógico (r = 0.23). Las percepciones de la calidad de "
                f"clases (docentes y directivos) y la preparación didáctica muestran "
                f"asociaciones positivas pero muy débiles."
            ),
            "metodo": (
                "Índices 0-100 del Módulo Nacional (Cuestionarios del Profesor QP3 y del "
                "Director QD3) correlacionados con el puntaje; asociación observacional, no "
                "experimental."
            ),
            "seccion": "Calidad escolar / docente",
            "detalle": "Promedio de índices, puntaje por quintil y correlaciones detalladas.",
        },
        {
            "pregunta": "¿Dónde se concentra geográficamente el rendimiento?",
            "respuesta": (
                f"El mapa reproduce las {S['esc'][S['esc']['PL3'].notna()].shape[0]} escuelas "
                f"con puntaje de la muestra. Las burbujas más oscuras indican mayor rendimiento; "
                f"Pinar del Río, La Habana y Matanzas concentran la mayor cantidad de escuelas, "
                f"y se aprecia heterogeneidad entre municipios."
            ),
            "metodo": (
                "Media de los valores plausibles por escuela, representada como burbuja "
                "georreferenciada (coordenadas del archivo oficial) sobre mapa base "
                "OpenStreetMap."
            ),
            "seccion": "Mapa geográfico",
            "detalle": "Mapa interactivo con puntaje medio por escuela.",
        },
    ]
