"""
Dashboard Meteorológico — Medellín y Área Metropolitana
=========================================================
Aplicación Streamlit que simula datos meteorológicos por comuna/municipio
del Valle de Aburrá, con fines de análisis de riesgo para la Alcaldía
(temperatura, humedad, viento, precipitación, calidad del aire, población).

Ejecutar con:
    streamlit run main_app.py

Clave de acceso al dashboard: 4650
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Configuración general de la página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Meteorológico - Valle de Aburrá",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLAVE_ACCESO = "4650"

# Zonas: 16 comunas de Medellín + 5 corregimientos + 9 municipios del área
# metropolitana. Se asigna un tipo de zona, población aproximada (sintética)
# y una altitud aproximada (msnm) para dar contexto físico realista.
ZONAS = {
    # Comuna: (Tipo, Población aprox., Altitud aprox. msnm)
    "Popular": ("Comuna Medellín", 130_000, 1600),
    "Santa Cruz": ("Comuna Medellín", 105_000, 1550),
    "Manrique": ("Comuna Medellín", 155_000, 1580),
    "Aranjuez": ("Comuna Medellín", 160_000, 1520),
    "Castilla": ("Comuna Medellín", 150_000, 1490),
    "Doce de Octubre": ("Comuna Medellín", 200_000, 1600),
    "Robledo": ("Comuna Medellín", 200_000, 1650),
    "Villa Hermosa": ("Comuna Medellín", 135_000, 1600),
    "Buenos Aires": ("Comuna Medellín", 135_000, 1550),
    "La Candelaria": ("Comuna Medellín", 90_000, 1495),
    "Laureles-Estadio": ("Comuna Medellín", 120_000, 1480),
    "La América": ("Comuna Medellín", 95_000, 1550),
    "San Javier": ("Comuna Medellín", 135_000, 1650),
    "El Poblado": ("Comuna Medellín", 135_000, 1550),
    "Guayabal": ("Comuna Medellín", 95_000, 1470),
    "Belén": ("Comuna Medellín", 195_000, 1560),
    "San Sebastián de Palmitas": ("Corregimiento", 6_000, 1900),
    "San Cristóbal": ("Corregimiento", 60_000, 1830),
    "Altavista": ("Corregimiento", 25_000, 1750),
    "San Antonio de Prado": ("Corregimiento", 100_000, 1900),
    "Santa Elena": ("Corregimiento", 20_000, 2500),
    "Bello": ("Municipio Metropolitano", 480_000, 1450),
    "Itagüí": ("Municipio Metropolitano", 280_000, 1550),
    "Envigado": ("Municipio Metropolitano", 240_000, 1650),
    "Sabaneta": ("Municipio Metropolitano", 55_000, 1650),
    "La Estrella": ("Municipio Metropolitano", 75_000, 1775),
    "Caldas": ("Municipio Metropolitano", 85_000, 1750),
    "Copacabana": ("Municipio Metropolitano", 75_000, 1450),
    "Girardota": ("Municipio Metropolitano", 55_000, 1425),
    "Barbosa": ("Municipio Metropolitano", 50_000, 1300),
}
LISTA_ZONAS = list(ZONAS.keys())


# --------------------------------------------------------------------------
# Generación de datos sintéticos meteorológicos
# --------------------------------------------------------------------------
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Genera un DataFrame sintético meteorológico para el Valle de Aburrá.

    Columnas (10), con tipos de datos mixtos:
        1. Fecha                 -> fecha (serie de tiempo)
        2. Zona                  -> categórica (comuna / corregimiento / municipio)
        3. Poblacion              -> numérico entero
        4. Temperatura_C          -> numérico continuo (float)
        5. Humedad_Relativa       -> numérico continuo (float, %)
        6. Velocidad_Viento_kmh   -> numérico continuo (float)
        7. Precipitacion_mm       -> numérico continuo (float)
        8. Indice_Calidad_Aire    -> numérico entero (ICA)
        9. Altitud_msnm           -> numérico entero
        10. Nivel_Riesgo          -> categórico (Bajo / Medio / Alto / Crítico)
    """
    rng = np.random.default_rng(semilla)

    fecha_ini = pd.Timestamp("2025-07-01")
    dias_rango = 365
    fechas = fecha_ini + pd.to_timedelta(rng.integers(0, dias_rango, size=n_registros), unit="D")

    zonas_sel = rng.choice(LISTA_ZONAS, size=n_registros)
    tipo_zona = np.array([ZONAS[z][0] for z in zonas_sel])
    poblacion = np.array([ZONAS[z][1] for z in zonas_sel])
    altitud_base = np.array([ZONAS[z][2] for z in zonas_sel])

    # La altitud modula temperatura (a mayor altura, menor temperatura)
    altitud = altitud_base + rng.integers(-15, 15, size=n_registros)
    temperatura = np.round(
        26 - (altitud - 1450) * 0.0065 + rng.normal(0, 1.8, size=n_registros), 1
    ).clip(8, 34)

    # Estacionalidad simple con dos temporadas de lluvia típicas del Valle de Aburrá
    dia_del_anio = fechas.dayofyear.to_numpy()
    factor_lluvia = 0.5 + 0.5 * np.sin((dia_del_anio / 365) * 4 * np.pi)
    precipitacion = np.round(
        np.clip(rng.gamma(shape=2.0, scale=8.0, size=n_registros) * (0.5 + factor_lluvia), 0, 140), 1
    )

    humedad = np.round(
        np.clip(60 + 15 * factor_lluvia + rng.normal(0, 6, size=n_registros), 35, 99), 1
    )

    viento = np.round(np.clip(rng.gamma(shape=2.2, scale=4.5, size=n_registros), 1, 55), 1)

    ica = np.round(
        np.clip(
            55 + (poblacion / poblacion.max()) * 60 + rng.normal(0, 15, size=n_registros) - factor_lluvia * 20,
            10, 300,
        )
    ).astype(int)

    # Nivel de riesgo derivado de la combinación de variables (útil para la
    # Alcaldía: identifica posibles riesgos de inundación, vendaval o
    # contaminación del aire)
    riesgo_score = (
        (precipitacion > 80).astype(int) * 2
        + (viento > 35).astype(int) * 2
        + (ica > 150).astype(int) * 2
        + (humedad > 90).astype(int)
        + rng.integers(0, 2, size=n_registros)
    )
    nivel_riesgo = pd.cut(
        riesgo_score,
        bins=[-1, 0, 2, 4, 100],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
    ).astype(str)

    df = pd.DataFrame(
        {
            "Fecha": fechas,
            "Zona": zonas_sel,
            "Tipo_Zona": tipo_zona,
            "Poblacion": poblacion.astype(int),
            "Temperatura_C": temperatura,
            "Humedad_Relativa": humedad,
            "Velocidad_Viento_kmh": viento,
            "Precipitacion_mm": precipitacion,
            "Indice_Calidad_Aire": ica,
            "Altitud_msnm": altitud.astype(int),
            "Nivel_Riesgo": nivel_riesgo,
        }
    )
    # Se mantienen 10 columnas "núcleo" solicitadas; Tipo_Zona se deja como
    # columna auxiliar de apoyo al análisis cualitativo.
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def columnas_numericas(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def columnas_categoricas(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ==========================================================================
# CONTROL DE ACCESO — clave 4650
# ==========================================================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

with st.sidebar:
    st.markdown("## 🎓 EAFIT 2026")
    st.markdown(
        """
        **Ciencia de Datos**
        Profesor: **Jorge Padilla**
        Julio de 2026
        """
    )
    st.divider()

if not st.session_state.autenticado:
    st.title("🌦️ Dashboard Meteorológico — Valle de Aburrá")
    st.info("Este panel requiere una clave de acceso para operar.")
    with st.form("form_acceso"):
        clave_ingresada = st.text_input("🔑 Ingrese la clave de acceso", type="password")
        enviado = st.form_submit_button("Ingresar")
    if enviado:
        if clave_ingresada == CLAVE_ACCESO:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Clave incorrecta. Intente nuevamente.")
    st.stop()

# --------------------------------------------------------------------------
# A partir de aquí el usuario ya está autenticado
# --------------------------------------------------------------------------
COLUMNAS_ESPERADAS = {
    "Fecha", "Zona", "Tipo_Zona", "Poblacion", "Temperatura_C",
    "Humedad_Relativa", "Velocidad_Viento_kmh", "Precipitacion_mm",
    "Indice_Calidad_Aire", "Altitud_msnm", "Nivel_Riesgo",
}

# Si el estado de sesión trae un DataFrame de una versión anterior de la app
# (p. ej. el dataset de COVID previo, con otras columnas), se descarta y se
# regenera con el esquema meteorológico actual para evitar KeyError.
necesita_regenerar = (
    "df" not in st.session_state
    or not COLUMNAS_ESPERADAS.issubset(set(st.session_state.df.columns))
)
if necesita_regenerar:
    st.session_state.df = generar_datos(500, semilla=42)
if "semilla" not in st.session_state:
    st.session_state.semilla = 42


with st.sidebar:
    st.subheader("⚙️ Simulación de datos")
    n_registros = st.slider("Número de registros", min_value=100, max_value=2000, value=500, step=50)
    semilla = st.number_input(
        "Semilla aleatoria (reproducibilidad)", min_value=0, max_value=9999, value=st.session_state.semilla
    )
    if st.button("🔄 Generar / Regenerar datos", use_container_width=True):
        st.session_state.df = generar_datos(n_registros, semilla)
        st.session_state.semilla = semilla

    st.divider()
    if st.button("🔒 Cerrar sesión", use_container_width=True):
        st.session_state.autenticado = False
        st.rerun()

df = st.session_state.df
num_cols = [c for c in columnas_numericas(df) if c != "Poblacion" or True]
cat_cols = columnas_categoricas(df)

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("🌦️ Dashboard Meteorológico — Medellín y Área Metropolitana")
st.markdown(
    "Datos **sintéticos** por comuna, corregimiento y municipio del Valle de Aburrá: "
    "temperatura, humedad, viento, precipitación, calidad del aire, población y altitud. "
    "Pensado como insumo de apoyo a decisiones de la Alcaldía sobre **riesgos climáticos** "
    "(inundaciones, vendavales, contaminación del aire)."
)
st.caption(f"Dataset actual: **{df.shape[0]} registros** × **{df.shape[1]} columnas**")

with st.expander("📄 Ver tabla de datos completa", expanded=False):
    st.dataframe(df, use_container_width=True)
    st.caption(
        "Tipos de dato → Fecha: fecha | Zona / Tipo_Zona / Nivel_Riesgo: categórico | "
        "Poblacion / Indice_Calidad_Aire / Altitud_msnm: entero | "
        "Temperatura_C / Humedad_Relativa / Velocidad_Viento_kmh / Precipitacion_mm: flotante"
    )

st.divider()

# --------------------------------------------------------------------------
# Filtros globales
# --------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🔍 Filtros globales")
zonas_filtro = st.sidebar.multiselect("Zona(s)", LISTA_ZONAS, default=[])
rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["Fecha"].min().date(), df["Fecha"].max().date()),
    min_value=df["Fecha"].min().date(),
    max_value=df["Fecha"].max().date(),
)

df_f = df.copy()
if zonas_filtro:
    df_f = df_f[df_f["Zona"].isin(zonas_filtro)]
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    ini, fin = rango_fechas
    df_f = df_f[(df_f["Fecha"].dt.date >= ini) & (df_f["Fecha"].dt.date <= fin)]

if df_f.empty:
    st.warning("No hay registros con los filtros seleccionados. Ajuste los filtros en la barra lateral.")
    st.stop()

num_cols_f = columnas_numericas(df_f)
cat_cols_f = columnas_categoricas(df_f)

# --------------------------------------------------------------------------
# Tabs principales
# --------------------------------------------------------------------------
tab_cuanti, tab_cuali, tab_serie, tab_viz, tab_riesgo, tab_corr = st.tabs(
    [
        "📊 Cuantitativa", "🗂️ Cualitativa", "⏱️ Serie de Tiempo",
        "📈 Visualización Dinámica", "🚨 Riesgo y Alertas", "🔗 Correlación",
    ]
)

# ==========================================================================
# TAB 1 — Estadística cuantitativa
# ==========================================================================
with tab_cuanti:
    st.subheader("Resumen de variables numéricas")

    kpi_vars = ["Temperatura_C", "Humedad_Relativa", "Velocidad_Viento_kmh", "Precipitacion_mm"]
    kpi_cols = st.columns(len(kpi_vars))
    for c, col in zip(kpi_cols, kpi_vars):
        c.metric(label=f"Promedio {col}", value=f"{df_f[col].mean():.2f}", delta=f"σ = {df_f[col].std():.2f}")

    st.markdown("##### Tabla estadística descriptiva")
    resumen = df_f[num_cols_f].describe().T
    resumen["mediana"] = df_f[num_cols_f].median()
    resumen["rango"] = resumen["max"] - resumen["min"]
    resumen = resumen.rename(
        columns={
            "count": "conteo", "mean": "media", "std": "desv_std",
            "min": "mínimo", "max": "máximo",
            "25%": "Q1", "50%": "Q2 (mediana)", "75%": "Q3",
        }
    )
    st.dataframe(resumen.style.format("{:.2f}"), use_container_width=True)

    st.markdown("##### Distribución individual")
    col_sel = st.selectbox("Selecciona una variable numérica", num_cols_f, key="hist_cuanti")
    fig_box_hist = px.histogram(
        df_f, x=col_sel, marginal="box", nbins=20,
        color_discrete_sequence=["#2E86AB"], title=f"Distribución de {col_sel}",
    )
    st.plotly_chart(fig_box_hist, use_container_width=True)

# ==========================================================================
# TAB 2 — Estadística cualitativa
# ==========================================================================
with tab_cuali:
    st.subheader("Resumen de variables categóricas")

    col_cat = st.selectbox("Selecciona una variable categórica", cat_cols_f, key="cat_sel")
    moda = df_f[col_cat].mode().iloc[0]
    st.metric(label=f"Moda de {col_cat}", value=str(moda))

    freq = df_f[col_cat].value_counts().reset_index()
    freq.columns = [col_cat, "Frecuencia"]
    freq["Porcentaje (%)"] = np.round(100 * freq["Frecuencia"] / freq["Frecuencia"].sum(), 1)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("##### Tabla de frecuencias")
        st.dataframe(freq, use_container_width=True, hide_index=True)
    with c2:
        fig_pie = px.pie(freq, names=col_cat, values="Frecuencia", title=f"Distribución de {col_cat}", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("##### Población por zona")
    pobl_zona = df_f.drop_duplicates("Zona")[["Zona", "Tipo_Zona", "Poblacion"]].sort_values(
        "Poblacion", ascending=False
    )
    fig_pobl = px.bar(
        pobl_zona, x="Zona", y="Poblacion", color="Tipo_Zona",
        title="Población por zona (según registros filtrados)",
    )
    fig_pobl.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pobl, use_container_width=True)

# ==========================================================================
# TAB 3 — Serie de tiempo
# ==========================================================================
with tab_serie:
    st.subheader("Comportamiento en el tiempo")

    c1, c2, c3 = st.columns(3)
    with c1:
        var_tiempo = st.selectbox("Variable a graficar", num_cols_f, key="serie_var")
    with c2:
        agrupacion = st.selectbox("Agregación temporal", ["Diaria", "Semanal", "Mensual"], index=2)
    with c3:
        zonas_serie = st.multiselect(
            "Zonas a comparar (vacío = todas agregadas)", LISTA_ZONAS, default=[], key="serie_zonas"
        )

    df_serie = df_f.copy()
    if zonas_serie:
        df_serie = df_serie[df_serie["Zona"].isin(zonas_serie)]

    regla = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}[agrupacion]

    if zonas_serie:
        serie_agg = (
            df_serie.set_index("Fecha").groupby("Zona")[var_tiempo]
            .resample(regla).mean().reset_index()
        )
        fig_serie = px.line(
            serie_agg, x="Fecha", y=var_tiempo, color="Zona", markers=True,
            title=f"{var_tiempo} en el tiempo ({agrupacion.lower()}) por zona",
        )
    else:
        serie_agg = df_serie.set_index("Fecha")[var_tiempo].resample(regla).mean().reset_index()
        fig_serie = px.line(
            serie_agg, x="Fecha", y=var_tiempo, markers=True,
            title=f"{var_tiempo} en el tiempo ({agrupacion.lower()}) — todas las zonas",
        )

    usar_umbral_serie = st.checkbox("Mostrar línea de umbral de alerta", value=False, key="umbral_serie")
    if usar_umbral_serie:
        v_min, v_max = float(df_f[var_tiempo].min()), float(df_f[var_tiempo].max())
        umbral_serie = st.slider(
            f"Umbral de alerta para {var_tiempo}",
            min_value=float(np.floor(v_min)), max_value=float(np.ceil(v_max)),
            value=float(np.round((v_min + v_max) / 2, 1)),
        )
        fig_serie.add_hline(
            y=umbral_serie, line_dash="dash", line_color="red",
            annotation_text=f"Umbral: {umbral_serie}", annotation_position="top left",
        )

    fig_serie.update_layout(template="plotly_white")
    st.plotly_chart(fig_serie, use_container_width=True)

# ==========================================================================
# TAB 4 — Visualización dinámica
# ==========================================================================
with tab_viz:
    st.subheader("Construye tu propia gráfica")

    c1, c2, c3 = st.columns(3)
    with c1:
        tipo_grafica = st.selectbox(
            "Tipo de gráfica", ["Dispersión (scatter)", "Barras", "Línea", "Caja (box)", "Histograma"]
        )
    with c2:
        eje_x = st.selectbox("Variable eje X", df_f.columns.tolist(), index=0)
    with c3:
        eje_y_opciones = ["(ninguna)"] + df_f.columns.tolist()
        eje_y = st.selectbox(
            "Variable eje Y", eje_y_opciones,
            index=(eje_y_opciones.index("Temperatura_C") if "Temperatura_C" in eje_y_opciones else 0),
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        color_por = st.selectbox("Colorear por", ["(ninguna)"] + cat_cols_f, index=0)
    with c5:
        paleta = st.selectbox("Paleta de color", ["Plotly", "Viridis", "Bluered", "Sunset", "Tealgrn", "Inferno"])
    with c6:
        color_base = st.color_picker("Color base (si no hay agrupación)", "#2E86AB")

    st.markdown("##### Umbral de referencia")
    c7, c8 = st.columns([2, 1])
    with c7:
        usar_umbral = st.checkbox("Mostrar línea/franja de umbral", value=True)

    umbral_valor = None
    if usar_umbral and eje_y != "(ninguna)" and eje_y in num_cols_f:
        y_min, y_max = float(df_f[eje_y].min()), float(df_f[eje_y].max())
        umbral_valor = st.slider(
            f"Valor de umbral para '{eje_y}'",
            min_value=float(np.floor(y_min - 5)), max_value=float(np.ceil(y_max + 5)),
            value=float(np.round((y_min + y_max) / 2, 1)),
        )
    elif usar_umbral and eje_x in num_cols_f and eje_y == "(ninguna)":
        x_min, x_max = float(df_f[eje_x].min()), float(df_f[eje_x].max())
        umbral_valor = st.slider(
            f"Valor de umbral para '{eje_x}'",
            min_value=float(np.floor(x_min - 5)), max_value=float(np.ceil(x_max + 5)),
            value=float(np.round((x_min + x_max) / 2, 1)),
        )

    titulo_custom = st.text_input("Título personalizado de la gráfica", "")
    df_plot = df_f.copy()
    titulo_final = titulo_custom if titulo_custom else f"{tipo_grafica}: {eje_x}" + (
        f" vs {eje_y}" if eje_y != "(ninguna)" else ""
    )

    try:
        if tipo_grafica == "Dispersión (scatter)":
            fig = px.scatter(
                df_plot, x=eje_x, y=(eje_y if eje_y != "(ninguna)" else None), title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
                hover_data=["Zona"] if "Zona" in df_plot.columns else None,
                size="Poblacion" if "Poblacion" in df_plot.columns else None,
            )
        elif tipo_grafica == "Barras":
            if eje_y != "(ninguna)" and eje_y in num_cols_f and eje_x in cat_cols_f:
                agg = df_plot.groupby(eje_x, as_index=False)[eje_y].mean()
                fig = px.bar(
                    agg, x=eje_x, y=eje_y, title=titulo_final,
                    color=eje_x if color_por == "(ninguna)" else None,
                    color_discrete_sequence=[color_base] if color_por == "(ninguna)" else None,
                )
            else:
                conteo = df_plot[eje_x].value_counts().reset_index()
                conteo.columns = [eje_x, "conteo"]
                fig = px.bar(conteo, x=eje_x, y="conteo", title=titulo_final, color_discrete_sequence=[color_base])
        elif tipo_grafica == "Línea":
            df_line = df_plot.sort_values(eje_x)
            fig = px.line(
                df_line, x=eje_x, y=(eje_y if eje_y != "(ninguna)" else num_cols_f[0]),
                markers=True, title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )
        elif tipo_grafica == "Caja (box)":
            fig = px.box(
                df_plot, x=eje_x, y=(eje_y if eje_y != "(ninguna)" else None), title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )
        else:  # Histograma
            fig = px.histogram(
                df_plot, x=eje_x, title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )

        if usar_umbral and umbral_valor is not None:
            if tipo_grafica == "Histograma":
                fig.add_vline(
                    x=umbral_valor, line_dash="dash", line_color="red",
                    annotation_text=f"Umbral: {umbral_valor}", annotation_position="top",
                )
            else:
                fig.add_hline(
                    y=umbral_valor, line_dash="dash", line_color="red",
                    annotation_text=f"Umbral: {umbral_valor}", annotation_position="top right",
                )

        fig.update_layout(template="plotly_white", legend_title_text=color_por)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"No fue posible construir la gráfica con esta combinación de variables: {e}")

# ==========================================================================
# TAB 5 — Riesgo y alertas (apoyo a decisión de la Alcaldía)
# ==========================================================================
with tab_riesgo:
    st.subheader("Panel de riesgo por zona")

    dist_riesgo = df_f["Nivel_Riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto", "Crítico"]).fillna(0)
    kpi_r = st.columns(4)
    colores_riesgo = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🟠", "Crítico": "🔴"}
    for c, nivel in zip(kpi_r, ["Bajo", "Medio", "Alto", "Crítico"]):
        c.metric(f"{colores_riesgo[nivel]} {nivel}", int(dist_riesgo[nivel]))

    st.markdown("##### Zonas con mayor proporción de riesgo Alto/Crítico")
    riesgo_zona = (
        df_f.assign(alto_critico=df_f["Nivel_Riesgo"].isin(["Alto", "Crítico"]))
        .groupby("Zona")["alto_critico"].mean().mul(100).round(1)
        .sort_values(ascending=False).reset_index()
    )
    riesgo_zona.columns = ["Zona", "% registros Alto/Crítico"]
    fig_riesgo_zona = px.bar(
        riesgo_zona.head(15), x="Zona", y="% registros Alto/Crítico",
        color="% registros Alto/Crítico", color_continuous_scale="Reds",
        title="Top 15 zonas por proporción de riesgo Alto/Crítico",
    )
    fig_riesgo_zona.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_riesgo_zona, use_container_width=True)

    st.markdown("##### Explorador de alertas por variable")
    c1, c2 = st.columns(2)
    with c1:
        var_alerta = st.selectbox(
            "Variable a monitorear",
            ["Precipitacion_mm", "Velocidad_Viento_kmh", "Indice_Calidad_Aire", "Humedad_Relativa", "Temperatura_C"],
        )
    with c2:
        v_min, v_max = float(df_f[var_alerta].min()), float(df_f[var_alerta].max())
        umbral_alerta = st.slider(
            f"Umbral de alerta para {var_alerta}",
            min_value=float(np.floor(v_min)), max_value=float(np.ceil(v_max)),
            value=float(np.round(np.percentile(df_f[var_alerta], 85), 1)),
        )

    df_alerta = df_f[df_f[var_alerta] >= umbral_alerta].sort_values(var_alerta, ascending=False)
    st.warning(
        f"⚠️ {len(df_alerta)} registros superan el umbral de {umbral_alerta} en '{var_alerta}' "
        f"({100 * len(df_alerta) / len(df_f):.1f}% del total filtrado)."
    )
    st.dataframe(
        df_alerta[["Fecha", "Zona", "Tipo_Zona", var_alerta, "Nivel_Riesgo"]].reset_index(drop=True),
        use_container_width=True,
    )

# ==========================================================================
# TAB 6 — Correlación
# ==========================================================================
with tab_corr:
    st.subheader("Matriz de correlación (variables numéricas)")
    if len(num_cols_f) >= 2:
        corr = df_f[num_cols_f].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto", title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Se requieren al menos dos variables numéricas para calcular correlaciones.")

st.divider()
st.caption(
    "Dashboard construido con Streamlit + Plotly · Datos 100% sintéticos, generados con fines "
    "académicos (EAFIT — Ciencia de Datos) y no representan mediciones reales del Valle de Aburrá."
)
