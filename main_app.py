
"""
Dashboard COVID - Datos Sintéticos
===================================
Aplicación Streamlit que genera datos sintéticos relacionados con COVID-19,
calcula un esquema de métricas (cuantitativas y cualitativas) y permite
explorar los datos con gráficas dinámicas construidas con Plotly.

Ejecutar con:
    streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Configuración general de la página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID - Datos Sintéticos",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CIUDADES = ["Medellín", "Bogotá", "Cali", "Barranquilla", "Bucaramanga"]
GENEROS = ["Femenino", "Masculino"]
ESTADOS = ["Activo", "Recuperado", "Fallecido"]
COMORBILIDADES = ["Ninguna", "Diabetes", "Hipertensión", "Obesidad", "Asma"]


# --------------------------------------------------------------------------
# Generación de datos sintéticos
# --------------------------------------------------------------------------
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Genera un DataFrame sintético relacionado con COVID-19.

    Columnas (8), con tipos de datos mixtos:
        1. Paciente_ID          -> texto/categórico (identificador)
        2. Edad                 -> numérico entero
        3. Genero                -> categórico
        4. Ciudad                -> categórico
        5. Fecha_Diagnostico      -> fecha
        6. Dias_Sintomas          -> numérico entero
        7. Saturacion_O2          -> numérico continuo (float)
        8. Estado                -> categórico (variable objetivo)
    """
    rng = np.random.default_rng(semilla)

    fecha_base = pd.Timestamp("2023-01-01")
    fechas = fecha_base + pd.to_timedelta(
        rng.integers(0, 365, size=n_registros), unit="D"
    )

    df = pd.DataFrame(
        {
            "Paciente_ID": [f"PAC-{i:04d}" for i in rng.choice(
                np.arange(1, 9999), size=n_registros, replace=False
            )],
            "Edad": rng.integers(1, 95, size=n_registros),
            "Genero": rng.choice(GENEROS, size=n_registros),
            "Ciudad": rng.choice(CIUDADES, size=n_registros),
            "Fecha_Diagnostico": fechas,
            "Dias_Sintomas": rng.integers(0, 21, size=n_registros),
            "Saturacion_O2": np.round(rng.normal(94, 4, size=n_registros).clip(70, 100), 1),
            "Estado": rng.choice(ESTADOS, size=n_registros, p=[0.45, 0.48, 0.07]),
        }
    )
    df = df.sort_values("Fecha_Diagnostico").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Utilidades de métricas
# --------------------------------------------------------------------------
def columnas_numericas(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def columnas_categoricas(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def columnas_fecha(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()


# --------------------------------------------------------------------------
# Estado de sesión: los datos se generan una vez y persisten mientras
# el usuario interactúa con el resto del dashboard.
# --------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = generar_datos(10, semilla=42)
if "semilla" not in st.session_state:
    st.session_state.semilla = 42

# --------------------------------------------------------------------------
# Sidebar — controles de simulación de datos
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Panel de control")
st.sidebar.subheader("1. Simulación de datos")

n_registros = st.sidebar.slider(
    "Número de registros a simular", min_value=5, max_value=200, value=10, step=1
)
semilla = st.sidebar.number_input(
    "Semilla aleatoria (reproducibilidad)", min_value=0, max_value=9999, value=st.session_state.semilla
)

if st.sidebar.button("🔄 Generar / Regenerar datos", use_container_width=True):
    st.session_state.df = generar_datos(n_registros, semilla)
    st.session_state.semilla = semilla

df = st.session_state.df

st.sidebar.caption(
    f"Dataset actual: **{df.shape[0]} registros** × **{df.shape[1]} columnas**"
)

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("🦠 Dashboard COVID — Datos Sintéticos")
st.markdown(
    "Explora un conjunto de datos **simulado** de casos COVID-19 con variables "
    "demográficas, clínicas y temporales. Usa el panel lateral para regenerar "
    "los datos y las pestañas para navegar entre estadística cuantitativa, "
    "cualitativa y visualización dinámica."
)

with st.expander("📄 Ver tabla de datos completa", expanded=False):
    st.dataframe(df, use_container_width=True)
    st.caption(
        "Tipos de dato → Paciente_ID: texto | Edad / Dias_Sintomas: entero | "
        "Saturacion_O2: flotante | Genero / Ciudad / Estado: categórico | "
        "Fecha_Diagnostico: fecha"
    )

st.divider()

# --------------------------------------------------------------------------
# Tabs principales
# --------------------------------------------------------------------------
tab_cuanti, tab_cuali, tab_viz, tab_corr = st.tabs(
    ["📊 Estadística Cuantitativa", "🗂️ Estadística Cualitativa", "📈 Visualización Dinámica", "🔗 Correlación"]
)

num_cols = columnas_numericas(df)
cat_cols = columnas_categoricas(df)
date_cols = columnas_fecha(df)

# ==========================================================================
# TAB 1 — Estadística cuantitativa
# ==========================================================================
with tab_cuanti:
    st.subheader("Resumen de variables numéricas")

    kpi_cols = st.columns(len(num_cols))
    for c, col in zip(kpi_cols, num_cols):
        c.metric(
            label=f"Promedio {col}",
            value=f"{df[col].mean():.2f}",
            delta=f"σ = {df[col].std():.2f}",
        )

    st.markdown("##### Tabla estadística descriptiva")
    resumen = df[num_cols].describe().T
    resumen["mediana"] = df[num_cols].median()
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
    col_sel = st.selectbox("Selecciona una variable numérica", num_cols, key="hist_cuanti")
    fig_box_hist = px.histogram(
        df, x=col_sel, marginal="box", nbins=15,
        color_discrete_sequence=["#2E86AB"],
        title=f"Distribución de {col_sel}",
    )
    st.plotly_chart(fig_box_hist, use_container_width=True)

# ==========================================================================
# TAB 2 — Estadística cualitativa
# ==========================================================================
with tab_cuali:
    st.subheader("Resumen de variables categóricas")

    kpi_cols = st.columns(len(cat_cols))
    for c, col in zip(kpi_cols, cat_cols):
        moda = df[col].mode().iloc[0]
        c.metric(label=f"Moda de {col}", value=str(moda))

    col_cat = st.selectbox("Selecciona una variable categórica", cat_cols, key="cat_sel")

    freq = df[col_cat].value_counts().reset_index()
    freq.columns = [col_cat, "Frecuencia"]
    freq["Porcentaje (%)"] = np.round(100 * freq["Frecuencia"] / freq["Frecuencia"].sum(), 1)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("##### Tabla de frecuencias")
        st.dataframe(freq, use_container_width=True, hide_index=True)
    with c2:
        fig_pie = px.pie(
            freq, names=col_cat, values="Frecuencia",
            title=f"Distribución de {col_cat}", hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================================================
# TAB 3 — Visualización dinámica (interactiva, elegida por el usuario)
# ==========================================================================
with tab_viz:
    st.subheader("Construye tu propia gráfica")

    c1, c2, c3 = st.columns(3)
    with c1:
        tipo_grafica = st.selectbox(
            "Tipo de gráfica",
            ["Dispersión (scatter)", "Barras", "Línea", "Caja (box)", "Histograma"],
        )
    with c2:
        eje_x = st.selectbox("Variable eje X", df.columns.tolist(), index=0)
    with c3:
        eje_y_opciones = ["(ninguna)"] + df.columns.tolist()
        eje_y = st.selectbox("Variable eje Y", eje_y_opciones,
                              index=(eje_y_opciones.index("Saturacion_O2")
                                     if "Saturacion_O2" in eje_y_opciones else 0))

    c4, c5, c6 = st.columns(3)
    with c4:
        color_por = st.selectbox("Colorear por", ["(ninguna)"] + cat_cols, index=0)
    with c5:
        paleta = st.selectbox(
            "Paleta de color",
            ["Plotly", "Viridis", "Bluered", "Sunset", "Tealgrn", "Inferno"],
        )
    with c6:
        color_base = st.color_picker("Color base (si no hay agrupación)", "#2E86AB")

    st.markdown("##### Umbral de referencia")
    c7, c8 = st.columns([2, 1])
    with c7:
        usar_umbral = st.checkbox("Mostrar línea/franja de umbral", value=True)
    with c8:
        umbral_valor = None

    if usar_umbral and eje_y != "(ninguna)" and eje_y in num_cols:
        y_min, y_max = float(df[eje_y].min()), float(df[eje_y].max())
        umbral_valor = st.slider(
            f"Valor de umbral para '{eje_y}'",
            min_value=float(np.floor(y_min - 5)),
            max_value=float(np.ceil(y_max + 5)),
            value=float(np.round((y_min + y_max) / 2, 1)),
        )
    elif usar_umbral and eje_x in num_cols and eje_y == "(ninguna)":
        x_min, x_max = float(df[eje_x].min()), float(df[eje_x].max())
        umbral_valor = st.slider(
            f"Valor de umbral para '{eje_x}'",
            min_value=float(np.floor(x_min - 5)),
            max_value=float(np.ceil(x_max + 5)),
            value=float(np.round((x_min + x_max) / 2, 1)),
        )

    titulo_custom = st.text_input("Título personalizado de la gráfica", "")

    color_kwargs = {}
    if color_por != "(ninguna)":
        color_kwargs["color"] = color_por
    else:
        color_kwargs["color_discrete_sequence"] = [color_base]

    color_kwargs["color_continuous_scale"] = paleta.lower() if color_por in num_cols else None

    df_plot = df.copy()
    titulo_final = titulo_custom if titulo_custom else f"{tipo_grafica}: {eje_x}" + (
        f" vs {eje_y}" if eje_y != "(ninguna)" else ""
    )

    try:
        if tipo_grafica == "Dispersión (scatter)":
            fig = px.scatter(
                df_plot, x=eje_x,
                y=(eje_y if eje_y != "(ninguna)" else None),
                title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
                hover_data=["Paciente_ID"] if "Paciente_ID" in df_plot.columns else None,
                size="Dias_Sintomas" if "Dias_Sintomas" in df_plot.columns else None,
            )
        elif tipo_grafica == "Barras":
            if eje_y != "(ninguna)" and eje_y in num_cols and eje_x in cat_cols + date_cols:
                agg = df_plot.groupby(eje_x, as_index=False)[eje_y].mean()
                fig = px.bar(
                    agg, x=eje_x, y=eje_y, title=titulo_final,
                    color=eje_x if color_por == "(ninguna)" else None,
                    color_discrete_sequence=[color_base] if color_por == "(ninguna)" else None,
                )
            else:
                conteo = df_plot[eje_x].value_counts().reset_index()
                conteo.columns = [eje_x, "conteo"]
                fig = px.bar(
                    conteo, x=eje_x, y="conteo", title=titulo_final,
                    color_discrete_sequence=[color_base],
                )
        elif tipo_grafica == "Línea":
            df_line = df_plot.sort_values(eje_x)
            fig = px.line(
                df_line, x=eje_x,
                y=(eje_y if eje_y != "(ninguna)" else num_cols[0]),
                markers=True, title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )
        elif tipo_grafica == "Caja (box)":
            fig = px.box(
                df_plot, x=eje_x,
                y=(eje_y if eje_y != "(ninguna)" else None),
                title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )
        else:  # Histograma
            fig = px.histogram(
                df_plot, x=eje_x, title=titulo_final,
                color=color_por if color_por != "(ninguna)" else None,
                color_discrete_sequence=None if color_por != "(ninguna)" else [color_base],
            )

        # Línea/franja de umbral
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
# TAB 4 — Correlación entre variables numéricas
# ==========================================================================
with tab_corr:
    st.subheader("Matriz de correlación (variables numéricas)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto", title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Se requieren al menos dos variables numéricas para calcular correlaciones.")

st.divider()
st.caption(
    "Dashboard construido con Streamlit + Plotly · Datos 100% sintéticos, "
    "generados con fines demostrativos y no representan casos reales."
)
