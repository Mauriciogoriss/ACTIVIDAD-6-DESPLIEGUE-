import streamlit as st
import plotly.express as px
import pandas as pd 

# Carga de datos
@st.cache_resource
def load_data():
    df = pd.read_csv("Usuarios.csv")
    df["Usuario"] = df["Usuario"].str.lower()
    Lista = ["mini juego", "color presionado", "dificultad", "Juego"]  
    return df, Lista

df, Lista = load_data()

st.sidebar.title("Usuarios Wuupi")
View = st.sidebar.selectbox(label="Tipo de Análisis", options=["Extracción de Características"])

if View == "Extracción de Características":
    Variable_Cat = st.sidebar.selectbox(label="Variable Categórica", options=Lista)
    
    st.title("Comparación de Frecuencias por Usuario")

    tabla_usuario = df.groupby(["Usuario", Variable_Cat]).size().reset_index(name="frecuencia")
    pivot_tabla = tabla_usuario.pivot(index="Usuario", columns=Variable_Cat, values="frecuencia").fillna(0)
    pivot_prop = pivot_tabla.div(pivot_tabla.sum(axis=1), axis=0).round(3)

    # -------- GRÁFICAS --------
    # Gráfico de Barras Agrupadas
    st.subheader("Gráfico de Barras Agrupadas")
    fig1 = px.bar(tabla_usuario, x=Variable_Cat, y="frecuencia", color="Usuario", barmode="group")
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico de Burbujas
    st.subheader("Gráfico de Burbujas")
    fig2 = px.scatter(tabla_usuario, x=Variable_Cat, y="Usuario", size="frecuencia", color="Usuario", size_max=40)
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico de Barras Apiladas
    st.subheader("Gráfico de Barras Apiladas")
    fig3 = px.bar(tabla_usuario, x="Usuario", y="frecuencia", color=Variable_Cat, barmode="stack")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Tabla
    st.subheader("Tabla de Frecuencias (Conteo)")
    st.dataframe(pivot_tabla.style.background_gradient(cmap="YlOrRd"))

    # Heatmap
    st.subheader("Heatmap de Proporciones (Por Usuario)")
    fig5 = px.imshow(pivot_prop, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
    st.plotly_chart(fig5, use_container_width=True)

    # Boxplot 
    if st.checkbox("Mostrar Boxplot"):
        cols_num = df.select_dtypes(include="number").columns.tolist()
        if cols_num:
            var_num = st.selectbox("Variable numérica", cols_num)
            fig_box = px.box(df.dropna(subset=[Variable_Cat, var_num]), 
                            x=Variable_Cat, y=var_num, color="Usuario", points="all")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No hay variables numéricas disponibles.")






