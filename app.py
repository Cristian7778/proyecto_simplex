import streamlit as st
import pandas as pd
import plotly.express as px
from simplex_solver import resolver_simplex_propio, resolver_pulp

st.set_page_config(page_title="OptimiApp - Simplex Solver", layout="wide")

st.title("📊 Dashboard de Optimización y Programación Lineal")
st.caption("Solucionador dinámico con comparación de algoritmos y módulo de escenarios")

# --- BARRA LATERAL: Configuración ---
st.sidebar.header(" JOSUE MIMIR Configuración del Modelo")

nombre_problema = st.sidebar.text_input("Nombre del Escenario", "Asignación de Recursos")
tipo_opt = st.sidebar.radio("Objetivo", ["Maximizar", "Minimizar"])
es_max = tipo_opt == "Maximizar"

num_vars = st.sidebar.number_input("Número de Variables", min_value=2, max_value=8, value=3)
num_rest = st.sidebar.number_input("Número de Restricciones", min_value=1, max_value=8, value=3)

st.sidebar.subheader("Nombres de Etiquetas")
nombres_vars = [st.sidebar.text_input(f"Nombre x{i+1}", f"Variable {i+1}", key=f"v_{i}") for i in range(num_vars)]
nombres_rest = [st.sidebar.text_input(f"Nombre Recurso {i+1}", f"Recurso {i+1}", key=f"r_{i}") for i in range(num_rest)]

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Formulación", "2. Paso a Paso (Simplex vs PuLP)", "3. Resultados", "4. Escenarios"])

# TAB 1: FORMULACIÓN
with tab1:
    st.subheader("Ingreso de Parámetros del Modelo")
    
    st.markdown("#### Coeficientes de la Función Objetivo (C)")
    cols_obj = st.columns(num_vars)
    coef_c = [cols_obj[i].number_input(f"x{i+1} ({nombres_vars[i]})", value=float((i+1)*100)) for i in range(num_vars)]

    st.markdown("#### Matriz de Restricciones (A) y Disponibilidad (B)")
    matriz_A = []
    vector_b = []

    for i in range(num_rest):
        st.write(f"**{nombres_rest[i]}**")
        cols = st.columns(num_vars + 1)
        row_a = [cols[j].number_input(f"x{j+1} en R{i+1}", value=float((i+j+1)*2), key=f"a_{i}_{j}") for j in range(num_vars)]
        matriz_A.append(row_a)
        val_b = cols[-1].number_input(f"Disponible (RHS {i+1})", value=float(100 * (i+1)), key=f"b_{i}")
        vector_b.append(val_b)

    st.markdown("---")
    st.subheader("📐 Modelo Matemático Formalizado")
    eq_obj = f"\\text{{{tipo_opt}}} \\quad Z = " + " + ".join([f"{coef_c[i]}x_{{{i+1}}}" for i in range(num_vars)])
    st.latex(eq_obj)
    st.write("Sujeto a:")
    for i in range(num_rest):
        eq_rest = " + ".join([f"{matriz_A[i][j]}x_{{{j+1}}}" for j in range(num_vars)]) + f" \\leq {vector_b[i]}"
        st.latex(eq_rest)

# Ejecutar Algoritmos
res_propio = resolver_simplex_propio(coef_c, matriz_A, vector_b, es_max)
res_pulp = resolver_pulp(coef_c, matriz_A, vector_b, es_max)

# TAB 2: PASO A PASO
with tab2:
    st.subheader("🔄 Iteraciones del Algoritmo Simplex Propio")
    for iter_info in res_propio["iteraciones"]:
        with st.expander(f"Paso {iter_info['paso']}", expanded=(iter_info['paso'] == 0)):
            if iter_info["paso"] > 0:
                st.info(f"**Variable Entrante:** {iter_info['entrante']} | **Variable Saliente:** {iter_info['saliente']}")
            st.dataframe(iter_info["tabla"])

    st.markdown("---")
    st.subheader("⚖️ Comparación: Simplex Propio vs. PuLP")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Método A (Simplex Propio)")
        st.metric("Resultado Z", f"Q{res_propio['z']:,}")
        st.json(res_propio["solucion"])
    with col_b:
        st.write("### Método B (PuLP)")
        st.metric("Resultado Z", f"Q{res_pulp['z']:,}")
        st.json(res_pulp["solucion"])

# TAB 3: DASHBOARD
with tab3:
    st.subheader("📈 Visualización e Interpretación")
    c1, c2 = st.columns(2)
    c1.metric("Valor Óptimo Z", f"Q{res_propio['z']:,}")
    c2.metric("Estado de la Solución", res_propio["estado"])

    df_sol = pd.DataFrame({
        "Variable": [nombres_vars[i] for i in range(num_vars)],
        "Valor Óptimo": [res_propio["solucion"][f"x{i+1}"] for i in range(num_vars)]
    })
    fig = px.bar(df_sol, x="Variable", y="Valor Óptimo", color="Variable", title="Distribución Óptima de Variables")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗣️ Interpretación Automática")
    texto = f"Para optimizar el problema **{nombre_problema}**, la solución recomienda:\n\n"
    for i in range(num_vars):
        texto += f"- **{res_propio['solucion'][f'x{i+1}']} unidades** de *{nombres_vars[i]}*\n"
    texto += f"\nGenerando un resultado óptimo de **Q{res_propio['z']:,}**."
    st.info(texto)

# TAB 4: ESCENARIOS
with tab4:
    st.subheader("🧪In Análisis de Escenarios")
    porcentaje = st.slider("Variar Recursos Disponibles (%)", -50, 50, 0, step=10)
    
    nuevos_b = [b * (1 + porcentaje / 100.0) for b in vector_b]
    res_esc = resolver_pulp(coef_c, matriz_A, nuevos_b, es_max)
    
    df_esc = pd.DataFrame({
        "Escenario": ["Original", f"Variación ({porcentaje:+}%)"],
        "Valor Óptimo Z": [res_propio["z"], res_esc["z"]]
    })
    st.table(df_esc)
    st.plotly_chart(px.bar(df_esc, x="Escenario", y="Valor Óptimo Z", color="Escenario"), use_container_width=True)
