import streamlit as st
import pandas as pd
import plotly.express as px
from simplex_solver import resolver_simplex_propio, resolver_pulp

st.set_page_config(page_title="OptimiApp - Simplex Solver", layout="wide")

st.title("Dashboard de Optimización y Programación Lineal")
st.caption("Solucionador dinámico con comparación de algoritmos y módulo de escenarios")

# --- BARRA LATERAL: Configuración ---
st.sidebar.header("Configuración del Modelo")

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
    st.subheader("Modelo Matemático Formalizado")
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
    st.subheader("Iteraciones del Algoritmo Simplex Propio")
    for iter_info in res_propio["iteraciones"]:
        with st.expander(f"Paso {iter_info['paso']}", expanded=(iter_info['paso'] == 0)):
            if iter_info["paso"] > 0:
                st.info(f"**Variable Entrante:** {iter_info['entrante']} | **Variable Saliente:** {iter_info['saliente']}")
            st.dataframe(iter_info["tabla"])

    st.markdown("---")
    st.subheader("Comparación: Simplex Propio vs. PuLP")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Método A (Simplex Propio)")
        st.metric("Resultado Z", f"Q{res_propio['z']:,}")
        st.json(res_propio["solucion"])
    with col_b:
        st.write("### Método B (PuLP)")
        st.metric("Resultado Z", f"Q{res_pulp['z']:,}")
        st.json(res_pulp["solucion"])

# --- TAB 3: DASHBOARD DE RESULTADOS E INTERFAZ ---
with tab3:
    st.subheader("📈 Visualización e Interpretación de Resultados")
    
    # Indicadores Métricos
    c1, c2 = st.columns(2)
    c1.metric("Valor Óptimo Z", f"Q{res_propio['z']:,}")
    c2.metric("Estado de la Solución", res_propio["estado"])

    st.markdown("---")
    
    # 1. TABLA DE RECURSOS (Disponible vs. Utilizado vs. Holgura)
    st.subheader("📋 Tabla de Análisis de Recursos y Holguras")
    
    recursos_usados = []
    for i in range(num_rest):
        # Calcular el consumo real del recurso i: sum(a_ij * x_j)
        uso = sum(matriz_A[i][j] * res_propio["solucion"][f"x{j+1}"] for j in range(num_vars))
        recursos_usados.append(round(uso, 4))
    
    holguras_vals = [round(vector_b[i] - recursos_usados[i], 4) for i in range(num_rest)]
    
    df_recursos = pd.DataFrame({
        "Recurso": nombres_rest,
        "Disponible (B)": vector_b,
        "Utilizado": recursos_usados,
        "Holgura (Sobrante)": holguras_vals,
        "Estado": ["Agotado (Activo)" if h == 0 else "Sobrante" for h in holguras_vals]
    })
    
    st.dataframe(df_recursos, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualizaciones del Modelo")
    
    col_graf1, col_graf2 = st.columns(2)
    
    # Gráfico 1: Participación de Variables en la Solución
    with col_graf1:
        df_sol = pd.DataFrame({
            "Variable": nombres_vars,
            "Valor Óptimo": [res_propio["solucion"][f"x{i+1}"] for i in range(num_vars)]
        })
        fig_vars = px.bar(
            df_sol, 
            x="Variable", 
            y="Valor Óptimo", 
            color="Variable", 
            title="1. Participación de Cada Variable en la Solución"
        )
        st.plotly_chart(fig_vars, use_container_width=True)
        
    # Gráfico 2: Recursos Utilizados vs. Disponibles (SEGUNDO GRÁFICO REQUERIDO)
    with col_graf2:
        df_comparativo_rec = pd.DataFrame({
            "Recurso": nombres_rest * 2,
            "Cantidad": vector_b + recursos_usados,
            "Tipo": ["Disponible"] * num_rest + ["Utilizado"] * num_rest
        })
        fig_rec = px.bar(
            df_comparativo_rec, 
            x="Recurso", 
            y="Cantidad", 
            color="Tipo", 
            barmode="group",
            title="2. Recursos Utilizados vs. Disponibles"
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    # Resumen e Interpretación Automática
    st.markdown("---")
    st.subheader("🗣️ Interpretación Automática")
    texto = f"Para optimizar el escenario **{nombre_problema}**, la solución óptima recomienda:\n\n"
    for i in range(num_vars):
        cant = res_propio['solucion'][f'x{i+1}']
        texto += f"- **{cant} unidades** de *{nombres_vars[i]}*\n"
    texto += f"\nGenerando un resultado óptimo de **Q{res_propio['z']:,}**."
    st.info(texto)

# --- TAB 4: MÓDULO DE ESCENARIOS Y SENSIBILIDAD ---
with tab4:
    st.subheader("🧪 Análisis de Escenarios y Sensibilidad")
    st.write("Modifica los recursos disponibles para evaluar el impacto en la solución y comparar alternativas.")

    # Slider para ajustar los recursos
    porcentaje = st.slider("Variar Recursos Disponibles (%)", -50, 50, 0, step=5)

    # Recalcular vector_b con la variación porcentual
    nuevos_b = [b * (1 + porcentaje / 100.0) for b in vector_b]
    res_esc = resolver_pulp(coef_c, matriz_A, nuevos_b, es_max)

    # 1. TABLA COMPARATIVA DETALLADA (Z + Cantidades por Variable)
    st.markdown("### 📋 Tabla Comparativa de Escenarios")
    
    # Construcción de la tabla con Z y variables individuales
    datos_comparativos = {
        "Métrica / Variable": ["Valor Óptimo Z ($Q$)"] + [f"Cantidad {nombres_vars[j]}" for j in range(num_vars)],
        "Escenario Original": [f"{res_propio['z']:,}"] + [f"{res_propio['solucion'][f'x{j+1}']}" for j in range(num_vars)],
        f"Escenario Variado ({porcentaje:+d}%)": [f"{res_esc['z']:,}"] + [f"{res_esc['solucion'][f'x{j+1}']}" for j in range(num_vars)],
        "Diferencia": [f"{res_esc['z'] - res_propio['z']:+,}"] + [
            f"{res_esc['solucion'][f'x{j+1}'] - res_propio['solucion'][f'x{j+1}']:+}" for j in range(num_vars)
        ]
    }
    
    df_esc = pd.DataFrame(datos_comparativos)
    st.dataframe(df_esc, use_container_width=True)

    # 2. GRÁFICO COMPARATIVO DE UTILIDAD Z
    st.markdown("### 📊 Comparativa de Impacto en el Resultado Óptimo")
    df_graf_esc = pd.DataFrame({
        "Escenario": ["Original", f"Variación ({porcentaje:+d}%)"],
        "Valor Óptimo Z": [res_propio["z"], res_esc["z"]]
    })
    
    fig_esc = px.bar(
        df_graf_esc, 
        x="Escenario", 
        y="Valor Óptimo Z", 
        color="Escenario",
        text_auto=True,
        title="Impacto en el Valor Óptimo Z por Cambio en Recursos"
    )
    st.plotly_chart(fig_esc, use_container_width=True)

    # Interpretación del Escenario
    diferencia_z = res_esc['z'] - res_propio['z']
    if diferencia_z > 0:
        st.success(f"💡 Al aumentar los recursos en **{porcentaje}%**, la utilidad incrementa en **Q{diferencia_z:,}**.")
    elif diferencia_z < 0:
        st.warning(f"⚠️ Al reducir los recursos en **{porcentaje}%**, la utilidad disminuye en **Q{abs(diferencia_z):,}**.")
    else:
        st.info("ℹ️ El cambio en recursos no altera el valor óptimo actual.")
