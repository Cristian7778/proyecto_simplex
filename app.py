import numpy as np
import pandas as pd
import pulp

def resolver_simplex_propio(c, A, b, signos, es_max=True):
    """
    Resuelve un problema de Programación Lineal utilizando el Algoritmo Simplex Primal
    con soporte para el Método de la Gran M (para operadores <=, >=, =).
    
    Parámetros:
    - c: Coeficientes de la función objetivo.
    - A: Matriz de coeficientes tecnológicos de las restricciones.
    - b: Vector de disponibilidades (Lado Derecho / RHS).
    - signos: Lista con los operadores relacionales de cada restricción ("<=", ">=", "=").
    - es_max: Booleano que indica si se desea Maximizar (True) o Minimizar (False).
    """
    num_vars = len(c)
    num_rest = len(b)
    
    # Si el objetivo es minimizar, se invierten los signos de c para transformar a maximización
    c_mod = np.array(c, dtype=float) if es_max else -np.array(c, dtype=float)
    A_mod = np.array(A, dtype=float)
    b_mod = np.array(b, dtype=float)
    
    # --- PASO 1: Normalización de Lados Derechos Negativos (b_i < 0) ---
    for i in range(num_rest):
        if b_mod[i] < 0:
            b_mod[i] *= -1
            A_mod[i] *= -1
            # Inversión del operador al multiplicar por -1
            if signos[i] == "<=":
                signos[i] = ">="
            elif signos[i] == ">=":
                signos[i] = "<="

    # --- PASO 2: Clasificación de Variables Especiales (Holgura, Exceso y Artificiales) ---
    var_names = [f"x{j+1}" for j in range(num_vars)]  # Variables de decisión x1, x2...
    slack_names = []   # Variables de Holgura (s)
    excess_names = []  # Variables de Exceso (e)
    artif_names = []   # Variables Artificiales (a)
    
    row_slack_map = {}
    row_artif_map = {}
    
    s_count, e_count, a_count = 1, 1, 1
    for i, sg in enumerate(signos):
        if sg == "<=":
            name = f"s{s_count}"
            slack_names.append(name)
            row_slack_map[i] = (name, 1.0)
            s_count += 1
        elif sg == ">=":
            e_name = f"e{e_count}"
            a_name = f"a{a_count}"
            excess_names.append(e_name)
            artif_names.append(a_name)
            row_slack_map[i] = (e_name, -1.0)
            row_artif_map[i] = a_name
            e_count += 1
            a_count += 1
        elif sg == "=":
            a_name = f"a{a_count}"
            artif_names.append(a_name)
            row_artif_map[i] = a_name
            a_count += 1

    # Estructuración de los nombres de todas las columnas del Tableau
    all_cols = var_names + slack_names + excess_names + artif_names + ["LD", "Razón"]
    
    # Matriz inicial llena de ceros: (Restricciones + 1 Fila Z) x Columnas Totales
    tableau = np.zeros((num_rest + 1, len(all_cols)))
    var_basicas = []
    
    # Asignación de coeficientes A y disponibilidades LD en el Tableau
    tableau[:num_rest, :num_vars] = A_mod
    tableau[:num_rest, -2] = b_mod
    
    # Colocación de coeficientes de variables especiales e identificación de la Base Inicial
    for i in range(num_rest):
        if i in row_slack_map:
            col_name, val = row_slack_map[i]
            col_idx = all_cols.index(col_name)
            tableau[i, col_idx] = val
            if val == 1.0:
                var_basicas.append(col_name)  # Las holguras positivas entran a la base
                
        if i in row_artif_map:
            artif_name = row_artif_map[i]
            col_idx = all_cols.index(artif_name)
            tableau[i, col_idx] = 1.0
            var_basicas.append(artif_name)  # Las artificiales entran a la base

    # --- PASO 3: Método de la Gran M (Penalización en la Fila Z) ---
    M = 1e6  # Valor simbólico para representar un número infinitamente grande
    tableau[-1, :num_vars] = -c_mod
    
    # Asignar la penalización M en las columnas artificiales
    for artif in artif_names:
        col_idx = all_cols.index(artif)
        tableau[-1, col_idx] = M
        
    # Convertir la fila Z a forma canónica (dejar ceros en las columnas básicas artificiales)
    for i in range(num_rest):
        if i in row_artif_map:
            tableau[-1, :-1] -= M * tableau[i, :-1]

    # --- PASO 4: Bucle Iterativo Simplex (Gauss-Jordan) ---
    iteraciones = []
    paso = 0

    while True:
        fila_z = tableau[-1, :-2]
        
        # Criterio de Variable Entrante: Selecciona el coeficiente más negativo en la fila Z
        col_pivote_temp = np.argmin(fila_z) if np.min(fila_z) < -1e-7 else -1
        
        # Cálculo de Razones (Lado Derecho / Elemento Pivote)
        for i in range(num_rest):
            if col_pivote_temp != -1 and tableau[i, col_pivote_temp] > 1e-7:
                tableau[i, -1] = tableau[i, -2] / tableau[i, col_pivote_temp]
            else:
                tableau[i, -1] = np.nan
        tableau[-1, -1] = np.nan
        
        # Construcción del DataFrame para visualización gráfica en Streamlit
        df_tableau = pd.DataFrame(tableau, index=var_basicas + ["Z"], columns=all_cols)
        no_basicas = [col for col in all_cols[:-2] if col not in var_basicas]
        
        info_paso = {
            "paso": paso,
            "tabla": df_tableau.round(4),
            "basicas": var_basicas.copy(),
            "no_basicas": no_basicas,
            "entrante": None,
            "saliente": None,
            "pivote_val": None,
            "razon_min": None,
            "es_final": False
        }
        
        # Condición de Parada: No quedan coeficientes negativos en Z (Solución Óptima)
        if col_pivote_temp == -1:
            info_paso["es_final"] = True
            iteraciones.append(info_paso)
            break
            
        col_pivote = col_pivote_temp
        var_entrante = all_cols[col_pivote]
        info_paso["entrante"] = var_entrante
        
        # Criterio de Variable Saliente: Prueba del Cociente Mínimo no negativo
        cocientes = [tableau[i, -1] if not np.isnan(tableau[i, -1]) and tableau[i, -1] >= 0 else np.inf for i in range(num_rest)]
        fila_pivote = np.argmin(cocientes)
        
        # Evaluación de Modelo No Acotado
        if cocientes[fila_pivote] == np.inf:
            info_paso["mensaje"] = "No acotado"
            iteraciones.append(info_paso)
            break
            
        var_saliente = var_basicas[fila_pivote]
        val_pivote = tableau[fila_pivote, col_pivote]
        
        info_paso["saliente"] = var_saliente
        info_paso["pivote_val"] = round(val_pivote, 4)
        info_paso["razon_min"] = round(cocientes[fila_pivote], 4)
        
        iteraciones.append(info_paso)
        
        # --- Operaciones de Pivoteo Gauss-Jordan ---
        # 1. Normalizar la fila pivote dividiendo por el elemento pivote (convertirlo en 1)
        tableau[fila_pivote, :-1] /= val_pivote
        
        # 2. Eliminar coeficientes de las demás filas (convertir en 0 el resto de la columna)
        for i in range(num_rest + 1):
            if i != fila_pivote:
                fact = tableau[i, col_pivote]
                tableau[i, :-1] -= fact * tableau[fila_pivote, :-1]
                
        # Reemplazar la variable saliente por la entrante en la base
        var_basicas[fila_pivote] = var_entrante
        paso += 1

    # --- PASO 5: Verificación de Infactibilidad y Extracción de Resultados ---
    # Si alguna variable artificial queda en la base con un valor > 0, el problema es infactible
    es_infactible = any(art in var_basicas and tableau[var_basicas.index(art), -2] > 1e-4 for art in artif_names)
    
    solucion_completa = {col: 0.0 for col in all_cols[:-2]}
    for i, var in enumerate(var_basicas):
        solucion_completa[var] = round(tableau[i, -2], 4)
        
    solucion_vars = {f"x{j+1}": solucion_completa[f"x{j+1}"] for j in range(num_vars)}
    val_z = round(tableau[-1, -2] if es_max else -tableau[-1, -2], 4)
    
    return {
        "estado": "Infactible" if es_infactible else "Óptimo",
        "z": val_z,
        "solucion": solucion_vars,
        "solucion_completa": solucion_completa,
        "iteraciones": iteraciones,
        "basicas_final": var_basicas,
        "no_basicas_final": [col for col in all_cols[:-2] if col not in var_basicas]
    }

def resolver_pulp(c, A, b, signos, es_max=True):
    """
    Resuelve el mismo modelo de optimización utilizando la librería industrial PuLP 
    y el motor precompilado en C++ COIN-OR CBC para realizar validación cruzada.
    """
    # Configuración de orientación del problema
    sentido = pulp.LpMaximize if es_max else pulp.LpMinimize
    prob = pulp.LpProblem("Modelo_Validacion", sentido)
    num_vars = len(c)
    
    # Declaración de variables de decisión continuas no negativas (x_i >= 0)
    vars_pulp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)]
    
    # Asignación de la Función Objetivo al problema
    prob += pulp.lpSum([c[i] * vars_pulp[i] for i in range(num_vars)])

    # Incorporación dinámica de restricciones evaluando el operador relacional
    for i in range(len(b)):
        expresion = pulp.lpSum([A[i][j] * vars_pulp[j] for j in range(num_vars)])
        if signos[i] == "<=":
            prob += expresion <= b[i]
        elif signos[i] == ">=":
            prob += expresion >= b[i]
        elif signos[i] == "=":
            prob += expresion == b[i]

    # Ejecución del solver COIN-OR CBC silenciando la salida técnico en consola
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extracción de valores óptimos calculados
    solucion = {f"x{i+1}": round(vars_pulp[i].varValue, 4) if vars_pulp[i].varValue is not None else 0.0 for i in range(num_vars)}
    val_z = round(pulp.value(prob.objective), 4) if pulp.value(prob.objective) is not None else 0.0

    return {"estado": pulp.LpStatus[prob.status], "z": val_z, "solucion": solucion}
