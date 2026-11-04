import numpy as np
import pandas as pd
import pulp

def resolver_simplex_propio(c, A, b, signos, es_max=True):
    # Ajuste según sentido de optimización
    c_mod = np.array(c, dtype=float) if es_max else -np.array(c, dtype=float)
    A_mod = np.array(A, dtype=float)
    b_mod = np.array(b, dtype=float)
    
    num_vars = len(c)
    num_rest = len(b)
    
    var_names = [f"x{j+1}" for j in range(num_vars)]
    slack_names = [f"s{i+1}" for i in range(num_rest)]
    all_cols = var_names + slack_names + ["LD", "Razón"]
    
    # Construcción del Tableau con Razón
    tableau = np.zeros((num_rest + 1, len(all_cols)))
    tableau[:num_rest, :num_vars] = A_mod
    tableau[:num_rest, num_vars:num_vars+num_rest] = np.eye(num_rest)
    tableau[:num_rest, -2] = b_mod
    tableau[-1, :num_vars] = -c_mod
    
    var_basicas = slack_names.copy()
    iteraciones = []
    paso = 0
    
    while True:
        fila_z = tableau[-1, :-2]
        col_pivote_temp = np.argmin(fila_z) if np.min(fila_z) < -1e-7 else -1
        
        # Cálculo de Razones para la columna
        for i in range(num_rest):
            if col_pivote_temp != -1 and tableau[i, col_pivote_temp] > 1e-7:
                tableau[i, -1] = tableau[i, -2] / tableau[i, col_pivote_temp]
            else:
                tableau[i, -1] = np.nan
        tableau[-1, -1] = np.nan
        
        df_tableau = pd.DataFrame(tableau, index=var_basicas + ["Z"], columns=all_cols)
        
        no_basicas = [col for col in var_names + slack_names if col not in var_basicas]
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
        
        # Criterio de parada (óptimo alcanzado)
        if col_pivote_temp == -1:
            info_paso["es_final"] = True
            iteraciones.append(info_paso)
            break
            
        col_pivote = col_pivote_temp
        var_entrante = all_cols[col_pivote]
        info_paso["entrante"] = var_entrante
        
        # Prueba de cociente mínimo
        cocientes = [tableau[i, -1] if not np.isnan(tableau[i, -1]) and tableau[i, -1] >= 0 else np.inf for i in range(num_rest)]
        fila_pivote = np.argmin(cocientes)
        
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
        
        # Operaciones de pivoteo Gauss-Jordan
        tableau[fila_pivote, :-1] /= val_pivote
        for i in range(num_rest + 1):
            if i != fila_pivote:
                fact = tableau[i, col_pivote]
                tableau[i, :-1] -= fact * tableau[fila_pivote, :-1]
                
        var_basicas[fila_pivote] = var_entrante
        paso += 1

    # Construir mapa de solución completa (variables x y holguras s)
    solucion_completa = {}
    for col in var_names + slack_names:
        solucion_completa[col] = 0.0
    for i, var in enumerate(var_basicas):
        solucion_completa[var] = round(tableau[i, -2], 4)
        
    solucion_vars = {f"x{j+1}": solucion_completa[f"x{j+1}"] for j in range(num_vars)}
    val_z = round(tableau[-1, -2] if es_max else -tableau[-1, -2], 4)
    
    return {
        "estado": "Óptimo",
        "z": val_z,
        "solucion": solucion_vars,
        "solucion_completa": solucion_completa,
        "iteraciones": iteraciones,
        "basicas_final": var_basicas,
        "no_basicas_final": [col for col in var_names + slack_names if col not in var_basicas]
    }

def resolver_pulp(c, A, b, signos, es_max=True):
    sentido = pulp.LpMaximize if es_max else pulp.LpMinimize
    prob = pulp.LpProblem("Modelo_Validacion", sentido)
    num_vars = len(c)
    vars_pulp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)]
    prob += pulp.lpSum([c[i] * vars_pulp[i] for i in range(num_vars)])

    for i in range(len(b)):
        expresion = pulp.lpSum([A[i][j] * vars_pulp[j] for j in range(num_vars)])
        if signos[i] == "<=":
            prob += expresion <= b[i]
        elif signos[i] == ">=":
            prob += expresion >= b[i]
        elif signos[i] == "=":
            prob += expresion == b[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solucion = {f"x{i+1}": round(vars_pulp[i].varValue, 4) if vars_pulp[i].varValue is not None else 0.0 for i in range(num_vars)}
    val_z = round(pulp.value(prob.objective), 4) if pulp.value(prob.objective) is not None else 0.0

    return {"estado": pulp.LpStatus[prob.status], "z": val_z, "solucion": solucion}
