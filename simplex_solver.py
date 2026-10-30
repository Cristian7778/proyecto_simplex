import numpy as np
import pandas as pd
import pulp

def resolver_simplex_propio(c, A, b, signos, es_max=True):
    # Convertir a problema de maximización
    c_mod = np.array(c, dtype=float) if es_max else -np.array(c, dtype=float)
    A_mod = np.array(A, dtype=float)
    b_mod = np.array(b, dtype=float)
    
    num_vars = len(c)
    num_rest = len(b)
    
    # Crear nombres de variables
    var_names = [f"x{j+1}" for j in range(num_vars)]
    slack_names = [f"s{i+1}" for i in range(num_rest)]
    all_cols = var_names + slack_names + ["RHS"]
    
    # Matriz del Tableau Inicial
    tableau = np.zeros((num_rest + 1, num_vars + num_rest + 1))
    tableau[:num_rest, :num_vars] = A_mod
    tableau[:num_rest, num_vars:num_vars+num_rest] = np.eye(num_rest)
    tableau[:num_rest, -1] = b_mod
    tableau[-1, :num_vars] = -c_mod
    
    var_basicas = slack_names.copy()
    iteraciones = []
    paso = 0
    
    while True:
        # Registrar estado de la iteración actual
        df_tableau = pd.DataFrame(tableau, index=var_basicas + ["Z"], columns=all_cols)
        
        # Identificar variables no básicas
        no_basicas = [col for col in all_cols[:-1] if col not in var_basicas]
        
        # Guardar paso inicial/actual
        info_paso = {
            "paso": paso,
            "tabla": df_tableau,
            "basicas": var_basicas.copy(),
            "no_basicas": no_basicas,
            "entrante": None,
            "saliente": None,
            "pivote_val": None
        }
        
        # Criterio de Optimalidad
        fila_z = tableau[-1, :-1]
        if np.all(fila_z >= -1e-7):
            info_paso["mensaje"] = "¡Solución Óptima Alcanzada!"
            iteraciones.append(info_paso)
            break
            
        # Variable Entrante (columna pivote)
        col_pivote = np.argmin(fila_z)
        var_entrante = all_cols[col_pivote]
        info_paso["entrante"] = var_entrante
        
        # Prueba del Cociente Mínimo (fila pivote)
        col_vals = tableau[:num_rest, col_pivote]
        rhs_vals = tableau[:num_rest, -1]
        
        cocientes = []
        for i in range(num_rest):
            if col_vals[i] > 1e-7:
                cocientes.append(rhs_vals[i] / col_vals[i])
            else:
                cocientes.append(np.inf)
                
        fila_pivote = np.argmin(cocientes)
        if cocientes[fila_pivote] == np.inf:
            info_paso["mensaje"] = "Problema no acotado"
            iteraciones.append(info_paso)
            break
            
        var_saliente = var_basicas[fila_pivote]
        val_pivote = tableau[fila_pivote, col_pivote]
        
        info_paso["saliente"] = var_saliente
        info_paso["pivote_val"] = round(val_pivote, 4)
        iteraciones.append(info_paso)
        
        # Operaciones de Pivoteo (Gauss-Jordan)
        tableau[fila_pivote, :] /= val_pivote
        for i in range(num_rest + 1):
            if i != fila_pivote:
                fact = tableau[i, col_pivote]
                tableau[i, :] -= fact * tableau[fila_pivote, :]
                
        var_basicas[fila_pivote] = var_entrante
        paso += 1

    # Extraer Solución Final
    solucion = {f"x{j+1}": 0.0 for j in range(num_vars)}
    for i, var in enumerate(var_basicas):
        if var.startswith("x"):
            solucion[var] = round(tableau[i, -1], 4)
            
    val_z = round(tableau[-1, -1] if es_max else -tableau[-1, -1], 4)
    
    return {
        "estado": "Óptimo",
        "z": val_z,
        "solucion": solucion,
        "iteraciones": iteraciones
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

    return {
        "estado": pulp.LpStatus[prob.status],
        "z": val_z,
        "solucion": solucion
    }
