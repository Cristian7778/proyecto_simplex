import numpy as np
import pandas as pd
import pulp

def resolver_simplex_propio(c, A, b, signos, es_max=True):
    num_vars = len(c)
    num_rest = len(b)
    
    c_mod = np.array(c, dtype=float) if es_max else -np.array(c, dtype=float)
    A_mod = np.array(A, dtype=float)
    b_mod = np.array(b, dtype=float)
    
    # Manejar lados derechos negativos (b_i < 0)
    for i in range(num_rest):
        if b_mod[i] < 0:
            b_mod[i] *= -1
            A_mod[i] *= -1
            if signos[i] == "<=":
                signos[i] = ">="
            elif signos[i] == ">=":
                signos[i] = "<="

    var_names = [f"x{j+1}" for j in range(num_vars)]
    slack_names = []
    excess_names = []
    artif_names = []
    
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

    all_cols = var_names + slack_names + excess_names + artif_names + ["LD", "Razón"]
    tableau = np.zeros((num_rest + 1, len(all_cols)))
    var_basicas = []
    
    tableau[:num_rest, :num_vars] = A_mod
    tableau[:num_rest, -2] = b_mod
    
    for i in range(num_rest):
        if i in row_slack_map:
            col_name, val = row_slack_map[i]
            col_idx = all_cols.index(col_name)
            tableau[i, col_idx] = val
            if val == 1.0:
                var_basicas.append(col_name)
                
        if i in row_artif_map:
            artif_name = row_artif_map[i]
            col_idx = all_cols.index(artif_name)
            tableau[i, col_idx] = 1.0
            var_basicas.append(artif_name)

    # Método de la Gran M
    M = 1e6
    tableau[-1, :num_vars] = -c_mod
    
    for artif in artif_names:
        col_idx = all_cols.index(artif)
        tableau[-1, col_idx] = M
        
    for i in range(num_rest):
        if i in row_artif_map:
            tableau[-1, :-1] -= M * tableau[i, :-1]

    # Inicialización correcta de variables de control
    iteraciones = []
    paso = 0

    while True:
        fila_z = tableau[-1, :-2]
        col_pivote_temp = np.argmin(fila_z) if np.min(fila_z) < -1e-7 else -1
        
        # Calcular Razones
        for i in range(num_rest):
            if col_pivote_temp != -1 and tableau[i, col_pivote_temp] > 1e-7:
                tableau[i, -1] = tableau[i, -2] / tableau[i, col_pivote_temp]
            else:
                tableau[i, -1] = np.nan
        tableau[-1, -1] = np.nan
        
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
        
        # Criterio de parada (Solución Óptima)
        if col_pivote_temp == -1:
            info_paso["es_final"] = True
            iteraciones.append(info_paso)
            break
            
        col_pivote = col_pivote_temp
        var_entrante = all_cols[col_pivote]
        info_paso["entrante"] = var_entrante
        
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
