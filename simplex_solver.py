import numpy as np
import pandas as pd
import pulp

def resolver_simplex_propio(c, A, b, es_max=True):
    """
    Algoritmo Simplex Primal paso a paso desde cero.
    """
    c_vec = np.array(c) if es_max else -np.array(c)

    num_vars = len(c_vec)
    num_rest = len(b)

    # Matriz inicial con variables de holgura
    matriz = np.hstack([A, np.eye(num_rest), np.array(b).reshape(-1, 1)])
    
    var_names = [f"x{i+1}" for i in range(num_vars)] + [f"s{i+1}" for i in range(num_rest)]
    var_basicas = [f"s{i+1}" for i in range(num_rest)]
    
    c_line = np.hstack([-c_vec, np.zeros(num_rest + 1)])
    tabla = np.vstack([matriz, c_line])

    iteraciones = []
    
    # Guardar estado inicial (Iteración 0)
    iteraciones.append({
        "paso": 0,
        "tabla": pd.DataFrame(tabla.copy(), columns=var_names + ["RHS"], index=var_basicas + ["Z"]),
        "basicas": list(var_basicas),
        "entrante": None,
        "saliente": None
    })

    paso = 1
    max_iter = 50

    while paso < max_iter:
        z_row = tabla[-1, :-1]
        
        # Criterio de optimalidad
        if np.all(z_row >= -1e-7):
            break

        # Variable entrante (columna pivote)
        col_pivote = np.argmin(z_row)
        var_entrante = var_names[col_pivote]

        # Prueba del cociente mínimo
        col_vals = tabla[:-1, col_pivote]
        rhs_vals = tabla[:-1, -1]

        cocientes = []
        for i in range(num_rest):
            if col_vals[i] > 1e-7:
                cocientes.append(rhs_vals[i] / col_vals[i])
            else:
                cocientes.append(np.inf)

        if np.all(np.isinf(cocientes)):
            return {"estado": "No acotado", "iteraciones": iteraciones}

        fila_pivote = np.argmin(cocientes)
        var_saliente = var_basicas[fila_pivote]
        elem_pivote = tabla[fila_pivote, col_pivote]

        # Pivoteo de Gauss-Jordan
        tabla[fila_pivote] = tabla[fila_pivote] / elem_pivote

        for i in range(len(tabla)):
            if i != fila_pivote:
                tabla[i] = tabla[i] - tabla[i, col_pivote] * tabla[fila_pivote]

        var_basicas[fila_pivote] = var_entrante

        iteraciones.append({
            "paso": paso,
            "tabla": pd.DataFrame(tabla.copy(), columns=var_names + ["RHS"], index=var_basicas + ["Z"]),
            "basicas": list(var_basicas),
            "entrante": var_entrante,
            "saliente": var_saliente
        })

        paso += 1

    # Extraer solución
    solucion = {f"x{i+1}": 0.0 for i in range(num_vars)}
    holguras = {f"s{i+1}": 0.0 for i in range(num_rest)}

    for i, var in enumerate(var_basicas):
        val = tabla[i, -1]
        if var in solucion:
            solucion[var] = round(val, 4)
        elif var in holguras:
            holguras[var] = round(val, 4)

    val_z = tabla[-1, -1]
    if not es_max:
        val_z = -val_z

    return {
        "estado": "Óptimo",
        "z": round(val_z, 4),
        "solucion": solucion,
        "holguras": holguras,
        "iteraciones": iteraciones
    }


def resolver_pulp(c, A, b, signos, es_max=True):
    sentido = pulp.LpMaximize if es_max else pulp.LpMinimize
    prob = pulp.LpProblem("Modelo_Validacion", sentido)

    num_vars = len(c)
    vars_pulp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)]

    # Función Objetivo
    prob += pulp.lpSum([c[i] * vars_pulp[i] for i in range(num_vars)])

    # Restricciones según el signo seleccionado por el usuario
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
