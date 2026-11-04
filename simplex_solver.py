import numpy as np

def resolver_simplex_propio(c, A, b, signos, es_max=True):
    """
    Simplex de dos fases.
    Soporta <=, >=, =, maximización y minimización.
    """

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    signos = list(signos)

    m, n = A.shape

    # Normalizar RHS negativo
    for i in range(m):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1
            signos[i] = {
                "<=": ">=",
                ">=": "<=",
                "=": "="
            }[signos[i]]

    # Normalizar signos
    signos = [
        {"<=": "<=", "≤": "<=",
         ">=": ">=", "≥": ">=",
         "=": "="}[s]
        for s in signos
    ]

    # Construir variables de holgura, exceso y artificiales
    columnas = [A[:, j].copy() for j in range(n)]
    nombres = [f"x{j+1}" for j in range(n)]
    basicas = [-1] * m
    artificiales = []

    for i in range(m):
        if signos[i] == "<=":
            col = np.zeros(m)
            col[i] = 1
            columnas.append(col)
            nombres.append(f"s{i+1}")
            basicas[i] = len(nombres) - 1

        elif signos[i] == ">=":
            col = np.zeros(m)
            col[i] = -1
            columnas.append(col)
            nombres.append(f"e{i+1}")

            col_a = np.zeros(m)
            col_a[i] = 1
            columnas.append(col_a)
            nombres.append(f"a{i+1}")
            basicas[i] = len(nombres) - 1
            artificiales.append(len(nombres) - 1)

        else:
            col_a = np.zeros(m)
            col_a[i] = 1
            columnas.append(col_a)
            nombres.append(f"a{i+1}")
            basicas[i] = len(nombres) - 1
            artificiales.append(len(nombres) - 1)

    T = np.column_stack(columnas)

    def simplex(T, rhs, basicas, costos):
        while True:
            cb = costos[basicas]
            zj = cb @ T
            zj_rhs = cb @ rhs
            reducido = costos - zj

            # Maximización: elegir costo reducido positivo
            candidatos = [
                j for j in range(len(costos))
                if reducido[j] > 1e-9
                and j not in basicas
            ]

            if not candidatos:
                return T, rhs, basicas, zj_rhs, "Óptimo"

            entrante = max(candidatos, key=lambda j: reducido[j])

            razones = [
                rhs[i] / T[i, entrante]
                if T[i, entrante] > 1e-9 else np.inf
                for i in range(m)
            ]

            saliente = int(np.argmin(razones))

            if razones[saliente] == np.inf:
                return T, rhs, basicas, zj_rhs, "No acotado"

            pivote = T[saliente, entrante]
            T[saliente] /= pivote
            rhs[saliente] /= pivote

            for i in range(m):
                if i != saliente:
                    factor = T[i, entrante]
                    T[i] -= factor * T[saliente]
                    rhs[i] -= factor * rhs[saliente]

            basicas[saliente] = entrante

    # FASE 1: minimizar artificiales equivale a maximizar su negativo
    if artificiales:
        costos_f1 = np.zeros(len(nombres))
        costos_f1[artificiales] = -1

        T, rhs, basicas, z1, estado = simplex(
            T, b.copy(), basicas, costos_f1
        )

        if estado != "Óptimo" or abs(z1) > 1e-7:
            return {
                "estado": "Problema infactible",
                "z": None,
                "solucion": None
            }

        # Eliminar artificiales de la base y de las columnas
        for i in range(m):
            if basicas[i] in artificiales:
                opciones = [
                    j for j in range(len(nombres))
                    if j not in artificiales
                    and j not in basicas
                    and abs(T[i, j]) > 1e-9
                ]

                if opciones:
                    j = opciones[0]
                    pivote = T[i, j]
                    T[i] /= pivote
                    rhs[i] /= pivote

                    for k in range(m):
                        if k != i:
                            factor = T[k, j]
                            T[k] -= factor * T[i]
                            rhs[k] -= factor * rhs[i]

                    basicas[i] = j
                else:
                    return {
                        "estado": "Restricción redundante",
                        "z": None,
                        "solucion": None
                    }

        conservar = [
            j for j in range(len(nombres))
            if j not in artificiales
        ]

        mapa = {viejo: nuevo for nuevo, viejo in enumerate(conservar)}
        T = T[:, conservar]
        nombres = [nombres[j] for j in conservar]
        basicas = [mapa[j] for j in basicas]

    # FASE 2: función objetivo original
    costos = np.zeros(len(nombres))
    costos[:n] = c if es_max else -c

    T, rhs, basicas, z, estado = simplex(
        T, rhs, basicas, costos
    )

    if estado != "Óptimo":
        return {
            "estado": estado,
            "z": None,
            "solucion": None
        }

    solucion = np.zeros(len(nombres))

    for i, j in enumerate(basicas):
        solucion[j] = rhs[i]

    variables = {
        f"x{j+1}": round(solucion[j], 6)
        for j in range(n)
    }

    valor_z = z if es_max else -z

    return {
        "estado": "Óptimo",
        "z": round(valor_z, 6),
        "solucion": variables,
        "basicas_final": [nombres[j] for j in basicas]
    }
