# Dígitos restringidos por día (último dígito de la placa)
RULES: dict[str, dict[int, list[int]]] = {
    "private": {
        0: [3, 4],
        1: [5, 6],
        2: [7, 8],
        3: [9, 0],
        4: [1, 2],
        5: [],
        6: [],
    },
    "public_service": {
        0: [9, 0],
        1: [1, 2],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8],
        5: [],
        6: [],
    },
}

DAY_NAMES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}