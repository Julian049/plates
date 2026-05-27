from src.pico_placa.checker import PicoPlacaResult

# Colores ANSI
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_VEHICLE_LABEL = {
    "private":        "Particular",
    "public_service": "Servicio Público",
}


def build_report(result: PicoPlacaResult) -> str:
    # Extrae la etiqueta amigable del vehículo y asegura que el dígito se pueda imprimir.
    label   = _VEHICLE_LABEL.get(result.vehicle_type, result.vehicle_type)
    digit   = str(result.last_digit) if result.last_digit is not None else "?"

    if result.has_restriction:
        status_line = f"{_BOLD}{_RED} HOY TIENE PICO Y PLACA{_RESET}"
    else:
        status_line = f"{_BOLD}{_GREEN} HOY NO TIENE PICO Y PLACA{_RESET}"

    # Controla el texto a mostrar en caso de que la lista de dígitos esté vacía (fines de semana).
    restricted_str = (
        str(result.restricted_digits) if result.restricted_digits
        else "ninguno (fin de semana)"
    )

    # Estructura y agrupa todas las líneas del reporte para su posterior impresión.
    lines = [
        f"{_CYAN}{'─'*45}{_RESET}",
        f"{_BOLD}{label}{_RESET}  |  Placa: {_YELLOW}{result.plate_text}{_RESET}",
        f"{result.day_name}",
        f"Último dígito: {_BOLD}{digit}{_RESET}",
        status_line,
        f"Restringidos hoy: {restricted_str}",
        f"{_CYAN}{'─'*45}{_RESET}",
    ]
    return "\n".join(lines)