import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from src.pico_placa.rules import DAY_NAMES, RULES

VehicleType = Literal["private", "public_service"]

# Estructura de datos que almacena el resultado completo de la evaluación de pico y placa.
@dataclass
class PicoPlacaResult:
    vehicle_type: str
    plate_text:   str
    last_digit:   int | None
    weekday:      int
    day_name:     str
    restricted_digits: list[int]
    has_restriction:   bool
    reason:       str

# Extrae el último dígito numérico de una placa
def _extract_last_digit(plate_text: str) -> int | None:
    digits = re.findall(r"\d", plate_text)
    if not digits:
        return None
    return int(digits[-1])

# Determina si un vehículo tiene restricción de movilidad en Tunja para una fecha específica.
# Cruza el tipo de vehículo, el día de la semana y el último dígito con las reglas vigentes.
def has_pico_placa(
    vehicle_type: VehicleType,
    plate_text:   str,
) -> PicoPlacaResult:
    date = datetime.today()

    weekday   = date.weekday()
    day_name  = DAY_NAMES[weekday]
    last_digit = _extract_last_digit(plate_text)

    type_rules  = RULES.get(vehicle_type, {})
    restricted  = type_rules.get(weekday, [])

    # Fin de semana o tipo desconocido
    if not restricted:
        return PicoPlacaResult(
            vehicle_type=vehicle_type,
            plate_text=plate_text,
            last_digit=last_digit,
            weekday=weekday,
            day_name=day_name,
            restricted_digits=restricted,
            has_restriction=False,
            reason=f"Los {day_name}s no hay pico y placa en Tunja.",
        )

    # No se pudo leer el último dígito
    if last_digit is None:
        return PicoPlacaResult(
            vehicle_type=vehicle_type,
            plate_text=plate_text,
            last_digit=None,
            weekday=weekday,
            day_name=day_name,
            restricted_digits=restricted,
            has_restriction=False,
            reason=(
                f"No se pudo determinar el último dígito de '{plate_text}'. "
                f"Hoy ({day_name}) restringen los dígitos: {restricted}."
            ),
        )

    # Evalúa finalmente si el dígito extraído se encuentra en la lista de restringidos.
    has_restriction = last_digit in restricted
    if has_restriction:
        reason = (
            f"SÍ tiene pico y placa. Placa '{plate_text}' termina en {last_digit}, "
            f"restringido los {day_name}s para vehículos {vehicle_type}."
        )
    else:
        reason = (
            f"NO tiene pico y placa. Placa '{plate_text}' termina en {last_digit}, "
            f"hoy ({day_name}) solo restringen: {restricted}."
        )

    return PicoPlacaResult(
        vehicle_type=vehicle_type,
        plate_text=plate_text,
        last_digit=last_digit,
        weekday=weekday,
        day_name=day_name,
        restricted_digits=restricted,
        has_restriction=has_restriction,
        reason=reason,
    )