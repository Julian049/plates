import re
import easyocr

# Mapeos de corrección: caracter erróneo → caracter correcto
_LETTER_TO_NUMBER = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_NUMBER_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}

# Inicializa el lector OCR configurado para detectar texto en español e inglés utilizando la GPU.
def build_reader() -> easyocr.Reader:
    return easyocr.Reader(["es", "en"], gpu=True)

# Ejecuta el motor OCR sobre la imagen o región de interés (ROI) preprocesada.
# Se usa una lista blanca para restringir la detección exclusivamente a caracteres alfanuméricos,
# lo que mejora la precisión al evitar leer ruido o símbolos extraños de la placa.
def read_text(reader: easyocr.Reader, image) -> str:
    # Extrae el texto en un solo bloque restringiendo el diccionario a mayúsculas y dígitos.
    raw = reader.readtext(
        image,
        detail=0,
        paragraph=True,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    # Limpia y une los fragmentos resultantes eliminando cualquier espacio residual.
    return "".join(raw).upper().replace(" ", "")

# Aplica heurísticas de corrección basadas en el estándar de placas vehiculares en Colombia (AAA000).
# Identifica confusiones típicas del OCR (como leer 'O' por '0' o '8' por 'B') y fuerza el
# tipo de carácter correcto según su posición (3 letras seguidas de 3 números).
def correct_plate(text: str) -> str:
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())

    chars = list(clean[:6])
    length = len(chars)

    # Aborta la corrección si hay muy pocos caracteres para formar una placa con sentido.
    if length < 3:
        return clean

    for i in range(min(3, length)):       # zona de letras
        if chars[i].isdigit():
            chars[i] = _NUMBER_TO_LETTER.get(chars[i], chars[i])

    for i in range(3, min(5, length)):    # zona de números
        if chars[i].isalpha():
            chars[i] = _LETTER_TO_NUMBER.get(chars[i], chars[i])

    return "".join(chars)
