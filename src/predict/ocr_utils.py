import re
import easyocr

# Mapeos de corrección: caracter erróneo → caracter correcto
_LETTER_TO_NUMBER = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "D": "0"}
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

    if "IM" in clean or "VV" in clean:
        temp = clean.replace("IM", "W").replace("VV", "W")

        # Validamos si el reemplazo es lógico.
        # Una placa colombiana tiene números estrictamente en las posiciones 3 y 4.
        if len(temp) >= 5:
            # Verificamos si en la nueva cadena las posiciones 3 y 4 son números
            # (o letras que el OCR suele confundir con números, como la 'S' o la 'O').
            valid_nums = set("0123456789OIZSBGD")
            if temp[3] in valid_nums and temp[4] in valid_nums:
                clean = temp  # El reemplazo encaja perfecto, lo aplicamos.

    chars = list(clean[:6])
    length = len(chars)

    # Aborta la corrección si hay muy pocos caracteres para formar una placa con sentido.
    if length < 5:
        return clean

    for i in range(3):
        if chars[i].isdigit():
            chars[i] = _NUMBER_TO_LETTER.get(chars[i], chars[i])

    for i in range(3, 5):
        if chars[i].isalpha():
            chars[i] = _LETTER_TO_NUMBER.get(chars[i], chars[i])

    if length == 6:
        if chars[5].isdigit():
            chars[5] = _NUMBER_TO_LETTER.get(chars[5], chars[5])

    if chars[3].isdigit() and chars[4].isdigit() and chars[5].isdigit():
        return "La imagen parece corresponder a un vehículo diferente a una moto"
    
    return "".join(chars)