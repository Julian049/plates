import cv2
import numpy as np

# Recorta la región de interés (ROI) de la imagen basándose en las coordenadas detectadas.
# El objetivo es aislar la placa, descartando idealmente franjas inferiores con texto irrelevante.
def crop_plate_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h = y2 - y1
    w = x2 - x1

    # Recortamos 15% arriba y abajo para eliminar concesionario y "COLOMBIA"
    y1_crop = y1 + int(h * 0.15)
    y2_crop = y2 - int(h * 0.15)
    # Recortamos 3% a los lados para evitar bordes negros
    x1_crop = x1 + int(w * 0.03)
    x2_crop = x2 - int(w * 0.03)

    return image[max(0, y1_crop):min(image.shape[0], y2_crop), max(0, x1_crop):min(image.shape[1], x2_crop)]

# Prepara el ROI para maximizar la precisión del OCR mediante técnicas de visión artificial.
# Incluye escalado para caracteres pequeños, reducción de ruido y binarización adaptativa.
def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thresh
