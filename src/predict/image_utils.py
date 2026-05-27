import cv2
import numpy as np

# Recorta la región de interés (ROI) de la imagen basándose en las coordenadas detectadas.
# El objetivo es aislar la placa, descartando idealmente franjas inferiores con texto irrelevante.
def crop_plate_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return image[y1:y2, x1:x2]

# Prepara el ROI para maximizar la precisión del OCR mediante técnicas de visión artificial.
# Incluye escalado para caracteres pequeños, reducción de ruido y binarización adaptativa.
def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Convierte la imagen a escala de grises para simplificar el procesamiento.
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Aplica CLAHE para maximizar el contraste entre el texto y el fondo amarillo.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Suaviza ligeramente para eliminar el ruido provocado por el aumento de contraste.
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blur
