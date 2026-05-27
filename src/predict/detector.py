from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import OUT_DIR
from src.predict.image_utils import crop_plate_roi, preprocess_for_ocr
from src.predict.ocr_utils import build_reader, read_text, correct_plate
from src.pico_placa.checker import has_pico_placa
from src.pico_placa.reporter import build_report

# Índice de clase → etiqueta interna
CLASSES = {0: "private", 1: "public_service"}

# Color BGR por clase (verde = particular, naranja = servicio público)
_COLORS = {"private": (0, 200, 0), "public_service": (0, 140, 255)}

# Carga los pesos del modelo YOLOv8 directamente desde el disco.
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

# Dibuja un rectángulo de fondo sólido con texto blanco encima para mejorar la legibilidad.
def _draw_label(image: np.ndarray, label: str, x1: int, y1: int, color: tuple) -> None:
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Ejecuta el pipeline completo de inferencia sobre una imagen dada.
# 1. Detecta cajas con YOLO. 2. Aplica OCR y corrige lectura por cada caja.
# 3. Verifica restricción de pico y placa. 4. Dibuja resultados y guarda la imagen.
def run_detection(image_path: str, model: YOLO, conf_threshold: float = 0.5) -> None:
    results = model(image_path, conf=conf_threshold, imgsz=1024)[0]
    image   = cv2.imread(image_path)

    if len(results.boxes) == 0:
        print("No se detectaron placas en la imagen.")
        return

    ocr = build_reader()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf            = float(box.conf[0])
        plate_class     = CLASSES.get(int(box.cls[0]), "unknown")
        color           = _COLORS.get(plate_class, (200, 200, 200))

        # Aplica recorte, preprocesamiento y lectura OCR a la región de la placa.
        roi        = crop_plate_roi(image, x1, y1, x2, y2)
        processed  = preprocess_for_ocr(roi)
        raw_text   = read_text(ocr, processed)
        plate_text = correct_plate(raw_text)

        # Evalúa la restricción de pico y placa y genera el reporte por consola.
        pico = has_pico_placa(vehicle_type=plate_class, plate_text=plate_text)
        print(build_report(pico))

        # Configura y dibuja las etiquetas visuales junto al cuadro delimitador.
        label = f"{plate_text} | {plate_class} | {conf:.0%} "
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        _draw_label(image, label, x1, y1, color)

    # Crea el directorio de salida si no existe y exporta la imagen procesada.
    OUT_DIR.mkdir(exist_ok=True)
    output_path = OUT_DIR / ("result_" + Path(image_path).name)
    cv2.imwrite(str(output_path), image)
    print(f"\nImagen guardada en: {output_path}")