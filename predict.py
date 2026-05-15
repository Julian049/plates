"""
detect.py — Test the trained model with an image
=======================================================
Usage:
  python detect.py --image photo.jpg
  python detect.py --image photo.jpg --model runs/detect/runs/train/plates_v1/weights/best.pt
"""

import re
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import easyocr

CLASSES = {0: "private", 1: "public_service"}


def correct_license_plate(text_input: str) -> str:
    """
    Corrects common OCR errors based on the
    Colombian license plate pattern: 3 letters + 3 numbers (AAA000)
    """
    LETTER_TO_NUMBER = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    NUMBER_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}

    # Limpiar cualquier caracter que no sea alfanumérico
    clean_text = re.sub(r'[^A-Z0-9]', '', text_input.upper())

    if len(clean_text) < 6:
        return clean_text

    result = list(clean_text[:6])  # tomar solo 6 caracteres

    # Posiciones 0,1,2 deben ser letras
    for i in range(3):
        if result[i].isdigit():
            result[i] = NUMBER_TO_LETTER.get(result[i], result[i])

    # Posiciones 3,4,5 deben ser números
    for i in range(3, 6):
        if result[i].isalpha():
            result[i] = LETTER_TO_NUMBER.get(result[i], result[i])

    return "".join(result)


def detect_plates(image_path: str, model_path: str):
    # Cargar modelo
    model = YOLO(model_path)

    # Inferencia
    results = model(image_path, conf=0.5)[0]
    image = cv2.imread(image_path)

    if len(results.boxes) == 0:
        print("⚠️  No license plates detected.")
        return

    # OCR para leer el texto de la placa
    ocr = easyocr.Reader(["es", "en"], gpu=False)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        plate_class = CLASSES.get(int(box.cls[0]), "unknown")

        # Recortar solo parte superior (evita leer "BOGOTA D.C." etc.)
        height = y2 - y1
        # Usamos 0.7 en lugar de 0.6 para tener un poco más de margen de seguridad
        roi = image[y1 : y1 + int(height * 0.7), x1:x2]

        # Preprocesar para mejorar OCR - (Lógica Mejorada)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # agrandar x2 usando interpolación cúbica para conservar bordes más nítidos
        roi_gray = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # filtro bilateral: suaviza ruido pero respeta los bordes duros de los caracteres
        roi_gray = cv2.bilateralFilter(roi_gray, 11, 17, 17)
        # thresholding adaptativo: mucho más robusto contra reflejos o sombras
        roi_bin = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        # Leer texto — solo letras y números
        text_data = ocr.readtext(roi_bin, detail=0, paragraph=True,
                                 allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        plate_text = "".join(text_data).upper().replace(" ", "")
        plate_text = correct_license_plate(plate_text)  # corregir formato AAA000

        print(f"✅ {plate_class.upper()} | Confidence: {conf:.1%} | Text: {plate_text}")

        # Dibujar en imagen (Interfaz elegante)
        # Naranja (BGR) para público, Verde para particular
        color = (0, 200, 0) if plate_class == "private" else (0, 140, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{plate_text} ({plate_class}) {conf:.0%}"

        # Calcular tamaño del texto para crear un fondo relleno
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Dibujar fondo sólido del texto
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        # Dibujar texto en blanco
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Guardar resultado
    output_path = "result_" + Path(image_path).name
    cv2.imwrite(output_path, image)
    print(f"\n💾 Image saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model",
                        default="runs/detect/runs/train/plates_v1/weights/best.pt")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("   Have you trained it yet? Run first: python train.py")
    else:
        detect_plates(args.image, args.model)