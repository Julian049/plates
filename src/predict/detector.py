from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.config.config import OUT_DIR
from src.predict.image_utils import crop_plate_roi, preprocess_for_ocr
from src.predict.ocr_utils import build_reader, read_text, correct_plate

# Índice de clase → etiqueta interna
CLASSES = {0: "placa"}

# Color BGR por clase (verde = particular, naranja = servicio público)
_COLORS = {"placa": (0, 200, 0)}
DETECTIONS_JSON = OUT_DIR / "detections.json"

# Carga los pesos del modelo YOLOv8 directamente desde el disco.
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

# Dibuja un rectángulo de fondo sólido con texto blanco encima para mejorar la legibilidad.
def _draw_label(image: np.ndarray, label: str, x1: int, y1: int, color: tuple) -> None:
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _load_history() -> list:
    """Lee el JSON acumulativo; si no existe devuelve lista vacía."""
    if DETECTIONS_JSON.exists():
        with open(DETECTIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_history(history: list) -> None:
    """Sobreescribe el JSON con el historial actualizado."""
    with open(DETECTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def _generate_pie_chart(history: list) -> None:
    """
    Torta: proporción de resultados en toda la sesión.
    Categorías: Moto válida / Otro vehículo / Sin detección
    """
    counts = {"Moto válida": 0, "Otro vehículo": 0, "Sin detección": 0}
    for entry in history:
        counts[entry["categoria"]] += 1

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"][:len(labels)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Distribución de resultados", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_pie.png", dpi=120)
    plt.close(fig)


def _generate_bar_chart(history: list) -> None:
    """
    Barras: confianza del modelo por cada detección (últimas 15).
    Solo incluye registros donde hubo detección real (confianza > 0).
    """
    detected = [e for e in history if e["confianza"] > 0][-15:]

    if not detected:
        return

    labels = [e["archivo"][:12] + "…" if len(e["archivo"]) > 12
              else e["archivo"] for e in detected]
    values = [e["confianza"] for e in detected]
    bar_colors = ["#4CAF50" if e["categoria"] == "Moto válida"
                  else "#F44336" for e in detected]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confianza del modelo")
    ax.set_title("Confianza por detección (últimas 15)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Línea de umbral en 50%
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Umbral 50%")
    ax.legend(fontsize=9)

    # Valor encima de cada barra
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_bar.png", dpi=120)
    plt.close(fig)





# Ejecuta el pipeline completo de inferencia sobre una imagen dada.
# 1. Detecta cajas con YOLO. 2. Aplica OCR y corrige lectura por cada caja.
# 3. Verifica restricción de pico y placa. 4. Dibuja resultados y guarda la imagen.
def _log(message: str, logs: list) -> None:
    print(message)
    logs.append(message)


def run_detection(image_path: str, model: YOLO, conf_threshold: float = 0.05) -> dict:
    logs = []
    results = model(image_path, conf=conf_threshold, imgsz=1024)[0]
    image = cv2.imread(image_path)
    history = _load_history()
    result_image_path = None

    if image is None:
        _log(f"No se pudo cargar la imagen: {image_path}", logs)
        return {
            "logs": logs,
            "result_image": None,
            "pie_chart": str(OUT_DIR / "chart_pie.png"),
            "bar_chart": str(OUT_DIR / "chart_bar.png"),
            "history": history,
            "json_path": str(DETECTIONS_JSON),
        }

    if len(results.boxes) == 0:
        _log("No se detectaron placas en la imagen.", logs)

        history.append({
            "archivo": Path(image_path).name,
            "placa": "—",
            "confianza": 0,
            "categoria": "Sin detección",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        _save_history(history)
        _generate_pie_chart(history)
        _generate_bar_chart(history)

        _log(f"Gráficas y JSON actualizados en: {OUT_DIR}", logs)
        return {
            "logs": logs,
            "result_image": None,
            "pie_chart": str(OUT_DIR / "chart_pie.png"),
            "bar_chart": str(OUT_DIR / "chart_bar.png"),
            "history": history,
            "json_path": str(DETECTIONS_JSON),
        }

    ocr = build_reader()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        plate_class = CLASSES.get(int(box.cls[0]), "unknown")
        color = _COLORS.get(plate_class, (0, 255, 0))

        roi = crop_plate_roi(image, x1, y1, x2, y2)
        processed = preprocess_for_ocr(roi)
        raw_text = read_text(ocr, processed)
        plate_text = correct_plate(raw_text)

        is_car = "vehículo diferente" in plate_text
        category = "Otro vehículo" if is_car else "Moto válida"

        _log(f"Placa extraída: {plate_text} | Confianza: {conf:.1%} | {category}", logs)

        box_color = (0, 0, 255) if is_car else color
        label = f"{plate_text if not is_car else 'No es moto'} | {conf:.0%}"
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        _draw_label(image, label, x1, y1, box_color)

        history.append({
            "archivo": Path(image_path).name,
            "placa": plate_text if not is_car else "Otro vehículo",
            "confianza": round(conf, 4),
            "categoria": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    output_path = OUT_DIR / ("result_" + Path(image_path).name)
    cv2.imwrite(str(output_path), image)
    result_image_path = str(output_path)
    _log(f"\nImagen guardada en: {output_path}", logs)

    _save_history(history)
    _generate_pie_chart(history)
    _generate_bar_chart(history)
    _log(f"Gráficas y JSON actualizados en: {OUT_DIR}", logs)

    return {
        "logs": logs,
        "result_image": result_image_path,
        "pie_chart": str(OUT_DIR / "chart_pie.png"),
        "bar_chart": str(OUT_DIR / "chart_bar.png"),
        "history": history,
        "json_path": str(DETECTIONS_JSON),
    }