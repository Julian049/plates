import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.predict.image_utils import crop_plate_roi, preprocess_for_ocr
from src.predict.ocr_utils import read_text, correct_plate

CLASSES = {0: "placa"}
_COLORS = {"placa": (0, 200, 0)}


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _draw_label(image: np.ndarray, label: str, x1: int, y1: int, color: tuple) -> None:
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _load_history(json_path: Path) -> list:
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_history(history: list, json_path: Path) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _generate_pie_chart(history: list, output_dir: Path) -> None:
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
    plt.savefig(output_dir / "chart_pie.png", dpi=120)
    plt.close(fig)


def _generate_bar_chart(history: list, output_dir: Path) -> None:
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

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Umbral 50%")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "chart_bar.png", dpi=120)
    plt.close(fig)


def _log(message: str, logs: list) -> None:
    print(message)
    logs.append(message)


def _no_detection_result(reason: str, image_path: str, logs: list, history: list,
                         json_path: Path, output_dir: Path) -> dict:
    _log(reason, logs)
    history.append({
        "archivo": Path(image_path).name,
        "placa": "—",
        "confianza": 0,
        "categoria": "Sin detección",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_history(history, json_path)
    _generate_pie_chart(history, output_dir)
    _generate_bar_chart(history, output_dir)
    return {
        "logs": logs,
        "result_image": None,
        "pie_chart": str(output_dir / "chart_pie.png"),
        "bar_chart": str(output_dir / "chart_bar.png"),
        "history": history,
        "json_path": str(json_path),
    }


def run_detection(image_path: str, model: YOLO, ocr, output_dir: Path,
                  conf_threshold: float = 0.30) -> dict:
    logs = []
    json_path = output_dir / "detections.json"
    history = _load_history(json_path)

    results = model(image_path, imgsz=640)[0]
    image = cv2.imread(image_path)

    if image is None:
        return _no_detection_result(
            f"No se pudo cargar la imagen: {image_path}",
            image_path, logs, history, json_path, output_dir,
        )

    valid_boxes = [box for box in results.boxes if float(box.conf[0]) >= conf_threshold]

    if not valid_boxes:
        return _no_detection_result(
            "No se detectaron placas con confianza suficiente.",
            image_path, logs, history, json_path, output_dir,
        )

    for box in valid_boxes:
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

    output_path = output_dir / ("result_" + Path(image_path).name)
    cv2.imwrite(str(output_path), image)

    _save_history(history, json_path)
    _generate_pie_chart(history, output_dir)
    _generate_bar_chart(history, output_dir)

    return {
        "logs": logs,
        "result_image": str(output_path),
        "pie_chart": str(output_dir / "chart_pie.png"),
        "bar_chart": str(output_dir / "chart_bar.png"),
        "history": history,
        "json_path": str(json_path),
    }