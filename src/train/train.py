import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from src.config.config import OUT_DIR, BEST_MODEL

# Configura y detecta si se usará GPU (CUDA) o CPU para la ejecución.
def resolve_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        return "0"
    print("Sin GPU, entrenando con procesador (más lento)")
    return "cpu"

# Instancia el modelo base YOLOv8
def build_model() -> YOLO:
    return YOLO("yolov8n.pt")

# Entrenamiento adaptado para la detección de placas.
# Configura paciencia de 15 épocas para early stopping y mosaic=1.0 para datasets pequeños.
def train(model: YOLO, yaml_path: Path, args: argparse.Namespace, device: str):
    print(f"\nÉpocas: {args.epochs} | imgsz: {args.imgsz} | batch: {args.batch}\n")

    return model.train(
        data=str(yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(OUT_DIR / "runs/train"),
        name="plates",
        patience=15,
        save=True,
        cls=6.0,
        save_period=10,
        # Parámetros de aumentación de datos adaptados a placas.
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

# Imprime las métricas finales obtenidas
def print_summary(results) -> None:
    print(f"\n{'='*45}")
    print(f"  Entrenamiento completado")
    print(f"  Modelo: {BEST_MODEL}")
    print(f"  mAP50:  {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"{'='*45}")
    print(f"\nSiguiente paso:\n  python predict.py --image foto.jpg --model {BEST_MODEL}")

# Coordina secuencialmente el flujo completo de validación y entrenamiento del modelo.
def main(args: argparse.Namespace) -> None:
    device   = resolve_device()
    yaml_path = Path(args.data)

    model   = build_model()
    results = train(model, yaml_path, args, device)
    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=8)
    main(parser.parse_args())