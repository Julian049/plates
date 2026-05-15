"""
train.py — Entrenamiento CNN (YOLOv8) para detección de placas colombianas
==========================================================================
Clases:
  0 = particular
  1 = servicio_publico

Uso:
  python train.py
  python train.py --epochs 100 --batch 8
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main(args):
    # ── 1. Dispositivo ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "0"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  Sin GPU, entrenando en CPU (más lento)")

    # ── 2. Verificar que el dataset existe ──────────────────────────
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        print(f"❌ No se encontró: {yaml_path}")
        print("   Verifica que la carpeta 'dataset/' esté en el mismo lugar que train.py")
        return

    # ── 3. Modelo ───────────────────────────────────────────────────
    # "yolov8n.yaml"  → arquitectura desde cero (sin pesos preentrenados)
    #                   ← esto es lo que pide el profe
    # "yolov8n.pt"    → con pesos COCO (converge mucho más rápido)
    model = YOLO("yolov8n.yaml")   # desde cero ✅

    # ── 4. Entrenamiento ────────────────────────────────────────────
    print(f"\n🚀 Entrenando {args.epochs} épocas | imgsz={args.imgsz} | batch={args.batch}\n")

    results = model.train(
        data=str(yaml_path.resolve()),   # ruta absoluta, evita errores de path
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/train",
        name="placas_v1",
        patience=15,          # early stopping: para si no mejora en 15 épocas
        save=True,
        save_period=10,

        # Augmentación (compensa dataset pequeño)
        flipud=0.0,           # NO voltear verticalmente (placas no van al revés)
        fliplr=0.5,           # sí voltear horizontal
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # ── 5. Resultado ────────────────────────────────────────────────
    best = Path("runs/train/placas_v1/weights/best.pt")
    print(f"\n{'='*45}")
    print(f"  Entrenamiento completado")
    print(f"  Modelo guardado en: {best}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"{'='*45}\n")
    print("Siguiente paso — probar el modelo:")
    print(f"  python detectar.py --imagen foto.jpg --modelo {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=8)
    args = parser.parse_args()
    main(args)