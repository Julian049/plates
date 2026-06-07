import argparse
from pathlib import Path

from src.config.config import BEST_MODEL, OUT_DIR
from src.predict.detector import load_model, run_detection

DEFAULT_MODEL = BEST_MODEL

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,  help="Ruta a la imagen de entrada")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al modelo .pt")
    parser.add_argument("--conf", type=float, default=0.30, help="Umbral mínimo de confianza")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not Path(args.model).exists():
        print(f"No se encontro el modelo {args.model}")
        return

    model = load_model(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_detection(args.image, model, OUT_DIR, conf_threshold=args.conf)

if __name__ == "__main__":
    main()