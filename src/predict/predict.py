import argparse
from pathlib import Path

from src.config.config import BEST_MODEL
from src.predict.detector import load_model, run_detection

DEFAULT_MODEL = BEST_MODEL

# Configura y captura los parámetros ingresados por consola (imagen y modelo).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,  help="Ruta a la imagen de entrada")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al modelo .pt")
    return parser.parse_args()

# Orquesta el flujo de predicción: valida la ruta, carga el modelo y ejecuta la detección.
def main() -> None:
    args = parse_args()

    # Verifica la existencia física del archivo de pesos
    if not Path(args.model).exists():
        print(f"No se encontro el modelo {args.model}")
        return

    model = load_model(args.model)
    run_detection(args.image, model)


if __name__ == "__main__":
    main()