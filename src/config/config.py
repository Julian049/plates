from pathlib import Path

# Buscamos la raíz del proyecto (subiendo desde src/config)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Carpeta raíz de todas las salidas en la raíz del proyecto
OUT_DIR = BASE_DIR / "out"

# Pesos del mejor modelo tras el entrenamiento
BEST_MODEL = OUT_DIR / "runs/train/plates-2/weights/best.pt"