from pathlib import Path
import json
import time
import uuid

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for, session
from werkzeug.utils import secure_filename

from src.config.config import BEST_MODEL, OUT_DIR
from src.predict.detector import load_model, run_detection
from src.predict.ocr_utils import build_reader

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Aseguramos que la base exista

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "replace_with_a_secure_key"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

try:
    model = load_model(str(BEST_MODEL))
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

try:
    print("Cargando modelo OCR en memoria...")
    ocr_global = build_reader()
except Exception as exc:
    ocr_global = None
    print(f"Error cargando OCR: {exc}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_dir() -> Path:
    """Retorna y asegura que exista la carpeta específica del usuario actual."""
    if 'user_id' not in session:
        session['user_id'] = uuid.uuid4().hex

    user_dir = OUT_DIR / session['user_id']
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_history(user_dir: Path) -> list:
    json_path = user_dir / "detections.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return []
    return []


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("La imagen es demasiado grande. El máximo permitido es 5 MB.")
    return redirect(url_for("index"))


@app.route("/out/<user_id>/<path:filename>")
def out_file(user_id: str, filename: str):
    user_dir = OUT_DIR / user_id
    return send_from_directory(str(user_dir), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    user_dir = get_user_dir()
    user_id = session['user_id']

    if request.method == "POST":
        if model is None:
            flash("El modelo no está disponible.")
            return redirect(url_for("index"))

        if "image" not in request.files or request.files["image"].filename == "":
            flash("No se seleccionó ninguna imagen.")
            return redirect(url_for("index"))

        image_file = request.files["image"]
        if not allowed_file(image_file.filename):
            flash("Solo se permiten archivos JPG y PNG.")
            return redirect(url_for("index"))

        filename = secure_filename(image_file.filename)
        image_path = UPLOAD_FOLDER / filename
        image_file.save(str(image_path))

        result = run_detection(str(image_path), model, ocr_global, user_dir, conf_threshold=0.30)

        session['show_results'] = True
        session['console_output'] = "\n".join(result.get("logs", []))
        session['result_image_name'] = Path(result["result_image"]).name if result.get("result_image") else None

        return redirect(url_for("index"))

    result = None
    console_output = None
    history = get_history(user_dir)

    if session.pop('show_results', False):
        img_name = session.pop('result_image_name', None)
        timestamp = int(time.time())

        result = {
            "result_image_url": f"{url_for('out_file', user_id=user_id, filename=img_name)}?t={timestamp}" if img_name else None,
            "pie_chart_url": f"{url_for('out_file', user_id=user_id, filename='chart_pie.png')}?t={timestamp}",
            "bar_chart_url": f"{url_for('out_file', user_id=user_id, filename='chart_bar.png')}?t={timestamp}",
            "json_url": url_for("out_file", user_id=user_id, filename="detections.json")
        }
        console_output = session.pop('console_output', None)

    return render_template(
        "index.html",
        model_error=model_error,
        result=result,
        console_output=console_output,
        history=history,
        max_size_mb=MAX_IMAGE_SIZE // (1024 * 1024),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)