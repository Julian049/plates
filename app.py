from pathlib import Path
import json

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src.config.config import BEST_MODEL, OUT_DIR
from src.predict.detector import load_model, run_detection

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_history() -> list:
    json_path = OUT_DIR / "detections.json"
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


@app.route("/out/<path:filename>")
def out_file(filename: str):
    return send_from_directory(str(OUT_DIR), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    console_output = None
    history = get_history()
    errors = []

    if request.method == "POST":
        if model is None:
            flash("El modelo de detección no está disponible. Revisa la configuración del servidor.")
            return redirect(url_for("index"))

        if "image" not in request.files:
            flash("No se seleccionó ninguna imagen.")
            return redirect(url_for("index"))

        image_file = request.files["image"]
        if image_file.filename == "":
            flash("No se seleccionó ninguna imagen.")
            return redirect(url_for("index"))

        if not allowed_file(image_file.filename):
            flash("Solo se permiten archivos JPG y PNG.")
            return redirect(url_for("index"))

        filename = secure_filename(image_file.filename)
        image_path = UPLOAD_FOLDER / filename
        image_file.save(str(image_path))

        result = run_detection(str(image_path), model)
        console_output = "\n".join(result.get("logs", []))
        history = result.get("history", history)

        if result.get("result_image"):
            result["result_image_url"] = url_for("out_file", filename=Path(result["result_image"]).name)
        else:
            result["result_image_url"] = None

        result["pie_chart_url"] = url_for("out_file", filename="chart_pie.png")
        result["bar_chart_url"] = url_for("out_file", filename="chart_bar.png")
        result["json_url"] = url_for("out_file", filename="detections.json")

    return render_template(
        "index.html",
        model_error=model_error,
        result=result,
        console_output=console_output,
        history=history,
        max_size_mb=MAX_IMAGE_SIZE // (1024 * 1024),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
