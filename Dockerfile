FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from easyocr import Reader; Reader(['es', 'en'], gpu=False)"

COPY src/ src/
COPY static/ static/
COPY templates/ templates/
COPY out/runs/train/plates/weights/best.pt out/runs/train/plates/weights/best.pt
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]