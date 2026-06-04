# Detector de Placas de moto
 
## Opciones de Ejecución
 
El sistema puede ejecutarse de tres formas distintas:
 
1. **Interfaz Web** *(recomendado)*
2. **Docker**
3. **Configuración manual del entorno**

---
 
## Opción 1: Interfaz Web
 
Puedes ejecutar la aplicación web desde la carpeta `plates` con:
 
```bash
pip install -r requirements.txt
python app.py
```
 
Luego abre en el navegador:
 
```
http://127.0.0.1:5000
```
 
La página permite subir una imagen de placa de moto, muestra el resultado del backend, la imagen procesada, ambas gráficas de estadísticas y la tabla con la información de `out/detections.json`.
 
---
 
## Opción 2: Ejecución mediante Docker

### 1. Construir la imagen

```bash
docker build -t detector-placas .
```

### 2. Ejecutar 

Ejecuta el siguiente comando, ajustando el nombre del archivo de la imagen al final:

```bash
docker run --rm \
  -v $HOME:$HOME:ro \
  -v ${PWD}/out:/app/out \
  detector-placas --image /ruta/absoluta.jpg --model best.pt
```

* **`/ruta/absoluta.jpg`:** Es la ruta absoluta de la imagen a analizar

¡IMPORTANTE para Windows!
Para que este comando funcione correctamente, debes ejecutarlo sí o sí desde PowerShell. Si intentas usar el Símbolo del sistema clásico (CMD), el comando fallará porque no reconoce las variables de entorno utilizadas.

## Configuración del Entorno (Opcion manual)

### 1. Crear el entorno virtual (`venv`)

En la raíz del proyecto (`plates/`) genera un entorno limpio de Python 3:

```bash
python3 -m venv venv
```

### 2. Activar el entorno virtual

Antes de ejecutar cualquier código o instalar librerías, activa el entorno:

```bash
source venv/bin/activate
```

### 3. Instalar dependencias necesarias

Asegúrate de tener instaladas las librerías base para procesamiento gráfico, deep learning y OCR:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Guía de Uso y Ejecución

Para evitar problemas con los mapeos de rutas internas (`ModuleNotFoundError`), **siempre debes ejecutar los comandos desde la raíz del proyecto** utilizando el parámetro `-m` (módulo).

### 1. Modo Entrenamiento (`train.py`)

Para lanzar el entrenamiento de la red neuronal encargada de detectar las placas usando tu arquitectura YOLOv8n:

* **Ejecución básica (parámetros por defecto):**
```bash
python -m src.train.train
```

Al finalizar, los pesos optimizados quedarán guardados de manera automática en la ruta `out/runs/train/plates/weights/best.pt`.

### 2. Modo Predicción / Inferencia (`predict.py`)

Este script se encarga de buscar placas en una imagen, leer su texto y exportar el resultado visual dentro del directorio `out/`.

```bash
python -m src.predict.predict --image ~/Downloads/1.jpg
```

En donde `~/Downloads/1.jpg` es la ruta de la imagen a analizar

---

## Estructura del Proyecto

A continuación se detalla la organización de los archivos y la función que cumple cada uno de ellos dentro del sistema:

```text
plates-temp/
│
├── src/
│   │
│   ├── config/
│   │   └── config.py          # Centraliza las rutas absolutas del proyecto (directorios de salida y modelos).
│   │
│   ├── train/
│   │   └── train.py           # Script para entrenar el modelo YOLOv8 desde cero con un dataset personalizado.
│   │
│   └── predict/
│       ├── predict.py         # Punto de entrada CLI principal para procesar imágenes desde la terminal.
│       ├── detector.py        # Orquestador del pipeline: inferencia YOLO, recorte de regiones y envío a OCR.
│       ├── image_utils.py     # Funciones de preprocesamiento de imagen (escala, filtros avanzados y binarización).
│       └── ocr_utils.py       # Inicialización de EasyOCR y lógica de post-corrección bajo el patrón colombiano (AAA000).
│
├── dataset/                   # (Opcional) Carpeta destinada a almacenar las imágenes y el archivo 'data.yaml' de entrenamiento.
├── out/                       # Carpeta autogenerada donde se almacenan los pesos entrenados y los resultados visuales.
└── venv/                      # Entorno virtual de Python con las dependencias instaladas.
```