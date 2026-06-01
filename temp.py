import os
from pathlib import Path

# Aquí pones la ruta a tus carpetas de etiquetas.
# Repite este proceso para tus carpetas de 'train', 'val' y 'test'
def limpiar_etiquetas():
    # Aquí pones la ruta a tus carpetas de etiquetas.
    # Asegúrate de que coincidan con la estructura de tu proyecto.
    carpetas_etiquetas = [
        Path("dataset/train/labels"),
        Path("dataset/valid/labels"),
        Path("dataset/test/labels")  # Si no existe, el código la saltará automáticamente
    ]

    for carpeta in carpetas_etiquetas:
        # CORRECCIÓN: 'not' en lugar de 'no'
        if not carpeta.exists():
            print(f"La carpeta {carpeta} no existe, saltando...")
            continue

        archivos_modificados = 0

        # Busca todos los archivos .txt de YOLO en la carpeta
        for txt_file in carpeta.glob("*.txt"):
            with open(txt_file, "r") as f:
                lineas = f.readlines()

            nuevas_lineas = []
            for linea in lineas:
                partes = linea.strip().split()
                # Verifica que sea una línea válida de YOLO (Clase + 4 coordenadas)
                if len(partes) >= 5:
                    # El secreto está aquí: forzamos que la clase sea '0'
                    partes[0] = "0"
                    nuevas_lineas.append(" ".join(partes) + "\n")

            # Sobrescribe el archivo con la clase corregida
            with open(txt_file, "w") as f:
                f.writelines(nuevas_lineas)

            archivos_modificados += 1

        if archivos_modificados > 0:
            print(f"Se unificaron {archivos_modificados} archivos en {carpeta}")

    print("\n¡Proceso finalizado! Todas tus etiquetas son ahora de la clase 0 (placa_moto).")

# El main que faltaba para ejecutarlo correctamente
if __name__ == "__main__":
    print("Iniciando unificación de clases...")
    limpiar_etiquetas()