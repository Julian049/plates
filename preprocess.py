import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(path):
    img = cv2.imread(path)

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # gris (reduce complejidad)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # filtro de ruido (MUY importante para el profe)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # normalización
    img = img / 255.0

    return img