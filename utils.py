import os
import numpy as np
from preprocess import preprocess_image

labels = {"carro": 0, "moto": 1, "otro": 2}

def load_dataset(dataset_path):
    X = []
    y = []

    for label in labels.keys():
        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            img = preprocess_image(path)

            X.append(img)
            y.append(labels[label])

    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)

    return X, y