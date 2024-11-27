import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(dataset_path='dataset'):  # Asegúrate de que dataset_path esté apuntando a la carpeta correcta
    # Variables
    images = []
    labels = []
    label_map = {'pulgar_arriba': 0, 'pulgar_abajo': 1, 'ok': 2, 'pausa': 3}  # Agrega más gestos si es necesario

    # Cargar las imágenes de cada carpeta de gesto
    for label, idx in label_map.items():
        folder_path = os.path.join(dataset_path, label)  # Ahora busca directamente en dataset/ sin "gestures"
        
        # Verifica que la carpeta de la categoría exista
        if not os.path.exists(folder_path):
            print(f"Carpeta {folder_path} no encontrada.")
            continue
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Intentar cargar la imagen
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
                if image is None:
                    print(f"Advertencia: No se pudo cargar la imagen {image_path}.")
                    continue  # Si la imagen no se pudo cargar, la omitimos

                image = cv2.resize(image, (64, 64))  # Redimensionar la imagen
                images.append(image)
                labels.append(idx)

            except Exception as e:
                print(f"Error al procesar la imagen {image_path}: {e}")

    # Convertir a numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalizar imágenes
    images = images / 255.0  # Normalización de los valores de píxeles

    # Redimensionar a 4D para TensorFlow (samples, width, height, channels)
    images = images.reshape(-1, 64, 64, 1)

    # Dividir en conjuntos de entrenamiento, validación y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
