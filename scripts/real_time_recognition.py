import cv2
import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
model = tf.keras.models.load_model('models/gesture_model.h5')

# Mapa de etiquetas a gestos
gesture_map = {0: "Pulgar Arriba", 1: "Pulgar Abajo", 2: "Por favor", 3: "Pausa"}

# Iniciar la cámara web
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar la imagen
    resized = cv2.resize(gray, (64, 64))

    # Normalizar
    resized = resized / 255.0
    resized = resized.reshape(1, 64, 64, 1)

    # Hacer la predicción
    prediction = model.predict(resized)
    predicted_class = np.argmax(prediction, axis=1)[0]
    gesture = gesture_map[predicted_class]

    # Mostrar la predicción en la imagen
    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen con la predicción
    cv2.imshow('Real-time Gesture Recognition', frame)

    # Romper el ciclo con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
