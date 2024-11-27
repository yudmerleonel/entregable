import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import preprocess_data

# Cargar los datos preprocesados
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout para evitar el sobreajuste
        layers.Dense(4, activation='softmax')  # 4 clases (pulgar_arriba, pulgar_abajo, ok, pausa)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    model = build_model()
    model.summary()

    # Entrenamos el modelo
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluar el modelo en los datos de prueba
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Precisión en los datos de prueba: {test_acc}")

    # Guardar el modelo entrenado
    model.save("models/gesture_model.h5")  # Guardar el modelo entrenado
    print("Modelo guardado como 'gesture_model.h5'")

if __name__ == "__main__":
    train_model()
