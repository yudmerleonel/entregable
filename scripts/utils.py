import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('models/gesture_model.h5')
    return model
