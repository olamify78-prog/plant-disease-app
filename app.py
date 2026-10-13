import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def load_my_model():
    # Use compile=False to bypass metric/optimizer deserialization issues
    return tf.keras.models.load_model("plant_disease_model.h5", compile=False)

model = load_my_model()
