import tensorflow as tf

model = tf.keras.models.load_model('model/neural.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open('model/neural.tflite', 'wb') as f:
    f.write(tflite_model)
