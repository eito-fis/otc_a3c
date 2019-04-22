import tensorflow as tf
import tensorflow_hub as tf_hub

class Teacher(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf_hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
            output_shape=[1280],
            trainable=False,
        )
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(8, activation='relu')
        self.result = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, data):
        data = tf.cast(data, tf.float32) / 255.
        data = self.conv(data)
        data = self.dense1(data)
        data = self.dense2(data)
        data = self.result(data)
        return data
