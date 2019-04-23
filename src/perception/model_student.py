import tensorflow as tf

class Student(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8,8), (4,4), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (4,4), (2,2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='same')
        self.conv4 = tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='same')
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.result = tf.keras.layers.Dense(5, activation='sigmoid')

    def call(self, data):
        data = tf.cast(data, tf.float32) / 255.

        data = self.conv1(data)
        data = tf.keras.activations.relu(data)

        data = self.conv2(data)
        data = tf.keras.activations.relu(data)
        data_tmp = data

        data = self.conv3(data)
        data = tf.concat([data, data_tmp], axis=0)
        data_tmp = data
        data = tf.keras.activations.relu(data)

        data = self.conv4(data)
        data = tf.concat([data, data_tmp], axis=0)
        data = tf.keras.activations.relu(data)

        data = self.dense1(data)
        data = self.result(data)
        return data
