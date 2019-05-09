import tensorflow as tf
import tensorflow_hub as tf_hub

class Student(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf_hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
            output_shape=[1280],
            trainable=False,
        )
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.result = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, data):
        data = tf.cast(data, tf.float32) / 255.
        data = self.conv(data)
        data = self.dense1(data)
        data = self.dense2(data)
        data = self.result(data)
        return data

class StudentO(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv11 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.conv12 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D((2,2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv21 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.conv22 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D((2,2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.conv31 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.conv32 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPool2D((2,2))
        self.dropout3 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.result = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, data):
        data = tf.cast(data, tf.float32) / 255.

        data = self.conv11(data)
        data = self.conv12(data)
        data = self.maxpool1(data)
        data = self.dropout1(data)

        data = self.conv21(data)
        data = self.conv22(data)
        data = self.maxpool2(data)
        data = self.dropout2(data)

        data = self.conv31(data)
        data = self.conv32(data)
        data = self.maxpool3(data)
        data = self.dropout3(data)

        data = self.flatten(data)
        data = self.dense1(data)
        data = self.result(data)
        return data


class StudentOwnConvs(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8,8), (4,4), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (4,4), (2,2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='same')
        self.conv4 = tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='sigmoid')
        self.result = tf.keras.layers.Dense(6, activation='sigmoid')

    def call(self, data):
        data = tf.cast(data, tf.float32) / 255.

        data = self.conv1(data)
        data = tf.keras.activations.relu(data)

        data = self.conv2(data)
        data = tf.keras.activations.relu(data)
        data_tmp = data

        data = self.conv3(data)
        data = tf.math.add(data, data_tmp)
        data_tmp = data
        data = tf.keras.activations.relu(data)

        data = self.conv4(data)
        data = tf.math.add(data, data_tmp)
        data = tf.keras.activations.relu(data)

        data = self.flatten(data)
        data = self.dense1(data)
        data = self.result(data)
        return data
