
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import matplotlib.pyplot as plt
import pickle
import numpy as np

class EncoderModel(keras.Model):
    def __init__(self):
        super().__init__()

        #self.mobilenet = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
        #                                   output_shape=[1280],
        #                                   trainable=False)
        self.mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
        self.mobilenet.trainable = False

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dense3 = keras.layers.Dense(3, activation="sigmoid")
        self.encoder = tf.keras.Sequential([self.mobilenet,
                                            self.global_average_layer,
                                            self.dense3])
        self.encoder.build([None] + [224, 224, 3])

    def call(self, inputs):
        out = self.encoder(inputs)
        return out

data_file = open("human_input/human_replay_200_image", 'rb')
memory_list = pickle.load(data_file)
data_file.close()
encoder = EncoderModel()
encoder.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.0005),
    metrics=["accuracy"],
    loss="sparse_categorical_crossentropy"
)

all_actions = np.array([frame for memory in memory_list for frame in memory.actions])
all_states = np.array([frame.astype(np.float32) for memory in memory_list for frame in memory.states])

print(all_states.shape)

tbCallBack = keras.callbacks.TensorBoard(log_dir="classifier", histogram_freq=0, write_graph=True, write_images=True)
filepath = "classifier/{epoch:02d}.hdf5"
saveCallBack = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=5)


print("Start training...")
encoder.fit(all_states,
            all_actions,
            batch_size=100,
            epochs=100,
            callbacks=[tbCallBack, saveCallBack])
print("Done!")

predictions = encoder.predict_on_batch(all_states)
predicitons = [max(prediction) for prediciton in predictions]

correct = 0
incorrect = []
for label, predict, image in zip(all_actions, predictions, all_states):
    if label == predict:
        correct += 1
    else:
        incorrect.append((label, predict, image))
print("Accuracy: {}".format(correct / len(all_actions)))

fig = plt.figure(figsize=(len(incorrect), len(incorrect)))
ax = []

for index, (target, predict, image) in enumerate(incorrect):
    ax.append(fig.add_subplot(224, 224, index + 1))
    ax[-1].set_title("Target: {}, Prediction: {}".format(target, predict))
    plt.imshow(image)

plt.show()



