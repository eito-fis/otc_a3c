import argparse
import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np
import os
import matplotlib.pyplot as plt

class Classifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.convolutions = tf_hub.KerasLayer(
            'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2',
            output_shape=[1280],
            trainable=False,
        )
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, actiovation='relu')
        self.result = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, data):
        data = self.convolutions(data)
        data = self.dense1(data)
        data = self.dense2(data)
        data = self.result(data)
        return data

def augment_color(image):
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image

def label_data(model, data, confidence):
    for image in data:
        # transform the image
        image_transformations = []
        image_transformations.append([augment_color(image) for i in range(3)])

        # # TODO:
        pass


def store_data(data, dir):
    os.mkdir(dir)

    for i, image in enumerate(data):
        plt.imsave(image,os.path.join(dir,'{}.png'.format(i)))
    print('{}: {} images'.format(dir,data.shape[0]))

def load_data(dir):
    data = []
    names = []
    for image_filename in os.listdir(dir):
        image = plt.imread(os.path.join(dir,image_filename))
        data.append(np.array(image))
        names.append(image_filename)
    data = np.array(data)
    print('{}: {} images with size {}'.format(dir,data.shape[0],tuple(data.shape[1:])))
    return data, names

def make_dataset(yes_data, no_data, batch_size=16):
    features = np.concatenate((yes_data,no_data))
    labels = np.array([1]*yes_data.shape[0] + [0]*no_data.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def make_model(input_size, learning_rate=0.01, model_path=None):
    model = Classifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='log_loss',
    )
    model.build((None,)+tuple(input_size))
    if model_path is not None:
        model.load_weights(model_path)
    return model

def main(input_dir, output_dir, model_path, confidence):
    # load all images
    yes_data, yes_data_names = load_data(os.path.join(input_dir,'yes'))
    no_data, no_data_names = load_data(os.path.join(input_dir,'no'))
    unlabeled_data, unlabeled_data_names = load_data(os.path.join(input_dir,'unlabeled'))

    assert yes_data.shape[1:] == no_data.shape[1:] == unlabeled_data.shape[1:], 'images shape should be same'
    input_size = yes_data.shape[1:]

    dataset = make_dataset(yes_data, no_data)
    model = make_model(input_size, model_path)

    model.fit(
        dataset,
        epochs=100,
        steps_per_epoch=10,
    )

    # label unlabeled data and get indices of new data
    new_yes_indices, new_no_indices, new_unlabeled_indices = label_data(model, unlabeled_data, confidence)

    # split unlabeled data to new labeled data
    new_yes_data, new_yes_data_names = unlabeled_data[new_yes_indices], unlabeled_data_names[new_yes_indices]
    new_no_data, new_no_data_names = unlabeled_data[new_no_indices], unlabeled_data_names[new_no_indices]
    new_unlabeled_data, new_unlabeled_data_names = unlabeled_data[new_unlabeled_indices], unlabeled_data_names[new_unlabeled_indices]

    # add new data to already labeled data
    yes_data, yes_data_names = np.concatenate((yes_data,new_yes_data)),np.concatenate((yes_data_names,new_yes_data_names))
    no_data, no_data_names = np.concatenate((no_data,new_no_data)),np.concatenate((no_data_names,new_no_data_names))

    # store combined data and model
    os.makedirs(output_dir)
    store_data(yes_data,os.path.join(output_dir,'yes'))
    store_data(no_data,os.path.join(output_dir,'no'))
    store_data(new_unlabeled_data,os.path.join(output_dir,'unlabeled'))
    new_model_path = os.path.join(output_dir,'model.h5')
    model.save_weights(new_model_path)
    print('Saved model to {}'.format(new_model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train classifier')
    parser.add_argument('--input-dir', type=str, required=True, help='directory with subdirectories of yes, no, unlabeled')
    parser.add_argument('--output-dir', type=str, default='./distilled/')
    parser.add_argument('--model-path', type=str, default=None, help='.h5 file of weights')
    parser.add_argument('--confidence', type=float, default=0.8)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_path = args.model_path
    confidence = args.confidence

    main(input_dir, output_dir, model_path, confidence)
