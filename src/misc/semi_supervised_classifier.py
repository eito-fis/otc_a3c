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
            'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2?tf-hub-format=compressed',
            output_shape=[1280],
            trainable=False,
        )
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
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

def label_data(model, data, confidence, transformations=5):
    yes_indices = []
    no_indices = []
    unlabeled_indices = []

    for i, image in enumerate(data):
        # transform the image
        # image_transformations = np.array([
        #     augment_color(image) for i in range(transformations)
        # ])
        image_transformations = np.array([image])

        yes_confidences = model(image_transformations)
        yes_confidence = np.mean(yes_confidences)
        if yes_confidence >= confidence:
            yes_indices.append(i)
        elif (1 - yes_confidence) >= confidence:
            no_indices.append(i)
        else:
            unlabeled_indices.append(i)

    return yes_indices, no_indices, unlabeled_indices

def save_data(data, names, dir):
    os.mkdir(dir)

    for image, name in zip(data,names):
        plt.imsave(os.path.join(dir,name),image)
    print('{}: {} images'.format(dir,data.shape[0]))

def load_data(dir):
    data = []
    names = []
    for image_filename in os.listdir(dir):
        if not image_filename.endswith('.png'): continue
        image = plt.imread(os.path.join(dir,image_filename))[:,:,:3]
        data.append(np.array(image))
        names.append(image_filename)
    data = np.array(data,dtype=np.float32)
    names = np.array(names)
    print('{}: {} images with size {}'.format(dir,data.shape[0],tuple(data.shape[1:])))
    return data, names

def make_datasets(yes_data, no_data, batch_size=16, train_eval_split=0.9):
    features = np.concatenate((yes_data,no_data))
    labels = np.array([1]*yes_data.shape[0] + [0]*no_data.shape[0],dtype=np.float32)
    print('Features shape: {}, type: {}'.format(features.shape,features.dtype))
    print('Labels shape: {}, type: {}'.format(labels.shape,labels.dtype))

    permutation = np.random.permutation(len(features))

    features = features[permutation]
    labels = labels[permutation]

    train_count = int(train_eval_split*len(features))
    train_features,eval_features = features[:train_count],features[train_count:]
    train_labels,eval_labels = labels[:train_count],labels[train_count:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features,train_labels))
    train_dataset = train_dataset.shuffle(1000).repeat().batch(batch_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_features,eval_labels))
    eval_dataset = eval_dataset.batch(1)
    return train_dataset,eval_dataset

def make_model(input_size, model_dir, learning_rate):
    model = Classifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.LogLoss(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')],
    )
    sample_input = np.zeros(input_size, dtype=np.float32)[None,:]
    sample_output = model(sample_input)
    if model_dir is not None:
        model.load_weights(model_dir)
    return model

def main(input_dir, output_dir, model_dir, learning_rate, confidence):
    # load all images
    yes_data, yes_data_names = load_data(os.path.join(input_dir,'yes'))
    no_data, no_data_names = load_data(os.path.join(input_dir,'no'))
    unlabeled_data, unlabeled_data_names = load_data(os.path.join(input_dir,'unlabeled'))

    assert yes_data.shape[1:] == no_data.shape[1:] == unlabeled_data.shape[1:], 'images shape should be same'
    input_size = yes_data.shape[1:]

    train_dataset, eval_dataset = make_datasets(yes_data, no_data)
    model = make_model(input_size, model_dir, learning_rate)

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=10,
        validation_data=eval_dataset,
    )

    # store model
    os.makedirs(output_dir)
    new_model_dir = os.path.join(output_dir,'model_dir', 'model')
    model.save_weights(new_model_dir)
    print('Saved model to {}'.format(new_model_dir))

    # label unlabeled data and get indices of new data
    new_yes_indices, new_no_indices, new_unlabeled_indices = label_data(model, unlabeled_data, confidence)

    # split unlabeled data to new labeled data
    new_yes_data, new_yes_data_names = unlabeled_data[new_yes_indices], unlabeled_data_names[new_yes_indices]
    new_no_data, new_no_data_names = unlabeled_data[new_no_indices], unlabeled_data_names[new_no_indices]
    new_unlabeled_data, new_unlabeled_data_names = unlabeled_data[new_unlabeled_indices], unlabeled_data_names[new_unlabeled_indices]

    # add new data to already labeled data
    yes_data, yes_data_names = np.concatenate((yes_data,new_yes_data)),np.concatenate((yes_data_names,new_yes_data_names))
    no_data, no_data_names = np.concatenate((no_data,new_no_data)),np.concatenate((no_data_names,new_no_data_names))

    # save combined data
    save_data(yes_data,yes_data_names,os.path.join(output_dir,'yes'))
    save_data(no_data,no_data_names,os.path.join(output_dir,'no'))
    save_data(new_unlabeled_data,new_unlabeled_data_names,os.path.join(output_dir,'unlabeled'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train classifier')
    parser.add_argument('--input-dir', type=str, required=True, help='directory with subdirectories of yes, no, unlabeled')
    parser.add_argument('--output-dir', type=str, default='./distilled/')
    parser.add_argument('--model-dir', type=str, default=None, help='path to model dir')
    parser.add_argument('--learning-rate', type=float, default=0.00042)
    parser.add_argument('--confidence', type=float, default=0.9)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    learning_rate = args.learning_rate
    confidence = args.confidence

    main(input_dir, output_dir, model_dir, learning_rate, confidence)
