import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.perception.model_student import Student

def load_images(path):
    images = []
    for f in os.listdir(path):
        if '.png' not in f: continue
        image = Image.open(os.path.join(path, f))
        image = image.resize((224,224), Image.NEAREST)
        images.append(np.array(image))
    images = np.array(images)
    print('{}: {} images with shape {}'.format(path, len(images), images.shape[1:]))
    return images

def make_datasets(all_images, labels):
    test_size = 20
    train_datasets = []
    test_features = []
    test_labels = []
    for i in range(len(labels)):
        i_features = all_images[i]
        i_permutation = np.random.permutation(len(i_features))
        i_features = i_features[i_permutation]

        i_labels = np.eye(len(labels))[[i]*len(i_features)]

        test_features.extend(i_features[:test_size])
        test_labels.extend(i_labels[:test_size])

        i_train_dataset = tf.data.Dataset.from_tensor_slices((i_features[test_size:],i_labels[test_size:]))
        i_train_dataset = i_train_dataset.shuffle(1000).repeat()
        train_datasets.append(i_train_dataset)

    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets)
    train_dataset = train_dataset.shuffle(1000).repeat().batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
    test_dataset = test_dataset.batch(64)
    print('train_dataset: {}'.format(train_dataset))
    print('test_dataset: {}'.format(test_dataset))

    return train_dataset, test_dataset

def lets_do_this(images_dir, model_dir):
    all_images = []
    labels = []
    for label in os.listdir(images_dir):
        path = os.path.join(images_dir, label)
        images = load_images(path)
        all_images.append(images)
        labels.append(label)
    print('Labels: {}'.format(', '.join(labels)))

    train_dataset, test_dataset = make_datasets(all_images, labels)

    model = Student()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00042),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    sample_input = tf.convert_to_tensor(np.zeros_like(all_images[0][0]),dtype=np.uint8)[None,:]
    sample_output = model(sample_input)
    print('sample input shape {}'.format(sample_input.shape))
    print('sample output shape {}'.format(sample_output.shape))

    model_name = 'student'
    log_dir = os.path.join(model_dir, 'logs', model_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    model.fit(
        x=train_dataset,
        epochs=100,
        steps_per_epoch=50,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback],
    )
    model.save_weights(os.path.join(model_dir, 'checkpoints', model_name))
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('student network for semi-supervised learning')
    parser.add_argument('--images-dir', type=str, required=True, help='root directory with labeled sub-directories')
    parser.add_argument('--model-dir', type=str, required=True, help='model directory with logs and checkpoints')
    args = parser.parse_args()

    images_dir = args.images_dir
    model_dir = args.model_dir

    lets_do_this(images_dir, model_dir)
