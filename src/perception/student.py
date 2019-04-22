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
        image = Image.open(os.path.join(path, f))
        image = image.resize((224,224), Image.NEAREST)
        images.append(np.array(image))
    images = np.array(images)
    print('{}: {} images with shape {}'.format(path, len(images), images.shape[1:]))
    return images

def make_datasets(*all_images):
    features = np.concatenate(all_images)
    labels = []
    for i, images in enumerate(all_images):
        labels += [i]*len(images)
    labels = np.array(labels)
    labels = np.eye(len(all_images))[labels]
    print('features shape {}'.format(features.shape))
    print('labels shape {}'.format(labels.shape))

    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.05)
    print('train_features shape {}'.format(train_features.shape))
    print('train_labels shape {}'.format(train_labels.shape))
    print('test_features shape {}'.format(test_features.shape))
    print('test_labels shape {}'.format(test_labels.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features,train_labels))
    train_dataset = train_dataset.shuffle(100).repeat().batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
    test_dataset = test_dataset.batch(64)
    print('train_dataset: {}'.format(train_dataset))
    print('test_dataset: {}'.format(test_dataset))

    return train_dataset, test_dataset

def lets_do_this(images_dir, model_dir):
    open_door_path = os.path.join(images_dir, 'open_door')
    closed_door_path = os.path.join(images_dir, 'closed_door')
    inside_door_path = os.path.join(images_dir, 'inside_door')
    exit_door_path = os.path.join(images_dir, 'exit_door')
    else_path = os.path.join(images_dir, 'else')

    open_door_images = load_images(open_door_path)
    closed_door_images = load_images(closed_door_path)
    inside_door_images = load_images(inside_door_path)
    exit_door_images = load_images(exit_door_path)
    else_images = load_images(else_path)

    train_dataset, test_dataset = make_datasets(open_door_images, closed_door_images, inside_door_images, exit_door_images, else_images)

    model = Student()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00042),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    sample_input = tf.convert_to_tensor(np.zeros_like(open_door_images[0]),dtype=np.uint8)[None,:]
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
