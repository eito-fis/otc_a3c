import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.perception.model_teacher import Teacher

def lets_do_this(images_dir, model_dir):
    labels = ['open_door', 'closed_door', 'inside_door', 'exit_door']
    paths = [os.path.join(images_dir, label) for label in labels]
    for path in paths: os.makedirs(path)

    models = []
    for label in labels:
        model = Teacher()
        sample_input = tf.convert_to_tensor(np.zeros((224,224,3)),dtype=np.uint8)[None,:]
        sample_output = model(sample_input)

        model_name = 'is_'+label
        model.load_weights(os.path.join(model_dir, 'checkpoints', model_name))
        models.append(model)

    taken = [0] * len(labels)
    all = 0
    files = sorted(os.listdir(images_dir))
    for f in files:
        if '.png' not in f: continue
        image_path = os.path.join(images_dir, f)
        image = Image.open(image_path)
        image_big = image.resize((224,224), Image.NEAREST)

        for i in range(len(labels)):
            confidence = tf.reshape(models[i]([np.array(image_big)]), (-1,)).numpy()[0]

            if confidence >= 0.9:
                new_image_path = os.path.join(paths[i], f)
                os.rename(image_path, new_image_path)
                print('\r{} -> {}'.format(image_path, new_image_path))
                taken[i] += 1
                break
        all += 1

        print('\r{}/{} {:.2f}%, {}'.format(
            all, len(files), all*100./len(files),
            ', '.join(['{} {:.2f}% {}'.format(taken[i], taken[i]*100./len(files), labels[i]) for i in range(len(labels))])
        ), end='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('use teacher model to select more images')
    parser.add_argument('--images-dir', type=str, required=True, help='root directory with labeled sub-directories')
    parser.add_argument('--model-dir', type=str, required=True, help='model directory with logs and checkpoints')
    args = parser.parse_args()

    images_dir = args.images_dir
    model_dir = args.model_dir

    lets_do_this(images_dir, model_dir)
