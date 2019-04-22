import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.perception.model_teacher import Teacher

def lets_do_this(images_dir, model_dir, label):
    path = os.path.join(images_dir, label)
    os.makedir(path)

    model = Teacher()
    sample_input = tf.convert_to_tensor(np.zeros_like(true_images[0]),dtype=np.uint8)[None,:]
    sample_output = model(sample_input)
    print('sample input shape {}'.format(sample_input.shape))
    print('sample output shape {}'.format(sample_output.shape))

    model_name = 'is_'+label
    model.load_weights(os.path.join(model_dir, 'checkpoints', model_name))

    good = 0
    bad = 0
    all = 0
    for f in os.listdir(path):
        if '.png' not in f: continue
        image = Image.open(os.path.join(path, f))
        image_big = image.resize((224,224), Image.NEAREST)

        confidence = tf.reshape(model([image_big]), (-1,)).numpy()[0]

        if confidence >= 0.9:
            image_name = os.path.join(path, '{}_{}_{:.2f}.png'.format(f, label, confidence))
            image.save(image_name)
            print('Saved {}'.format(image_name))
            good += 1
        else:
            bad += 1
        all += 1

        print('{} {:.2f}% good, {} {:.2f}% bad,'.format(good, good*100./all, bad, bad*100./all))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('use teacher model to select more images')
    parser.add_argument('--images-dir', type=str, required=True, help='root directory with labeled sub-directories')
    parser.add_argument('--model-dir', type=str, required=True, help='model directory with logs and checkpoints')
    parser.add_argument('--label', type=str, required=True, help='uses is_label model and creates label directory in images_dir')
    args = parser.parse_args()

    images_dir = args.images_dir
    model_dir = args.model_dir
    label = args.label

    lets_do_this(images_dir, model_dir, label)
