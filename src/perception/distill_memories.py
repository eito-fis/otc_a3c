import numpy as np
import argparse
import pickle
import sys
import os
from PIL import Image
import random
import tensorflow as tf

from src.perception.model_student import Student

def lets_do_this(memory_dir, output_dir, model_dir):
    labels = ['end_front_door', 'end_side_door', 'green_closed_door', 'green_opened_door', 'key', 'other', 'start_front_door', 'start_side_door', 'wall']
    paths = [os.path.join(output_dir, label) for label in labels]
    for path in paths: os.makedirs(path, exist_ok=True)

    model = Student()
    sample_input = tf.convert_to_tensor(np.zeros((224,224,3)),dtype=np.uint8)[None,:]
    sample_output = model(sample_input)
    model.load_weights(os.path.join(model_dir, 'checkpoints', 'student'))

    taken = [0] * len(labels)
    all = 0

    memory_names = sorted(os.listdir(memory_dir))

    for mi, memory_name in enumerate(memory_names):
        print('-> {}'.format(memory_name))

        with open(os.path.join(memory_dir,memory_name), 'rb') as mf:
            memory = pickle.load(mf)

        images = [Image.fromarray(np.array(memory.obs[i] * 255., dtype=np.uint8)) for i in range(len(memory.obs))]
        images_big = np.array([np.array(image.resize((224,224), Image.NEAREST)) for image in images])

        probs = model(images_big).numpy()
        save_ids = [
            i for i in range(len(probs)-1)
            if np.max(probs[i]) > 0.9 and np.max(probs[i+1]) > 0.9 and np.argmax(probs[i]) == np.argmax(probs[i+1])]
        for i in save_ids:
            label_i = np.argmax(probs[i])
            new_image_path = os.path.join(paths[label_i], '{}_{}.png'.format(memory_name, i))
            images[i].save(new_image_path)
            print(new_image_path)
            taken[label_i] += 1
        all += len(probs)

        print('{}/{} {:.2f}%, {}'.format(
            mi, len(memory_names), mi*100./len(memory_names),
            ', '.join(['{} {:.2f}% {}'.format(taken[i], taken[i]*100./all, labels[i]) for i in range(len(labels))])
        ))



if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('exract images based on some criteria')
    parser.add_argument('--memory-dir', type=str, required=True, help='directory with memory files')
    parser.add_argument('--output-dir', type=str, required=True, help='directory with extracted images')
    parser.add_argument('--model-dir', type=str, default=None, help='use model as criteria, path to directory with is_label model')
    args = parser.parse_args()

    memory_dir = args.memory_dir
    output_dir = args.output_dir
    model_dir = args.model_dir

    lets_do_this(memory_dir, output_dir, model_dir)
