import argparse
import pickle
import numpy as np
import os
import scipy.misc

def extract_images(memory_filename, output_dir, start_index):
    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)
    images = [np.array(i) for i in memory.obs]
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(output_dir, '{}.png'.format(start_index + i))
        scipy.misc.imsave(path, image)
    print('Done')

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('extract embeddings from memory file')
    parser.add_argument('--output-dir', type=str, default='./images')
    parser.add_argument('--memory', type=str, required=True)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()

    memory_filename = args.memory
    output_dir = args.output_dir
    start_index = args.start_index
    extract_images(memory_filename, output_dir, start_index)
