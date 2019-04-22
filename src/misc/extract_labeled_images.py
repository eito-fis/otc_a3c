import numpy as np
import argparse
import pickle
import sys
import os
from PIL import Image
import random
import tensorflow as tf

from src.perception.model_teacher import Teacher

def closed_door(memory):
    k = 1
    between_doors = 10

    true_inds = set()
    false_inds = set()
    inds = set(range(len(memory.rewards)))
    maybe_false_inds = set(inds)

    last_i = None
    for i in inds:
        if memory.rewards[i] > 0.05 and memory.rewards[i] < 0.15:
            if last_i is not None:
                if i - last_i < between_doors:
                    true_inds |= {j for j in range(last_i+1,last_i+1+k)}
                else:
                    maybe_false_inds -= {j for j in range(last_i,last_i+4)}
            last_i = i

    false_inds = random.sample(maybe_false_inds - true_inds, len(true_inds))
    return true_inds, false_inds

def open_door(memory):
    k = 1
    between_doors = 10

    true_inds = set()
    false_inds = set()
    inds = set(range(len(memory.rewards)))
    maybe_false_inds = set(inds)

    last_i = None
    for i in inds:
        if memory.rewards[i] > 0.05 and memory.rewards[i] < 0.15:
            if last_i is not None:
                if i - last_i < between_doors:
                    true_inds |= {j for j in range(last_i+5,i)}
                else:
                    maybe_false_inds -= {j for j in range(last_i-3,last_i+6)}
            last_i = i

    false_inds = random.sample(maybe_false_inds - true_inds, len(true_inds))
    return true_inds, false_inds

def inside_door(memory):
    k = 1
    between_doors = 10

    true_inds = set()
    false_inds = set()
    inds = set(range(len(memory.rewards)))
    maybe_false_inds = set(inds)

    last_i = None
    for i in inds:
        if memory.rewards[i] > 0.05 and memory.rewards[i] < 0.15:
            if last_i is not None:
                if i - last_i < between_doors:
                    true_inds |= {j for j in range(i+1,i+5)}
                else:
                    maybe_false_inds -= {j for j in range(i-2,i+5)}
            last_i = i

    false_inds = random.sample(maybe_false_inds - true_inds, len(true_inds))
    return true_inds, false_inds

def exit_door(memory):
    k = 3

    true_inds = set()
    false_inds = set()
    inds = set(range(len(memory.rewards)))
    maybe_false_inds = set(inds)

    last_i = None
    for i in inds:
        if memory.rewards[i] > 0.95 and memory.rewards[i] < 1.05:
            true_inds |= {j for j in range(i-k+1,i+1)}
            maybe_false_inds -= {j for j in range(i-10,i+1)}

    false_inds = random.sample(maybe_false_inds - true_inds, len(true_inds))
    return true_inds, false_inds

def model_door(model, memory):
    true_inds = []
    false_inds = []

    batch_size = 128
    threshold = 0.9

    for i in range(0,len(memory.rewards),batch_size):
        predictions = tf.reshape(model(memory.obs[i:i+batch_size]), (-1,)).numpy()
        for j, prediction in enumerate(predictions):
            if prediction >= threshold:
                true_inds.append(i+j)
            elif (1.-prediction) >= threshold:
                false_inds.append(i+j)

    return true_inds, false_inds

def save(true_inds, false_inds, memory_name, memory, true_path, false_path):
    for i in true_inds:
        img_name = os.path.join(true_path, '{}_{}.png'.format(memory_name, i))
        img = Image.fromarray(np.array(memory.obs[i][...,0] * 255., dtype=np.uint8), mode='L')
        img.save(img_name)
    for i in false_inds:
        img_name = os.path.join(false_path, '{}_{}.png'.format(memory_name, i))
        img = Image.fromarray(np.array(memory.obs[i][...,0] * 255., dtype=np.uint8), mode='L')
        img.save(img_name)

def save_all(memory_name, memory, path):
    for i in range(len(memory.obs)):
        img_name = os.path.join(path, '{}_{}.png'.format(memory_name, i))
        img = Image.fromarray(np.array(memory.obs[i][...,0] * 255., dtype=np.uint8), mode='L')
        img.save(img_name)


def lets_do_this(memory_dir, output_dir, model_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    if label != 'all':
        true_path = os.path.join(output_dir, label)
        false_path = os.path.join(output_dir, 'not_'+label)
        os.makedirs(true_path)
        os.makedirs(false_path)

    if model_dir is not None:
        model = Teacher()
        sample_input = tf.convert_to_tensor(np.zeros(224,224,3),dtype=np.uint8)[None,:]
        sample_output = model(sample_input)
        model_name = 'is_'+label
        model.load_weights(os.path.join(model_dir, 'checkpoints', model_name))

    for mi, memory_name in enumerate(os.listdir(memory_dir)):
        print('-> {}'.format(memory_name))
        with open(os.path.join(memory_dir,memory_name), 'rb') as mf:
            memory = pickle.load(mf)

        if type(memory) == type([]):
            memories = memory
        else:
            memories = [memory]
        print('   Number of memories: {}'.format(len(memories)))

        for i, memory in enumerate(memories):
            if label == 'all':
                save_all(memory_name, memory, output_dir)
                print('   #{}: {} images'.format(i+1,len(memory.obs)))
            else:
                if model_dir is not None:
                    true_inds, false_inds = model_door(model, memory)
                elif label == 'closed_door':
                    true_inds, false_inds = closed_door(memory)
                elif label == 'open_door':
                    true_inds, false_inds = open_door(memory)
                elif label == 'inside_door':
                    true_inds, false_inds = inside_door(memory)
                elif label == 'exit_door':
                    true_inds, false_inds = exit_door(memory)
                else:
                    raise Exception('Unknown label "{}"'.format(label))
                save(true_inds, false_inds, memory_name, memory, true_path, false_path)
                print('   #{}: {} true, {} false'.format(i+1,len(true_inds),len(false_inds)))


if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('exract images based on some criteria')
    parser.add_argument('--memory-dir', type=str, required=True, help='directory with memory files')
    parser.add_argument('--output-dir', type=str, required=True, help='directory with extracted images')
    parser.add_argument('--label', type=str, required=True, choices=['closed_door', 'open_door', 'inside_door', 'exit_door', 'all'], help='criteria for the images')
    parser.add_argument('--model-dir', type=str, default=None, help='use model as criteria, path to directory with is_label model')
    args = parser.parse_args()

    memory_dir = args.memory_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    label = args.label

    lets_do_this(memory_dir, output_dir, model_dir, label)
