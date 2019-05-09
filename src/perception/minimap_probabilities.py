import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import tensorflow as tf
from PIL import Image

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from src.perception.model_student import Student

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def actions_to_xy_angles(memory):
    angle_per_turn = np.pi / 10.
    a_to_m = { 0: [1., 0.], 1: [1., angle_per_turn], 2: [1., -angle_per_turn], 3: [1., 0.] }

    xy = [[0., 0.]]
    angles = [0.]
    angle = 0.
    x, y = 0., 0.

    for i in range(memory.steps-1):
        a = memory.actions[i]
        rad, ang = a_to_m[a]
        angle += ang

        dx, dy = pol2cart(rad, angle)
        x += dx
        y += dy
        xy.append([x,y])
        angles.append(angle)
    xy = np.array(xy)
    return xy, angles

def preprocess(memory):
    for i in range(len(memory.rewards)):
        if memory.rewards[i] > 0.95 and memory.rewards[i] < 1.05: break
    memory.steps = i+1
    memory.actions = np.array([a[0] if isinstance(a, list) else a for a in memory.actions])
    memory.obs = np.array([
        np.array(Image.fromarray(np.array(obs*225.,dtype=np.uint8)).resize((224,224), Image.NEAREST))
        for obs in memory.obs
    ])
    print('{} steps'.format(memory.steps))

def lets_do_this(memory_path, output_path, model_dir):
    labels = [
        'end_front_door',
        'end_side_door',
        'green_closed_door',
        'green_opened_door',
        'key',
        'other',
        'start_front_door',
        'start_side_door',
        'wall',
    ]

    print('-> {}'.format(memory_path))
    with open(memory_path, 'rb') as mf:
        memory = pickle.load(mf)[0]

    preprocess(memory)

    model = Student()
    sample_input = tf.convert_to_tensor(np.zeros((224,224,3)),dtype=np.uint8)[None,:]
    sample_output = model(sample_input)
    print('model sample input shape {}'.format(sample_input.shape))
    print('model sample output shape {}'.format(sample_output.shape))
    model.load_weights(model_dir)

    all_probs = model(np.array(memory.obs[:memory.steps])).numpy()

    color_to_labels = {
        '#f7c120': ['end_front_door', 'end_side_door'],
        '#2dc100': ['green_closed_door', 'green_opened_door'],
        '#ffbb00': ['key'],
        '#848484': ['start_front_door', 'start_side_door'],
    }
    vision_steps = 10.
    field_of_view = np.pi * 0.5

    xy, angles = actions_to_xy_angles(memory)

    for color, color_labels in color_to_labels.items():
        for (x0,y0), (x1,y1) in zip(*[xy[:-1],xy[1:]]):
            if x1 != x0 or y1 != y0:
                plt.arrow(x0, y0, (x1-x0), (y1-y0), length_includes_head=True, head_width=0.3, head_length=0.4, linewidth=0.5, color='black')

        labels_ids = [labels.index(l) for l in color_labels]
        color_probs = np.sum(all_probs[:,labels_ids],axis=-1)

        for i, xy0 in enumerate(xy):
            plt.gca().add_patch(mpatches.Wedge(
                xy0,
                vision_steps,
                180./np.pi*(angles[i]-field_of_view/2.),
                180./np.pi*(angles[i]+field_of_view/2.),
                alpha=1./vision_steps*color_probs[i],
                facecolor=color,
                edgecolor=None,
            ))
        plt.axis('equal')
        plt.axis('off')
        plt.rcParams['figure.figsize'] = 20, 20
        name = output_path.replace('.png', '_{}.png'.format(color_labels[0].split('_')[0]))
        plt.savefig(name, dpi=1000)
        plt.clf()

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('make minimap image of memory')
    parser.add_argument('--memory', type=str, required=True, help='memory filename')
    parser.add_argument('--output', type=str, required=True, help='minimap image filename')
    parser.add_argument('--model-dir', type=str, default=None, help='path to model directory with student model with 9 labels')
    args = parser.parse_args()

    memory_path = args.memory
    output_path = args.output
    model_dir = args.model_dir

    lets_do_this(memory_path, output_path, model_dir)
