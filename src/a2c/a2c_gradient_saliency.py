
import os
import argparse
import pickle

from src.a2c.models.actor_critic_model import ActorCriticModel
from src.a2c.eval.a2c_eval import Memory

import tensorflow as tf
import numpy as np
from PIL import Image

def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser(description='Render one memory.')
    parser.add_argument('--memory-path', required=True, help='path to saved memory object file')
    parser.add_argument('--output-dir', default='a2c_saliency', help='name of the video file')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model')
    args = parser.parse_args()

    #LOAD FILE#
    data_path = args.memory_path
    data_file = open(data_path, 'rb+')
    memory = pickle.load(data_file)
    data_file.close()

    #PARSE DATA#
    states, floor_datas = zip(*memory.states)
    states = np.array(states)
    floor_datas = np.array(floor_datas)
    frame_data = zip(states, floor_datas)
    print("Memory length: {}".format(len(states)))

    NUM_ACTIONS = 4
    STATE_SHAPE = [84,84,3]
    STACK_SIZE = 1
    CONV_SIZE = 'quake' #((8,4,16),(4,2,32),(3,1,64))

    model = ActorCriticModel(num_actions=NUM_ACTIONS,
                     state_size=STATE_SHAPE,
                     stack_size=STACK_SIZE,
                     actor_fc=(512,),
                     critic_fc=(512,),
                     conv_size=CONV_SIZE)
    model.load_weights(args.restore)

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    for index in range(len(states)):
        # Make gradient
        tensor_state = tf.convert_to_tensor(states[index], dtype=tf.float32)
        tensor_floor_data = tf.convert_to_tensor(floor_datas[index], dtype=tf.float32)
        action = memory.actions[index][0]
        with tf.GradientTape() as g:
            processed = model.process_inputs([(tensor_state, tensor_floor_data)])
            processed_state = tf.convert_to_tensor(processed[0], dtype=tf.float32)
            g.watch(processed_state)
            processed[0] = processed_state
            processed_floor = tf.convert_to_tensor(processed[1], dtype=tf.float32)
            processed[1] = processed_floor
            logits, _ = model(processed)
            logits = tf.squeeze(logits)
            action_logit = logits[action]
        saliency = np.squeeze(g.gradient(action_logit, processed_state).numpy())

        # Make gray image
        gray_image = VisualizeImageGrayscale(saliency)
        gray_image = (gray_image - np.amin(gray_image)) / (np.amax(gray_image) - np.amin(gray_image))
        gray_image = (gray_image * 255).astype(np.uint8)
        # Make color image
        saliency = (saliency - np.amin(saliency)) / (np.amax(saliency) - np.amin(saliency))
        saliency = (saliency * 255).astype(np.uint8)
        # Make OG image
        original_image = states[index]
        original_image = (original_image - np.amin(original_image)) / (np.amax(original_image) - np.amin(original_image))
        original_image = (original_image * 255).astype(np.uint8)
        # Make combined image
        stacked_gray_image = np.stack((gray_image,) * 3, axis=-1)
        combined_image = np.concatenate([original_image, stacked_gray_image, saliency], axis=1)

        # Save images
        #image = Image.fromarray(saliency, "RGB")
        #color_path = os.path.join(args.output_dir, "color_{}.png".format(index))
        #image.save(color_path)

        #image = Image.fromarray(gray_image, "L")
        #gray_path = os.path.join(args.output_dir, "gray_{}.png".format(index))
        #image.save(gray_path)

        #image = Image.fromarray(original_image, "RGB")
        #original_path = os.path.join(args.output_dir, "original_{}.png".format(index))
        #image.save(original_path)

        image = Image.fromarray(combined_image, "RGB")
        combined_path = os.path.join(args.output_dir, "combined_{}.png".format(index))
        image.save(combined_path)
        print("Frame {} done!".format(index))

    print(gray_image)
    input()

        
        
