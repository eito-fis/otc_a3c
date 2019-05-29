
import os
import argparse
import pickle
import cv2

from src.a2c.models.actor_critic_model import ActorCriticModel
from src.a2c.eval.a2c_eval import Memory

import tensorflow as tf
import numpy as np
from PIL import Image

NOISE_WEIGHT = .5
IMAGE_SHAPE = (84, 84, 3)

def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.max(image_2d)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def superimpose(source_image, noise_image):
    new_image = cv2.normalize(source_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    noise_image = cv2.normalize(noise_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for y in range(IMAGE_SHAPE[0]):
        for x in range(IMAGE_SHAPE[1]):
            new_image[x,y] = (np.multiply(NOISE_WEIGHT,noise_image[x,y]) + np.multiply((1-NOISE_WEIGHT),new_image[x,y]))
    return new_image

def draw_current_frame(frame, image):
    y_start, x_start = 0, 0
    image_height, image_width = 84*4, 256*4
    margin = 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(image_width,image_height), interpolation=cv2.INTER_NEAREST)

    y_pos, x_pos = y_start, x_start
    y_end, x_end = y_pos + image_height, x_pos + image_width
    frame[y_pos:y_end,x_pos:x_end,:] = image

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
    obs, floor_datas = zip(*memory.states)
    obs = np.array(obs)
    floor_datas = np.array(floor_datas)
    frame_data = zip(obs, floor_datas)
    print("Memory length: {}".format(len(obs)))

    #VIDEO PARAMETERS#
    fps    = 3
    width = 256*4
    height  = 84*4

    #INIT VIDEO#
    output_path = os.path.join(args.output_dir, 'gradient_saliency.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

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

    stdev = 0.01 * (np.amax(obs) - np.amin(obs))
    states = memory.states
    actions = [a[0] for a in memory.actions]
    total_gradients = np.zeros_like(obs)
    if args.smooth:
        n_samples = 25
    else:
        n_samples = 1
    magnitude = False

    # Make gradient
    for _ in range(n_samples):
        processed = model.process_inputs(states)
        if args.smooth:
            noise = np.random.normal(0, stdev, obs.shape)
            processed[0] = processed[0] + noise
        with tf.GradientTape() as g:
            processed_state = tf.convert_to_tensor(processed[0], dtype=tf.float32)
            g.watch(processed_state)
            processed[0] = processed_state
            processed_floor = tf.convert_to_tensor(processed[1], dtype=tf.float32)
            processed[1] = processed_floor

            logits, value = model(processed)
            logits = tf.squeeze(logits)
            action_logits = [t[a] for t,a in zip(logits, actions)]
        saliencies = np.square(np.squeeze(g.gradient(action_logits, processed_state).numpy()))
        if magnitude:
            total_gradients += (saliencies * saliencies)
        else:
            total_gradients += saliencies

    for index, saliency in enumerate(total_gradients):
        # Make gray image
        gray_image = VisualizeImageGrayscale(saliency)
        gray_image = (gray_image - np.amin(gray_image)) / (np.amax(gray_image) - np.amin(gray_image))
        gray_image = (gray_image * 255).astype(np.uint8)
        # Make color image
        saliency = (saliency - np.amin(saliency)) / (np.amax(saliency) - np.amin(saliency))
        saliency = (saliency * 255).astype(np.uint8)
        # Make OG image
        original_image = states[index][0]
        original_image = (original_image - np.amin(original_image)) / (np.amax(original_image) - np.amin(original_image))
        original_image = (original_image * 255).astype(np.uint8)
        # Make combined image
        stacked_gray_image = np.stack((gray_image,) * 3, axis=-1)
        superimposed_image = superimpose(original_image, stacked_gray_image)

        combined_image = np.concatenate([original_image, superimposed_image, saliency], axis=1)
        image = Image.fromarray(combined_image, "RGB")
        combined_path = os.path.join(args.output_dir, "combined_{}.png".format(index))
        image.save(combined_path)

        frame = np.zeros((height,width,3),dtype=np.uint8)
        draw_current_frame(frame, combined_image)
        video.write(frame)

        print("Frame {} done!".format(index))
    video.release()
    cv2.destroyAllWindows()

        
        
