import os
import argparse
import pickle
import cv2

from src.a2c.models.actor_critic_model import ActorCriticModel
from src.a2c.eval.a2c_eval import Memory

import tensorflow as tf
import numpy as np
from PIL import Image

# IMAGE PARAMETERS
NOISE_WEIGHT = .5
IMAGE_SHAPE = (84, 84, 3)
IMAGE_WIDTH, IMAGE_HEIGHT = 84*4, 84*4
MARGIN = 20

# VIDEO PARAMETERS
FPS = 10
WINDOW_WIDTH = (IMAGE_WIDTH*4) + (MARGIN*3)
WINDOW_HEIGHT = (IMAGE_HEIGHT*2) + (MARGIN*2)

# MODEL PARAMETERS
NUM_ACTIONS = 4
STATE_SHAPE = [84,84,3]
STACK_SIZE = 1
CONV_SIZE = 'quake'

def threshold(source_image, threshold_image):
    new_image = np.zeros_like(source_image)
    threshold_image = cv2.normalize(threshold_image, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    for y in range(IMAGE_SHAPE[0]):
        for x in range(IMAGE_SHAPE[1]):
            new_image[x,y] = (np.multiply(threshold_image[x,y],source_image[x,y]))
    new_image = cv2.normalize(new_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return new_image

def superimpose(source_image, noise_image):
    new_image = np.zeros_like(source_image)
    for y in range(IMAGE_SHAPE[0]):
        for x in range(IMAGE_SHAPE[1]):
            new_image[x,y] = (np.multiply(NOISE_WEIGHT,noise_image[x,y]) + np.multiply((1-NOISE_WEIGHT),source_image[x,y]))
    return new_image

def format_images(source_image, saliencymaps, action):
    source_image = cv2.cvtColor(cv2.normalize(source_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_BGR2RGB)

    gray_saliency = saliencymaps[action]
    gray_saliency = cv2.cvtColor(cv2.normalize(gray_saliency, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_BGR2GRAY)
    gray_saliency = superimpose(source_image, gray_saliency)

    source_image = cv2.resize(source_image, dsize=(IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    gray_saliency = cv2.resize(gray_saliency, dsize=(IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

    action_saliencies = []
    for i in range(NUM_ACTIONS):
        action_saliency = saliencymaps[i]
        action_saliency = cv2.cvtColor(cv2.normalize(action_saliency, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_BGR2RGB)
        action_saliency = cv2.resize(action_saliency, dsize=(IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        action_saliencies.append(action_saliency)
    
    return [source_image,   action_saliencies[0],   action_saliencies[3],
            gray_saliency,  action_saliencies[1],   action_saliencies[2]]

def draw_probability_distribution(frame, distribution, action_taken):
    x_start, y_start = 0, MARGIN*3
    width = int(IMAGE_WIDTH / NUM_ACTIONS)
    height = int(IMAGE_HEIGHT - (MARGIN*3))

    rearrange_dist = [1, 0, 2, 3]
    action_labels = ['<-', 'fwd', '->', 'jump']
    cv2.putText(frame,'Action Probabilites',(x_start+width+12,y_start-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    for i in range(NUM_ACTIONS):
        action = rearrange_dist[i]
        x_pos, y_pos = x_start+(width*i), y_start
        x_end, y_end = x_pos+width, y_pos+height
        cv2.rectangle(frame,(x_pos,y_pos),(x_end,y_end),(255,255,255))
        x_pos, y_pos = x_pos, int(y_end - (distribution[action] * height))
        frame[y_pos:y_end,x_pos:x_end,:] = 255
        color = (0,0,255) if action == action_taken else (255,255,255)
        cv2.putText(frame,action_labels[i],(x_pos+18,y_end+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,color)

def draw_current_frame(frame, images):
    x_start, y_start = IMAGE_WIDTH+MARGIN, 0

    # | 0 | 1 | 2 | #
    # | 3 | 4 | 5 | #
    positions = [
        (x_start,y_start), (x_start+(IMAGE_WIDTH+MARGIN),y_start), (x_start+(IMAGE_WIDTH + MARGIN)*2,y_start),
        (x_start,y_start+(IMAGE_HEIGHT+MARGIN)), (x_start+(IMAGE_WIDTH+MARGIN),y_start+(IMAGE_HEIGHT+MARGIN)), (x_start+(IMAGE_WIDTH + MARGIN)*2,y_start+(IMAGE_HEIGHT+MARGIN))
    ]
    labels = [
        'source_image', 'forward_saliency', 'jump_saliency',
        'gray_saliency', 'turnleft_saliency', 'turnright_saliency'
    ]

    for i in range(6):
        x_pos, y_pos = positions[i]
        x_end, y_end = x_pos + IMAGE_WIDTH, y_pos + IMAGE_HEIGHT
        frame[y_pos:y_end,x_pos:x_end,:] = images[i]
        cv2.putText(frame,labels[i],(x_pos,y_end+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser(description='Render one memory.')
    parser.add_argument('--memory-path', required=True, help='path to saved memory object file')
    parser.add_argument('--output-dir', default='a2c_saliency', help='name of the video file')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model')
    parser.add_argument('--smooth', default=False, action='store_true')
    args = parser.parse_args()

    #LOAD FILE#
    data_path = args.memory_path
    data_file = open(data_path, 'rb+')
    memory = pickle.load(data_file)
    data_file.close()

    #PARSE DATA#
    states = memory.states
    probs = np.squeeze(memory.probs)
    actions = np.squeeze(memory.actions)
    print('Memory length: {}'.format(len(states)))

    #INIT VIDEO#
    output_path = os.path.join(args.output_dir, 'gradient_saliency.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, FPS, (WINDOW_WIDTH,WINDOW_HEIGHT))
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    model = ActorCriticModel(num_actions=NUM_ACTIONS,
                     state_size=STATE_SHAPE,
                     stack_size=STACK_SIZE,
                     actor_fc=(512,),
                     critic_fc=(512,),
                     conv_size=CONV_SIZE)
    model.load_weights(args.restore)

    n_samples = 200 if args.smooth else 1

    # Make gradient
    print('Gradient Start!')
    for _ in range(n_samples):
        processed = model.process_inputs(states)
        if args.smooth:
            for i, (obs, floor_data) in enumerate(states):
                stdev = 0.01 * (np.amax(obs) - np.amin(obs))
                noise = np.random.normal(0, stdev, obs.shape)
                processed[0][i] = processed[0][i] + noise
        with tf.GradientTape(persistent=True) as g:
            processed_state = tf.convert_to_tensor(processed[0], dtype=tf.float32)
            g.watch(processed_state)
            processed[0] = processed_state
            processed_floor = tf.convert_to_tensor(processed[1], dtype=tf.float32)
            processed[1] = processed_floor

            logits, value = model(processed)
            logits = tf.squeeze(logits)
            logits_action0 = logits[:,0]
            logits_action1 = logits[:,1]
            logits_action2 = logits[:,2]
            logits_action3 = logits[:,3]
        all_logits = [logits_action0, logits_action1, logits_action2, logits_action3]
        all_saliencymaps = []
        for i in range(NUM_ACTIONS):
            action_saliencymap = np.square(np.squeeze(g.gradient(all_logits[i], processed_state).numpy()))
            all_saliencymaps.append(action_saliencymap)
    all_saliencymaps = np.swapaxes(np.squeeze(np.array(all_saliencymaps)), 0, 1)
    print('Gradient End!')

    for index, (saliencymaps, action, distribution) in enumerate(zip(all_saliencymaps, actions, probs)):
        source_image = states[index][0]
        formatted_images = format_images(source_image, saliencymaps, action)
        frame = np.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3),dtype=np.uint8)
        draw_current_frame(frame, formatted_images)
        draw_probability_distribution(frame, distribution, action)

        gray_saliency = saliencymaps[action]
        gray_saliency = cv2.cvtColor(cv2.normalize(gray_saliency, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), cv2.COLOR_BGR2GRAY)
        threshold_image = threshold(source_image, gray_saliency)
        threshold_image = cv2.cvtColor(cv2.resize(threshold_image, dsize=(IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)

        x_pos, y_pos = 0, IMAGE_HEIGHT + MARGIN
        x_end, y_end = x_pos + IMAGE_WIDTH, y_pos + IMAGE_HEIGHT
        frame[y_pos:y_end,x_pos:x_end,:] = threshold_image
        cv2.putText(frame,'saliency_mask',(x_pos,y_end+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

        video.write(frame)
        print('Frame {} done!'.format(index))
    video.release()
    cv2.destroyAllWindows()

        
        
