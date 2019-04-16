import os
import argparse
import pickle

import numpy as np
from functools import reduce

import cv2

from src.agents.a3c import ActorCriticModel as A3CModel
from src.agents.a3c import Memory

IMAGE_SIZE = 84
NOISE_WEIGHT = .5
THRESHOLD = 0

IMAGE_SHAPE = (84, 84, 1)
BLUR_COEFF = 10
MASK_RADIUS = 35
STRIDE = 4
OFFSET = 3

def superimpose(source_image, noise_image):
    new_image = cv2.cvtColor(np.copy(source_image), cv2.COLOR_GRAY2BGR)
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
    #       if noise_image[x,y] > THRESHOLD:
            new_image[x,y] = (np.multiply(NOISE_WEIGHT,noise_image[x,y]) + np.multiply((1-NOISE_WEIGHT),new_image[x,y]))
    new_image = cv2.normalize(new_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def blur_images(images, radius):
  return np.array([cv2.blur(image, (radius,radius)) for image in images])
#   return np.array([cv2.GaussianBlur(image, (5,5), 0) for image in source_images])

def generate_mask(image_shape, radius, x, y):
  mask_image = np.zeros(image_shape)
  mask_image[y,x] = 1.
  mask_image = cv2.GaussianBlur(mask_image, (radius,radius), 0)
  mask_image = cv2.normalize(mask_image, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  return mask_image

def generate_masks(image_shape, radius, stride):
  masks = []
  for y in range(image_shape[0])[::stride]:
    for x in range(image_shape[0])[::stride]:
      masks.append(generate_mask(image_shape, radius, x, y))
  return np.array(masks)

def perturb_image(source_image, blurred_image, mask):
  result = np.zeros(IMAGE_SHAPE)
  for y in range(IMAGE_SHAPE[0]):
    for x in range(IMAGE_SHAPE[1]):
        if mask[y,x] > 0: result[y,x] = np.multiply(mask[y,x],blurred_image[y,x]) + np.multiply(1-mask[y,x],source_image[y,x])
  return result

def generate_perturbations(source_image, masks, blur_coeff, stride):
  perturbed_images = []
  blurred_image = cv2.blur(source_image, (blur_coeff,blur_coeff))
  for y in range(IMAGE_SHAPE[0])[::stride]:
    for x in range(IMAGE_SHAPE[1])[::stride]:
      print(x//stride + (y//stride)*(IMAGE_SHAPE[0]//stride))
      perturbed_image = perturb_image(source_image, blurred_image, masks[x//stride + y*IMAGE_SHAPE[0]//stride])
      perturbed_images.append(perturbed_image)
  return perturbed_images

def generate_saliency(model, source_image, prev_states, floor, masks, blur_coeff, stride):
    floor = np.array([floor]).astype(np.float32)
    saliency_map = np.zeros((84,84))
    stacked_state = np.concatenate(prev_states + [source_image], axis=-1).astype(np.float32)
    logits = np.squeeze(model.predict(stacked_state[None,:]))
    blurred_image = cv2.blur(source_image, (blur_coeff,blur_coeff))
    for y in range(IMAGE_SHAPE[0])[::stride]:
        for x in range(IMAGE_SHAPE[1])[::stride]:
            mask = masks[x//stride + (y//stride)*(IMAGE_SHAPE[0]//stride)]
            perturbed_image = perturb_image(source_image, blurred_image, mask)
            stacked_state = np.concatenate(prev_states + [perturbed_image], axis=-1).astype(np.float32)
            perturbed_logits = np.squeeze(model.predict(stacked_state[None,:]))
            saliency = np.square(sum(logits - perturbed_logits)) / 2
            saliency_map = saliency_map + np.multiply(saliency, mask)
            # saliency_map[y,x] = saliency
    saliency_map = cv2.normalize(saliency_map, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rgb_saliency_map = (np.zeros(IMAGE_SHAPE) + saliency_map[:,:,None]) * [1,0,0]
    return rgb_saliency_map

masks = generate_masks((84,84), MASK_RADIUS, STRIDE)

def draw_observation(frame, observation):
    x = 0
    y = 0
    width = 672
    height = 672

    if observation.ndim < 3: # IF RETRO
        image = cv2.cvtColor(observation, cv2.COLOR_GRAY2RGB)
    else:
        image = observation
    image = cv2.resize(image, dsize=(height,width), interpolation=cv2.INTER_NEAREST)
    # image = cv2.normalize(image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame[y:y+height,x:x+width,:] = image

def draw_distributions(frame, distribution, max_dist):
    x = 800
    y = 350
    width = 120
    height = 200
    dx = 5
    dy = 15

    cv2.putText(frame,'{:+.3f}'.format(distribution[1]),(x+dx,y+height+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'camera',(x+dx,y+height+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'left',(x+dx,y+height+3*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x,y+height),(x+width,y+height+width),(255,255,255))
    frame[int(y+height*(1-distribution[1]/max_dist)):y+height,x:x+width,:] = 255

    cv2.putText(frame,'{:+.3f}'.format(distribution[0]),(x+width+dx,y+height+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'forward',(x+width+dx,y+height+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x+width,y+height),(x+2*width,y+height+width),(255,255,255))
    frame[int(y+height*(1-distribution[0]/max_dist)):y+height,x+width:x+2*width,:] = 255

    cv2.putText(frame,'{:+.3f}'.format(distribution[2]),(x+2*width+dx,y+height+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'camera',(x+2*width+dx,y+height+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'right',(x+2*width+dx,y+height+3*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x+2*width,y+height),(x+3*width,y+height+width),(255,255,255))
    frame[int(y+height*(1-distribution[2]/max_dist)):y+height,x+2*width:x+3*width,:] = 255

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser(description='Render one memory.')
    parser.add_argument('--memory-path', required=True, help='path to saved memory object file')
    parser.add_argument('--output', default='conv_saliency/saliency_render.mp4', help='name of the video file')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model')
    args = parser.parse_args()

    #LOAD FILE#
    data_path = args.memory_path
    data_file = open(data_path, 'rb+')
    memory = pickle.load(data_file)
    data_file.close()

    #PARSE DATA#
    states = memory.states
    floors = memory.floors
    frame_data = zip(states, floors)
    # distributions = memory.probs
    # frame_data = zip(observations, distributions, saliency_maps)
    # dist_max = max(map(max, distributions))
    print("Memory length: {}".format(len(states)))

    #VIDEO PARAMETERS#
    fps    = 10.
    width = 1280
    height  = 720

    #INIT VIDEO#
    output = args.output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (width,height))

    NUM_ACTIONS = 4
    STATE_SHAPE = [84,84,1]
    STACK_SIZE = 4
    CONV_SIZE = ((8,4,32),(4,2,64),(3,1,64))

    model = A3CModel(num_actions=NUM_ACTIONS,
                     state_size=STATE_SHAPE,
                     stack_size=STACK_SIZE,
                     actor_fc=(1024,512),
                     critic_fc=(1024,512),
                     conv_size=CONV_SIZE)
    model.load_weights(args.restore)

    prev_states = [np.random.random(STATE_SHAPE) for _ in range(STACK_SIZE)]
    #RENDER VIDEO#
    print("Rendering...")
    for i, (state, floor) in enumerate(frame_data):
        prev_states = prev_states[1:] + [state]
        frame = np.zeros((height,width,3),dtype=np.uint8)
        saliency_map = generate_saliency(model.actor_model, state, prev_states[:-1], floor, masks, BLUR_COEFF, STRIDE)
        image = superimpose(state.astype(np.float32), saliency_map.astype(np.float32))
        cv2.imwrite('conv_saliency/saliency_map_{}.png'.format(i), image)
        draw_observation(frame,image)
        cv2.putText(frame,'Step: {}'.format(i),(750,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        # draw_distributions(frame, dist, dist_max)
        video.write(frame)
    print("Done!")

    video.release()
    cv2.destroyAllWindows()