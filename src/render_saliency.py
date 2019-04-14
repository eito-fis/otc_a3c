import os
import argparse
import pickle

import numpy as np
import cv2

from src.agents.a3c import Memory

IMAGE_SIZE = 168
NOISE_WEIGHT = .5
THRESHOLD = 0

def superimpose(source_image, noise_image):
  new_image = np.copy(source_image)
  for y in range(IMAGE_SIZE):
    for x in range(IMAGE_SIZE):
#       if noise_image[x,y] > THRESHOLD:
      new_image[x,y] = np.multiply(NOISE_WEIGHT,noise_image[x,y]) + np.multiply((1-NOISE_WEIGHT),new_image[x,y])
  return new_image

def draw_observation(frame, observation):
    x = 0
    y = 0
    width = 672
    height = 672

    if observation.ndim < 3: # IF RETRO
        image = cv2.cvtColor(observation, cv2.COLOR_GRAY2RGB)
    else:
        # print(observation.shape)
        # image = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        image = observation
    image = cv2.resize(image, dsize=(height,width), interpolation=cv2.INTER_NEAREST)
    image = cv2.normalize(image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    parser.add_argument('--output', default='memory_output/output_memory.mp4', help='name of the video file')
    args = parser.parse_args()

    #LOAD FILE#
    data_path = args.memory_path
    data_file = open(data_path, 'rb+')
    memory = pickle.load(data_file)
    data_file.close()

    #PARSE DATA#
    observations = memory.obs
    distributions = np.squeeze(memory.probs)
    saliency_maps = memory.novelty
    frame_data = zip(observations, distributions, saliency_maps)
    dist_max = max(map(max, distributions))

    #VIDEO PARAMETERS#
    fps    = 10.
    width = 1280
    height  = 720

    #INIT VIDEO#
    output = args.output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (width,height))

    #RENDER VIDEO#
    print("Rendering...")
    for i, (obs, dist, saliency_map) in enumerate(frame_data):
        frame = np.zeros((height,width,3),dtype=np.uint8)
        image = superimpose(obs, saliency_map.astype(np.float32))
        draw_observation(frame,image)
        cv2.putText(frame,'Step: {}'.format(i),(750,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        draw_distributions(frame, dist, dist_max)

        video.write(frame)
    print("Done!")

    video.release()
    cv2.destroyAllWindows()