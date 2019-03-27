import argparse
import pickle
import numpy as np
import os
import time
import cv2
import sys

def draw_observation(frame, observation,x=100,y=50,width=448,height=448):
    image = cv2.resize(observation, dsize=(height,width), interpolation=cv2.INTER_NEAREST)
    frame[y:y+height,x:x+width] = image

def draw_prev_observations(frame, observations):
    dx = 5
    dy = 5
    width = (448 - (len(observations) - 1)*dx)//len(observations)
    for i, observation in enumerate(reversed(observations)):
        draw_observation(frame,observation,x=100+i*(width+dx),y=448+50+dy,width=width,height=width)

def draw_state(frame, state):
    x = 750
    y = 200
    width = 400
    height = 320

    # total 1280 - state size
    rows = 32
    columns = len(state) // rows

    per_row = height // rows
    per_column = width // columns

    for r in range(rows):
        for c in range(columns):
            frame[y+r*per_row:y+(r+1)*per_row,x+c*per_column:x+(c+1)*per_column] = [255*state[r*columns+c]] * 3

def draw_actions(frame, probabilities):
    x = 750
    y = 596
    width = 120
    height_rectangle = 50
    height_bar = 50

    dx = 5
    dy = 15

    cv2.putText(frame,'{:+.3f}'.format(probabilities[1]),(x+dx,y+height_bar+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'camera',(x+dx,y+height_bar+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'left',(x+dx,y+height_bar+3*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x,y+height_bar),(x+width,y+height_bar+height_rectangle),(255,255,255))
    frame[int(y+height_bar*(1-probabilities[1])):y+height_bar,x:x+width,:] = 255

    cv2.putText(frame,'{:+.3f}'.format(probabilities[0]),(x+width+dx,y+height_bar+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'forward',(x+width+dx,y+height_bar+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x+width,y+height_bar),(x+2*width,y+height_bar+height_rectangle),(255,255,255))
    frame[int(y+height_bar*(1-probabilities[0])):y+height_bar,x+width:x+2*width,:] = 255

    cv2.putText(frame,'{:+.3f}'.format(probabilities[2]),(x+2*width+dx,y+height_bar+dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'camera',(x+2*width+dx,y+height_bar+2*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,'right',(x+2*width+dx,y+height_bar+3*dy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.rectangle(frame,(x+2*width,y+height_bar),(x+3*width,y+height_bar+height_rectangle),(255,255,255))
    frame[int(y+height_bar*(1-probabilities[2])):y+height_bar,x+2*width:x+3*width,:] = 255


def render(output_filename, memory):

    fps    = 5.
    width  = 1280
    height = 720

    # init video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width,height))

    # pack data
    attrs = ['states', 'actions', 'rewards', 'obs', 'probs', 'values']
    data = []
    for attr in attrs:
        attr_data = getattr(memory, attr)
        if np.array(attr_data[0]).ndim > 0:
            data.append([np.array(a) for a in attr_data])
        else:
            data.append(list(attr_data))

    max_state_value = np.max(np.array(data[attrs.index('states')]))
    data[attrs.index('states')] /= max_state_value

    frame_data = zip(*data)
    level = 0
    steps = len(data[0])

    stack_size = 4

    # make video
    for i, (state, action, reward, observation, probabilities, value) in enumerate(frame_data):
        sys.stdout.write('frame {}/{}\r'.format(i+1,steps))
        sys.stdout.flush()

        frame = np.zeros((height,width,3),dtype=np.uint8)

        obs_zeros_num = np.max([stack_size-1-i,0])
        prev_observations = [np.zeros_like(observation) for j in range(obs_zeros_num)] + \
            data[attrs.index('obs')][i-(stack_size-obs_zeros_num-1):i]


        draw_observation(frame,observation)
        draw_prev_observations(frame,prev_observations)
        draw_actions(frame,probabilities)
        draw_state(frame,state)

        cv2.putText(frame,'Step: {}'.format(i),(750,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        cv2.putText(frame,'Level: {}'.format(level),(750,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        cv2.putText(frame,'Reward: {}'.format(reward),(750,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

        if reward >= 1.:
            level += 1

        video.write(frame)
    print('\ndone')

    video.release()

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('render video visualizing memory from games')
    parser.add_argument('--memory', type=str, default='./memories/memory_0_0', help='memory filename')
    parser.add_argument('--output', type=str, default='./memory_0_0.mp4', help='video filename')
    args = parser.parse_args()

    output_filename = args.output
    memory_filename = args.memory

    stack_size = 10
    num_actions = 3
    boredom_thresh = 10

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)


    # attrs = ['states', 'actions', 'rewards', 'obs', 'probs', 'values']
    # for attr in attrs:
    #     arr = getattr(memory, attr)
    #     print('{}: {}'.format(attr, np.array(arr[0]).shape))

    render(output_filename, memory)
