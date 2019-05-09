import os
import colorsys
import numpy as np
from PIL import Image


dir = '/Volumes/Storage'
label = 'closed_door'
h_color = 139 / 360.

label_dir = dir+'/'+label
not_label_dir = dir+'/not_'+label

os.makedirs(not_label_dir, exist_ok=True)

files = os.listdir(label_dir)

good = 0
bad = 0
all = 0

for f in files:
    passed, floor, steps, episode, step_i = f.replace('.png', '').split('_')
    passed = passed == 'pass'
    floor = int(floor.replace('floor',''))
    steps = int(steps.replace('steps',''))
    episode = int(episode.replace('episode',''))
    step_i = int(step_i)

    colors = sum([
        abs(colorsys.rgb_to_hsv(*c)[0] - h_color) < 0.05
        for row in np.array(Image.open(label_dir+'/'+f)) for c in row
    ])

    if colors < 10:
        os.rename(label_dir+'/'+f,not_label_dir+'/'+f)
        bad += 1
    else:
        good += 1
    all += 1

    print('\r{}/{} {:.2f}% {} {:.2f}% good {} {:.2f}% bad'.format(
        all,len(files),all*100./len(files),
        good, good*100./all, bad, bad*100./all,
    ),end='')
print('\nDone')
