import os
import numpy as np
from PIL import Image
import colorsys

def cross_entropy(arr):
    arr = np.array(arr, dtype=np.float)
    arr /= np.sum(arr)
    return -np.sum(arr * np.log(arr))


def color_it(in_name, out_name):
    near_color = 122 / 360.

    image = Image.open(in_name)
    image = np.array(image)
    max = int(0.55 * 255)
    min = int(0.45 * 255)

    for i in range(168):
        for j in range(168):
            good = False
            for c in range(3):
                if image[i,j,c] >= min and image[i,j,c] <= max:
                    good = True
                    break
            imin = np.min(image[i,j])
            imax = np.max(image[i,j])
            good = good and image[i,j,1]-np.mean(image[i,j,0::2]) > 20.
            for c in range(3):
                if good:
                    image[i,j,c] = (image[i,j,c] - imin) *255. / (imax-imin) if (imax-imin) != 0 else 0.
                    # image[i,j,c] = image[i,j,c] * cross_entropy(image[i,j])
                else:
                    image[i,j,c] = 0.
            h,_,_ = colorsys.rgb_to_hsv(*(image[i,j]/255.))
            if abs(h-near_color) > 0.1:
                for c in range(3):
                    image[i,j,c] = 0.

    image = np.clip(image,0,255)
    Image.fromarray(image).save(out_name)

os.makedirs('/Volumes/Storage/selected_green_cool/',exist_ok=True)
for f in os.listdir('/Volumes/Storage/selected_green'):
    color_it('/Volumes/Storage/selected_green/'+f,'/Volumes/Storage/selected_green_cool/'+f)
