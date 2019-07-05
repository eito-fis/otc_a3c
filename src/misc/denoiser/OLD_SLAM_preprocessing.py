import os
import argparse
import pickle

import numpy as np

import cv2
from PIL import Image

#--GLOBAL VARIABLES--#
IMAGE_SIZE = 168
THRESHOLD = 128
#--------------------#

def save_images_from_dict(dict_filepath, output_dir):
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  with open(dict_filepath, 'rb') as dict_file:
    slam_dict = pickle.load(dict_file)
    images = slam_dict['obs']
    sample_images = np.array([cv2.normalize(image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for image in images])
    for i, img in enumerate(sample_images):
      im = Image.fromarray(img)
      im.save("SLAM_images/SLAM_image_{}.jpeg".format(i+1))

def load_images_from_dir(image_dir):
  images = []
  image_filenames = os.listdir(image_dir)
  for image_filename in image_filenames:
    image_path = os.path.join(image_dir, image_filename)
    images.append(Image.open(image_path))
  return images

def threshold_images(images, output_dir):
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  thresh_imgs = np.array([cv2.threshold(image,THRESHOLD,255,cv2.THRESH_TOZERO)[1] for image in images])
  for i, img in enumerate(thresh_imgs):
    im = Image.fromarray(img)
    output_path = os.path.join(output_dir, "thresh_img_{}.jpeg".format(i+1))
    im.save(output_path)

def superimpose_images(source_images, noise_images, output_dir):
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  for i, source_image in enumerate(source_images):
    for j, noise_image in enumerate(noise_images):
      new_image = source_image.copy()
      new_img = new_image.load()
      noise_img = noise_image.load()
      for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
          if sum(noise_img[x,y]) > THRESHOLD:
            new_img[x,y] = tuple(map(int,np.multiply(.5,noise_img[x,y]) + np.multiply(.5,new_img[x,y])))
      output_path = os.path.join(output_dir, "superimposed_image_{}_{}.jpeg".format(i+1,j+1))
      new_image.save(output_path)

if __name__ == '__main__':
  #COMMAND-LINE ARGUMENTS#
  parser = argparse.ArgumentParser('OTC - Image Preprocessor')
  parser.add_argument('--preprocess', type=str, default=None, required=True)
  parser.add_argument('--output-dir', type=str, default=None, required=True)
  parser.add_argument('--source-dir', type=str, default=None)
  parser.add_argument('--threshold-dir', type=str, default=None)
  parser.add_argument('--dict-filepath', type=str, default=None)
  args = parser.parse_args()

  #SAVE IMAGES FROM DICTIONARY#
  if args.preprocess == 'load_dict' and args.dict_filepath:
    save_images_from_dict(args.dict_filepath, args.output_dir)
    exit()

  #THRESHOLD IMAGES#
  if args.preprocess == 'threshold' and args.output_dir:
    threshold_images(sample_images, args.output_dir)
    exit()

  #SUPERIMPOSE IMAGES#
  if args.preprocess == 'superimpose' and args.source_dir and args.threshold_dir:
    print("Loading source images...")
    source_images = load_images_from_dir(args.source_dir)
    print("Loading noise images...")
    threshold_images = load_images_from_dir(args.threshold_dir)
    print("Superimposing images...")
    superimpose_images(source_images, threshold_images, args.superimpose_dir)
    exit()