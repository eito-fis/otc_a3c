import os
import argparse
import pickle

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

#--GLOBAL VARIABLES--#
IMAGE_SIZE = 84
THRESHOLD = .5
NOISE_WEIGHT = .5
#--------------------#

def save_images_from_dict(dict_filepath, output_dir):
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  with open(dict_filepath, 'rb') as dict_file:
    slam_dict = pickle.load(dict_file)
    images = slam_dict['obs']
    sample_images = np.array([cv2.normalize(image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for image in images])
    for i, img in enumerate(sample_images):
      im = Image.fromarray(img)
      output_path = os.path.join(output_dir, "SLAM_image_{}.jpeg".format(i+1))
      im.save("SLAM_image_{}.jpeg".format(i+1))

def load_images_from_dir(image_dir):
  images = []
  image_filenames = os.listdir(image_dir)
  for image_filename in image_filenames:
    image_path = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_path)
    image = cv2.normalize(image, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    images.append(image)
  return np.array(images)

def threshold_images(images, output_dir):
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  thresh_imgs = np.array([cv2.threshold(image,128,255,cv2.THRESH_TOZERO)[1] for image in images])
  for i, img in enumerate(thresh_imgs):
    im = Image.fromarray(img)
    output_path = os.path.join(output_dir, "thresh_img_{}.jpeg".format(i+1))
    im.save(output_path)
    input()

def generate_dataset(source_images, noise_images):
  train_images = []
  label_images = []
  for i, source_image in enumerate(source_images):
    for j, noise_image in enumerate(noise_images):
      new_image = np.copy(source_image)
      for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
          if sum(noise_image[x,y]) > THRESHOLD:
            new_image[x,y] = list(np.multiply(NOISE_WEIGHT,noise_image[x,y]) + np.multiply((1-NOISE_WEIGHT),new_image[x,y]))
      plt.imshow(new_image)
      plt.show()
      train_images.append(new_image)
      label_images.append(source_image)
  return train_images, label_images
  

if __name__ == '__main__':
  #COMMAND-LINE ARGUMENTS#
  parser = argparse.ArgumentParser('OTC - Image Preprocessor')
  parser.add_argument('--preprocess', type=str, default=None, required=True)
  parser.add_argument('--output-dir', type=str, default=None)
  parser.add_argument('--source-dir', type=str, default=None)
  parser.add_argument('--threshold-dir', type=str, default=None)
  parser.add_argument('--dict-filepath', type=str, default=None)
  args = parser.parse_args()

  #SAVE IMAGES FROM DICTIONARY#
  if args.preprocess == 'load_dict' and args.dict_filepath:
    save_images_from_dict(args.dict_filepath, args.output_dir)
    exit()

  #THRESHOLD IMAGES#
  if args.preprocess == 'threshold' and args.source_dir and args.output_dir:
    source_images = load_images_from_dir(args.source_dir)
    threshold_images(source_images, args.output_dir)
    exit()

  #SUPERIMPOSE IMAGES#
  if args.preprocess == 'superimpose' and args.source_dir and args.threshold_dir:
    print("Loading source images...")
    source_images = load_images_from_dir(args.source_dir)
    print("Loading noise images...")
    threshold_images = load_images_from_dir(args.threshold_dir)
    print("Superimposing images...")
    train_images, label_images = generate_dataset(source_images, threshold_images)