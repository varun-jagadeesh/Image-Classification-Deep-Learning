#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob

#sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[2]:


#train_dicom_dir = os.path.join(DATA_DIR, 'train_images')
#test_dicom_dir = os.path.join(DATA_DIR, 'test_images')


# In[5]:


# load data
train_dicom_dir
test_dicom_dir


# In[6]:


COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"


# In[7]:


class autoConfig(Config):
    NAME = "auto"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 3  
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  
    TRAIN_ROIS_PER_IMAGE = 8
    STEPS_PER_EPOCH = 10
    
config = autoConfig()
config.display()


# In[8]:


class autoDataset(utils.Dataset):
    def load_auto(self, count, height, width):
        # Add classes
        self.add_class("auto", 1, "car")
        self.add_class("auto", 2, "person")
        self.add_class("auto", 3, "traffic")

        # Add images
        for i in range(count):
            bg_color, auto = self.random_image(height, width)
            self.add_image("auto", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, auto=auto)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['auto']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "auto":
            return info["auto"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        auto = info['auto']
        count = len(auto)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['auto']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in auto])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'car':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "person":
            cv2.person(image, (x, y), s, color, -1)
        elif shape == "traffic":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        # auto
        shape = random.choice(["car", "person", "traffic"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random auto and record their
        # bounding boxes
        auto = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            auto.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        auto = [s for i, s in enumerate(auto) if i in keep_ixs]
        return bg_color, auto


# In[11]:


dataset_train = autoDataset()
dataset_train.load_auto(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()
dataset_val = autoDataset()
dataset_val.load_auto(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# In[10]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
# Save weights
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)
history = model.keras_model.history.history


# In[14]:


class InferenceConfig(autoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
model_path = model.find_last()
#print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
history = model.keras_model.history.history


# In[17]:


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# In[20]:


results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


# In[22]:


plt.figure(figsize=(8,4))
plt.subplot(131)
plt.plot(epochs, history["loss"], label="Test loss")
plt.legend()
plt.show()


# In[23]:
plt.figure(figsize=(8,4))
plt.subplot(131)
plt.plot(epochs, history["accuracy"], label="Test accuracy")
plt.legend()
plt.show()






