#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 04:43:42 2019

@author: Vikki
"""
import sys
import os
import cv2
import tensorflow as tf
import numpy as np
import six.moves.urllib as urllib
import tarfile


from PIL import Image
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
sys.path.append("..")

from object_detection.utils import ops as utils_ops
 
from utils import label_map_util
 
from utils import visualization_utils as vis_util

#COCO trained Model to download from tensorflow
M_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
M_FILE = M_NAME + '.tar.gz'
D_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
PATH_TO_CKPT = M_NAME + '/frozen_inference_graph.pb'
 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NO_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(D_BASE + M_FILE, M_FILE)
tar_file = tarfile.open(M_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NO_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph = detection_graph) as sess:
    while True:
    ret, image_ny = capture.read()
  # Expand dimensions  the model expects images to have shape: [1, 0, 0, 3]
  image_ny_expanded = np.expand_dims(image_ny, axis=0)
  image_tensorflow = detection_graph.get_tensor_by_name('image_tensor:0')
  # detect object into the boxes which is predicts
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  # Detection scores represents the score od prediction each every objects
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  # Detection classes represents the detection image with image catagory label.
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  no_detections = detection_graph.get_tensor_by_name('No.of Detections:0')
  # Actual detection.
  (detection_boxes, detection_scores, detection_classes,no_detections) = sess.run([detection_boxes,detection_scores,detection_classes,no_detections], feed_dict = {image_tensorflow: image_ny_expanded})
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_ny,
      ny.squeeze('detection_boxes'),
      ny.squeeze('detection_classes').astype(ny.int32),
      ny.squeeze('detection_scores'),
      category_index,
      use_normalized_coordinates= True,
      line_thickness=8)
  cv2.imshow('Real-Time Object Detection', cv2.resize(image_ny, (800,600)))
  if cv2.waitkey(25) 0xFF == ord('f'):
      cv2.destroyAllWindows()
      break


