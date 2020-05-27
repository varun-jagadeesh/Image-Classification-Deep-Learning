#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install opencv-python


# In[2]:


# !pip install pillow


# In[3]:


import cv2


# In[4]:


import numpy as np
import time


# In[5]:


net = cv2.dnn.readNet("yolov3-tiny.weights", "yolo_realtime_detection_cpu/cfg/yolov3-tiny.cfg")
classes = []
with open("yolo_realtime_detection_cpu/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[6]:


# Loading camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


# In[7]:


# _, frame = cap.read()
# len(frame)


# In[8]:


while(cap.isOpened()):
# while True:
    _, frame = cap.read()
    frame_id += 1
#     print (frame_id)
    height, width, channels = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# from keras.models import load_model


# In[ ]:




