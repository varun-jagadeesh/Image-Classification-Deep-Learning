#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import libaries
import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# load data

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
print(train_set.shape)
print(test_set.shape)


# In[6]:


test_set.head()


# In[7]:


train_set.head()


# In[8]:


TRAIN_PATH = 'train_img/'
TEST_PATH = 'test_img/'


# In[9]:


# function to read image
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (299,299))
    #print(img)
    return img


# In[10]:


# load data
train_img, test_img = [],[]
for img_path in tqdm(train_set['image_id'].values):
    train_img.append(read_img(TRAIN_PATH + img_path + '.png'))

for img_path in tqdm(test_set['image_id'].values):
    test_img.append(read_img(TEST_PATH + img_path + '.png'))


# In[12]:


# normalize images
x_train = np.array(train_img, np.float32) / 255.
x_test = np.array(test_img, np.float32) / 255.


# In[ ]:





# In[13]:


# target variable - encoding numeric value
label_list = train_set['label'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]  
y_train = np.array(y_train)
y_train = to_categorical(y_train)


# In[14]:


#Transfer learning with Inception V3
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
## set model2 architechture
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),metrics=['accuracy'])
batch_size = 28 # tune it
epochs = 12 # increase it


# In[15]:


base_model.summary()


# In[ ]:





# In[ ]:





# In[13]:


#Transfer learning with Inception V3
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
## set model2 architechture
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])


batch_size = 30 # tune it
epochs = 20 # increase it

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train)


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('Inception-transferlearning.model', monitor='val_acc', save_best_only=True)]
)


# In[ ]:


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('Inception-transferlearning.model', monitor='val_acc', save_best_only=True)]
)


# In[39]:


y_train.shape[1]


# In[31]:


y_test.shape


# In[49]:


#Transfer learning with Inception V3
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
## set model3 architechture
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])


batch_size = 28 # tune it
epochs = 12 # increase it

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train2)


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data = (x_test,y_test),
    callbacks=[ModelCheckpoint('Inception-transferlearning.model', monitor='val_acc', save_best_only=True)]
)


# In[ ]:





# In[33]:


label_list = sub['label'].tolist()
y_test = {k:v+1 for v,k in enumerate(set(label_list3))}
y_test = [y_test[k] for k in label_list3]  
y_test = np.array(y_test)
y_test = to_categorical(y_test)


# In[34]:


y_test.shape


# In[18]:


#object = pd.read_pickle(r'model_object_detection_history_3.sav')


# In[ ]:





# In[21]:


#import pickle
#filename = 'model_object_detection_3.sav'
#pickle.dump(model, open(filename, 'wb'))


# In[22]:


#filename = 'model_object_detection_history_3.sav'
#pickle.dump(history, open(filename, 'wb'))


# In[26]:


#predict test data
predictions = model.predict(x_test)
# get labels
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in Y_train.items()}
pred_labels = [rev_y[k] for k in predictions]


# In[27]:



## make submission
sub = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels2})
sub.to_csv('sub.csv', index=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


# In[28]:


sub = pd.read_csv('sub.csv')


# In[29]:


y_test = sub['label']


# In[30]:


y_test


# In[ ]:





# In[25]:


# Loss curves
plt.figure(figsize=[8,4])
plt.plot(history2.history['loss'],'r',linewidth=3.0)
plt.plot(object.history['loss'],'b',linewidth=3.0)
plt.legend(['Testing loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()


# In[26]:


# Accuracy Curves
plt.figure(figsize=[8,4])
plt.plot(history2.history['accuracy'],'r',linewidth=3.0)
plt.plot(object.history['accuracy'],'b',linewidth=3.0)
plt.legend(['Testing Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()

