#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:09:10 2024

@author: machine
"""

from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sn; sn.set(font_scale=1.2)
import os 
import tensorflow as tf
import numpy as np
import cv2
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19


def PlotAccuracyLoss(history):
    #Plot Loss and Accuracy 
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history['accuracy'],'bo--', label = "acc")
    plt.plot(history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history['loss'],'bo--', label = "loss")
    plt.plot(history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


def DisplayRandomImage(class_names, images):
    #Function to show random Images
    data_iterator = images.as_numpy_iterator()
    img_and_label = data_iterator.next()
    img = img_and_label[0]
    label = img_and_label[1]
    index = np.random.randint(img.shape[0])
    plt.figure()
    plt.imshow(img[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[label[index]])
    plt.show()


def DisplayExamples(class_names,datapipeline):
    #Display Random examples
    data_iterator = datapipeline.as_numpy_iterator()
    img_and_label = data_iterator.next()
    img = img_and_label[0]
    label = img_and_label[1]
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[label[i]])
    plt.show()


def PrintMislabelledImages(class_names, test_datapipeline,model):
    #Print mislabeled Images
    data_iterator = test_datapipeline.as_numpy_iterator()
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of incorrect predictions of images", fontsize=16)
    j=0
    for img_and_label in data_iterator:
        img  = img_and_label[0]
        labels = img_and_label[1]
        predictions = model.predict(img)
        pred_labels = np.argmax(predictions, axis = 1)
        
        BOO = (pred_labels == labels)
        mislabeled_indices = np.where(BOO == 0)
        mislabeled_images = img[mislabeled_indices]
        mislabeled_labels = pred_labels[mislabeled_indices]
        index = np.random.randint(mislabeled_images.shape[0])
        plt.subplot(5,5,j+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(mislabeled_images[index], cmap=plt.cm.binary)
        plt.xlabel(class_names[mislabeled_labels[index]])
        j=j+1
        if j>=25:
            break
    plt.show()


def PrintConfusionMatrix(data_pipeline,model):
    #Print Confusion Matrix 
    predictions = model.predict(data_pipeline)
    pred_labels = np.argmax(predictions,axis=1)
    test_labels = np.concatenate([test_labels for x, test_labels in data_pipeline], axis=0)
    CM = confusion_matrix(test_labels, pred_labels)
    ax = plt.axes()
    sn.heatmap(CM, annot=True, 
           annot_kws={"size": 7}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
    ax.set_title('Confusion matrix')
    plt.show()

def PredictImage(classname, folder_path, model):
    #Predicts category of a given images
    for file in os.listdir(folder_path) :
        img_path = os.path.join(folder_path,file)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        fig = plt.figure(figsize=(10,10))
        plt.imshow(img, cmap=plt.cm.binary)
        
        
        img_pred = img/255
        img_pred = np.expand_dims(img, axis=0)
        
        y = model.predict(img_pred)
        
        y = np.argmax(y, axis = 1)
        category = classname[y[0]]
        
        plt.xlabel(category)
        plt.show()
    
    



IMG_SIZE = (224,224)

#Loading Data
data_train_dir = '/home/machine/Documents/project_files/CV/landscape_classifier/i_i_c/seg_train/seg_train'
data_test_dir ='/home/machine/Documents/project_files/CV/landscape_classifier/i_i_c/seg_test/seg_test'

dp_train = tf.keras.utils.image_dataset_from_directory(data_train_dir,labels='inferred',label_mode='int',class_names=None,
                                                       color_mode='rgb', batch_size=72, image_size=IMG_SIZE, shuffle=True, 
                                                       seed=1, validation_split=0.2, subset='training', interpolation='bilinear')

dp_validation = tf.keras.utils.image_dataset_from_directory(data_train_dir,labels='inferred',label_mode='int',class_names=None,
                                                       color_mode='rgb', batch_size=72, image_size=IMG_SIZE, shuffle=True, 
                                                       seed=1, validation_split=0.2, subset='validation', interpolation='bilinear')

dp_test = tf.keras.utils.image_dataset_from_directory(data_test_dir,labels='inferred',label_mode='int',class_names=None,
                                                       color_mode='rgb', batch_size=72, image_size=IMG_SIZE, shuffle=True, 
                                                       seed=None, validation_split=None, subset=None, interpolation='bilinear')

#Intializing Class Names and Labels
class_names = dp_train.class_names
class_names_labels = {class_name : i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

#Scaling data
dp_train=dp_train.map(lambda x,y:(x/255,y))
dp_test=dp_test.map(lambda x,y:(x/255,y))
dp_validation=dp_validation.map(lambda x,y:(x/255,y))

vgg16 = VGG16()
vgg16_layer_list = vgg16.layers
vgg16Model = Sequential()
for i in range(len(vgg16_layer_list)-1):
    vgg16Model.add(vgg16_layer_list[i])
for layers in vgg16Model.layers:
    layers.trainable = False

vgg16Model.add(Dense(6, activation = "softmax"))
vgg16Model.layers[-2].trainable = True
#vgg16Model.layers[-3].trainable = True
print(vgg16Model.summary(show_trainable=True))
vgg16Model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='vgg16callback/ModelCheckpoint',
                                                            save_weights_only=False,verbose=1),
                tf.keras.callbacks.BackupAndRestore(backup_dir='vgg16callback/BackupandRestore',save_freq="epoch"),
                tf.keras.callbacks.TensorBoard(log_dir='vgg16callback/logs'),
                tf.keras.callbacks.CSVLogger(filename='vgg16callback/CSVLogger/logger.csv')]

saved_model_file = Path("vgg16weights/checkpoint")
if saved_model_file.exists():
    vgg16Model.load_weights('vgg16weights/landscape_classifier_model1_weights')
    history_dict = json.load(open('history_vgg16model', 'r'))
    print("Model Loaded")
else:
    history_vgg16model = vgg16Model.fit(dp_train, validation_data=dp_validation,epochs=12,callbacks=my_callbacks)
    vgg16Model.save_weights('vgg16weights/landscape_classifier_model1_weights')
    history_dict = history_vgg16model.history
    json.dump(history_dict,open('history_vgg16model','w'))
    print("Model Saved")

PlotAccuracyLoss(history_dict)   
DisplayRandomImage(class_names, dp_train) 
PrintConfusionMatrix(dp_validation,vgg16Model)
PrintMislabelledImages(class_names,dp_validation,vgg16Model)

eval_result=vgg16Model.evaluate(dp_validation)
print("Evaluation result: ",eval_result )

folder_path = input("Enter path to prediction folder: ")
PredictImage(class_names,folder_path, vgg16Model)