# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:10:14 2019

@author: Bhawna
"""

from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl
from keras.models import load_model


PADDING = 50
ready_to_detect_identity = True
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)   
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)  
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

load_weights_from_FaceNet(FRmodel)

def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database

def webcam_face_recognizer(database):
    
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i=0

    while i<2:
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(1000) 
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
        #print(i)
        i+=1
    
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        cv2.imwrite('example.png',img)

        ready_to_detect_identity = False
        welcome_users(identities)
        
    return img

def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc-encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
           # print(identity)
    
    if min_dist > 0.5:
        print("No Match Found")
        return None
    else:
        print("Match Found")
        return str(identity)
    

def welcome_users(identities):
   
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
        print(welcome_message)
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'
        print(welcome_message)



    # Allow the program to start detecting identities again
    ready_to_detect_identity = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)


