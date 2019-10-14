# Face-recognition-using-tensorflow
This repository contains program for face recognition using transfer learning.

This program captures your face using webcam and comapre it with database images and results shows the distance between yourface and database images, if distance is < threshold, it shows welcome message.

# FaceNet
FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

# Triplet Loss
Triplet loss function is used which compares three inputs (i.e. Anchor, positive, negative) and tries to bring encoding of two image of same person closer and pulling encoding of different person far away.

# How to run program
Install all required libraries.

To prepare database, create images folder in your working directory and put your images with your name, so that it can identify your name and print welcome #yourname.

Then run B_face_rec.py file.

After executing above file, it will show distance of test image (i.e. Webcam capture Image) and training images (i.e. Database images).
