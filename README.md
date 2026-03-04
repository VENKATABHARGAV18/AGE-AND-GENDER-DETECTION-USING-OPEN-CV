# AGE-AND-GENDER-DETECTION-USING-OPEN-CV #


### OVERVIEW

This project implements an Age and Gender Detection system using Deep Learning and Computer Vision. The system detects human faces in an image and predicts the gender and approximate age group of each detected face.

The project uses OpenCV's Deep Neural Network (DNN) module along with pre-trained Caffe models to perform face detection and classification tasks. Once a face is detected, the system extracts the face region and feeds it into deep learning models that predict the person's gender and age group.

This project demonstrates how deep learning models can be integrated with computer vision techniques to perform demographic analysis from images.

### OBJECTIVES

The main objectives of this project are:

* Detect human faces in an image using deep learning

* Predict the gender of the detected person

* Estimate the age group of the detected person

* Display prediction results on the image

* Demonstrate the use of OpenCV DNN with pre-trained models

### TECHNOLOGIES USED

* Python

* OpenCV

* Deep Learning (DNN)

* Caffe Pre-trained Models

* NumPy

### HARDWARE REQUIREMENTS

* Computer or Laptop

* Webcam or image input (optional)

### SOFTWARE REQUIREMENTS

* Python 3.x

* OpenCV

* NumPy

INSTALL REQUIRED LIBRARIES :

* pip install opencv-python numpy

### PROJECT FILES

The project consists of the following files:

Age-Gender-Detection
│

├── main.py

├── age_deploy.prototxt

├── age_net.caffemodel

├── gender_deploy.prototxt

├── gender_net.caffemodel

├── opencv_face_detector.pbtxt

├── opencv_face_detector_uint8.pb

└── image.jpg


### FILE DESCRIPTION

* main.py
Main program that performs face detection and age-gender prediction.

* opencv_face_detector.pbtxt
Configuration file for the face detection model.

* opencv_face_detector_uint8.pb
Pre-trained face detection model.

* age_deploy.prototxt
Model architecture file for age prediction.

* age_net.caffemodel
Pre-trained deep learning model used to predict age groups.

* gender_deploy.prototxt
Model architecture file for gender prediction.

* gender_net.caffemodel
Pre-trained deep learning model used to predict gender.

* image.jpg
Input image used for testing the system.

### HOW THE SYSTEM WORKS

The system follows these steps:

* Load the input image.

* Detect faces using a pre-trained face detection model.

* Extract the detected face region.

* Convert the face image into a blob for deep learning processing.

* Use the gender prediction model to determine gender.

* Use the age prediction model to estimate age group.

* Display the predicted age and gender on the image.

### AGE CATEGORIES

The model predicts age within the following ranges:

(0–2)

(4–6)

(8–12)

(15–20)

(25–32)

(38–43)

(48–53)

(60–100)

OUTPUT

The system displays the following information on the detected face:

Gender (Male / Female)

Age Group

Confidence percentage

Example output:

Male, (25-32)
Female, (15-20)

### APPLICATIONS

Age and gender detection systems can be used in several real-world applications:

* Smart surveillance systems

* Customer demographic analysis

* Retail marketing analytics

* Human-computer interaction systems

* Security and monitoring systems

### FUTURE IMPROVEMENTS

Several improvements can be made to enhance the project:

* Implement real-time webcam-based detection

* Improve accuracy using modern deep learning models

* Add emotion detection

* Deploy the system as a web application

* Integrate with IoT or edge devices

### CONCLUSION

This project demonstrates how computer vision and deep learning techniques can be used to detect faces and predict demographic attributes such as age and gender. By using OpenCV and pre-trained deep learning models, the system can analyze images and display predictions directly on the detected faces.

The project provides practical experience in working with deep neural networks, image processing, and machine learning models, and serves as a foundation for building more advanced computer vision applications.
