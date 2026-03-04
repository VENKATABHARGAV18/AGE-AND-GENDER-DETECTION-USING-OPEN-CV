import cv2
import numpy as np

def faceBox(faceNet, frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frameWidth, x2)
            y2 = min(frameHeight, y2)
            if x2 > x1 and y2 > y1:
                bboxs.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxs

import os

# determine directory where script resides so relative model paths are correct
base_dir = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")

ageProto = os.path.join(base_dir, "age_deploy.prototxt")
ageModel = os.path.join(base_dir, "age_net.caffemodel")

genderProto = os.path.join(base_dir, "gender_deploy.prototxt")
genderModel = os.path.join(base_dir, "gender_net.caffemodel")

# verify model files exist before loading
for path in (faceProto, faceModel, ageProto, ageModel, genderProto, genderModel):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required model file not found: {path}")

try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
except cv2.error as e:
    print("Failed to load one of the networks. Check that the model files are valid.")
    raise

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderlist = ['Male', 'Female']

import argparse

# --- input handling ------------------------------------------------------
parser = argparse.ArgumentParser(description="Age and gender detection on image or webcam")
parser.add_argument("source", nargs="?", default="camera",
                    help="Path to image file, or 'camera' (default) to use the webcam")
args = parser.parse_args()

use_camera = args.source.lower() == "camera"

if use_camera:
    print("Opening default webcam (device 0)... press 'q' to quit")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check that a camera is connected.")
else:
    image_path = args.source
    if not os.path.isabs(image_path):
        # interpret relative paths relative to script directory
        image_path = os.path.join(base_dir, image_path)
    print(f"Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

# --- frame processing helper --------------------------------------------
def process_frame(frame):
    frame = cv2.resize(frame, (640, 480))
    frame, bboxs = faceBox(faceNet, frame)
    if not bboxs:
        return frame  # nothing to draw

    for bbox in bboxs:
        padding = 20
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(frame.shape[1], bbox[2] + padding)
        y2 = min(frame.shape[0], bbox[3] + padding)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        genderProb = genderPred[0].max()
        gender = genderlist[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        ageProb = agePred[0].max()
        age = agelist[agePred[0].argmax()]

        label = "{}, {} (G:{:.2f}% A:{:.2f}%)".format(gender, age, genderProb*100, ageProb*100)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 35), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# --- execution -----------------------------------------------------------
if use_camera:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        output = process_frame(frame)
        cv2.imshow("Age-Gender Detection", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    output = process_frame(frame)
    cv2.imshow("Age-Gender Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
