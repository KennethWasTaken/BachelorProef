# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import pickle as pickle
import os

# detector = A pre-trained Caffe DL model to detect where in the image the faces are
# embedder = A pre-trained Torch DL model to calculate our 128-D face embeddings
# recognizer = self trained linear SVM face recognition model

# path to input images, images that need to be recognised.
from imutils import paths

path_images = r"./images/"

# The path to OpenCV’s deep learning face detector.
path_detector = r"./face_detection_model"

# The path to OpenCV’s deep learning face embedding model. We’ll use this model to extract the 128-D face embedding
# from the face ROI
path_openface = r"./openface_nn4.small2.v1.t7"

# The path to the recognizer SVM model
path_recognizer = r"./output/recognizer.pickle"

# The path to the label encoder. label = name of owner pic
path_labeler = r"./output/le.pickle"

# threshold to filter weak face detections
thresh = 0.6

# face detector
print("face detector")
proto_path = "./face_detection_model/deploy.prototxt"
model_path = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# serialized face embedding model
print("face recognizer")
embedder = cv2.dnn.readNetFromTorch(path_openface)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(path_recognizer, "rb").read())
le = pickle.loads(open(path_labeler, "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then get the image dimensions
imagePaths = list(paths.list_images(path_images))

for img in imagePaths:
	image = cv2.imread(img)
	cv2.imshow("test pic", image)
	cv2.waitkey(5000)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# make blob from image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to detect faces in image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# get confidence of prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.6:
			# calculate coordinates of bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI based on coordinates
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height large enough
			# if fW < 20 or fH < 20:
			# 	continue

			# make blob for face ROI, pass blob through face embedding model to get the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# classification to "recognize" face aka label it right
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the associated
			# probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)