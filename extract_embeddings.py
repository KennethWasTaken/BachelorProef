import re

from imutils import paths
import numpy as np
import imutils
import pickle
import cv2

# path to input directory of faces + images
path_dataset = r"./dataset/"

# path to output serialized db of facial embeddings
path_output= r"./result/embeddings.pickle"

# path to OpenCV's deep learning face detector
path_detector = r"./face_detection_model"

#path to OpenCV's deep learning face embedding model
path_embeddingmodel = "./openface_nn4.small2.v1.t7"

# min probability to filter out weak detections
set_confidence = 0.5

# load face detector, Caffe based detector to localize faces in an image
proto_path = "./face_detection_model/deploy.prototxt"
model_path = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# load face extractor
# embedder extracts faces through deep learning
embedder = cv2.dnn.readNetFromTorch(path_embeddingmodel)

# paths to ID-cards
print("quantify faces")
imagePaths = list(paths.list_images(path_dataset))
# initialize our lists of extracted facial embeddings and
# corresponding people names (in the image names)
knownEmbeddings = []
knownNames = []

# total = the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("image {}/{}".format(i + 1, len(imagePaths)))
	a = imagePath.split("/")
	b = a[2].split(".")
	name = b[0]

	# load the image, resize it to have a width of 600 pixels +  image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image, blobs can store binairy data
	# reshape to 300,300 for detector
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image by passing blobbed input to detector
	detector.setInput(imageBlob)
	# Blob is an iterable object-type so forward = next blob
	detections = detector.forward()

	# Processing the detected faces:
	# ensure at least one face was found
	if len(detections) > 0:
		# assuming every image has only 1 face, find bounding box with the largest probability (confidence)
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# confidence must be higher then given confidence
		if confidence > set_confidence:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# box = [471..., 227... 571... 355...]

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# face must be large enough
			# if fW > 20 or fH > 20:
			# 	continue
			# else:
			# 	print("face not large enough")


			# Maek blob from face ROI and use face embedding model to quantify face in 128 features
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			# vec = 128-D quantification of face
			vec = embedder.forward()

			# add the name of the person + corresponding face embedding to lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1 #faces detected counter

# dump the facial embeddings + names to disk
print("Number of faces detected " + str(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}

# "wb" = write and binary, binary to write pickle
f = open(path_output, "wb")

# Pickle module is used to convert any kind of python object into byte streams.
f.write(pickle.dumps(data))
f.close()