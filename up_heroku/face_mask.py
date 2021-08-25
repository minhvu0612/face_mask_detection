# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time
import cv2
import os
import streamlit as st
import sys
from PIL import Image
import imutils
import tempfile as te
from datetime import datetime

# Lớp Thời Gian
class RealTime(object):
	def __init__(self):
		self.now = datetime.now()
	def getTime(self):
		time = self.now.strftime("%d/%m/%Y-%H:%M:%S")
		return time

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds, faces)

# load our serialized face detector model from disk
try:
	prototxtPath = r"face_detector\deploy.prototxt"
	weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
except:
	st.write("Can not load model!")
	sys.exit()


faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# face mask for image

def faceMaskImage(image, faceNet, maskNet):
	image = np.array(image.convert("RGB"))
	image = cv2.resize(image, (400,300))
	label_data = []
	faces = 0
	try:
		(locs, preds, faces) = detect_and_predict_mask(image, faceNet, maskNet)
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			if mask > withoutMask:
				label_data.append("Mask")
				label = "Mask"
			else:
				label_data.append("No Mask")
				label = "No Mask"
			if label == "Mask":
				color = (0,255,0)
			else:
				color = (255,0,0)
			labels = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.putText(image, labels, (startX, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	except:
		return "Error!"
	return image, label_data

def checkCount(label_data):
	countMask = 0
	countNoMask = 0
	for i in label_data:
		if i == "Mask":
			countMask += 1
		else:
			countNoMask += 1
	return countMask, countNoMask

# face mask for video
def setVideo():
	file_video = st.file_uploader("Upload Video", type = ['mp4'])
	if file_video is not None and st.button("Run"):
		video = te.NamedTemporaryFile(delete = False)
		video.write(file_video.read())
		vs = cv2.VideoCapture(video.name)
		while True:
			ret, frame = vs.read()
			if not ret:
				st.write("Can not load video!")
				break
			frame = cv2.resize(frame, (800,600))
			(locs, preds, faces) = detect_and_predict_mask(frame, faceNet, maskNet)
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				cv2.putText(frame, label, (startX, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				cv2.imshow("Frame", frame)
			key = cv2.waitKey(2) & 0xFF
			if key == ord("q"):
				vs.release()
				cv2.destroyAllWindows()
				break
# face mask for webcam

def setFaceMask():
	fps = imutils.video.FPS().start()
	vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	while True:
		ret, frame = vs.read()
		count = []
		if not ret:
			st.write("Not connection!")
			break
		frame = cv2.resize(frame, (800, 600))
		(locs, preds, faces) = detect_and_predict_mask(frame, faceNet, maskNet)
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			label = "Mask" if mask > withoutMask else "No Mask"
			count.append(label)
			time = RealTime()
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.putText(frame, label, (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
			cv2.putText(frame, "face: " + str(len(faces)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1)
			cv2.putText(frame, time.getTime(), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 1)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		countMask, countNoMask = checkCount(count)
		cv2.putText(frame, "with_mask: " + str(countMask), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
		cv2.putText(frame, "without_mask: " + str(countNoMask), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(10) & 0xFF
		fps.update()
		if key == ord('q'):
			vs.release()
			cv2.destroyAllWindows()
			break

def faceMaskWebCam():
	if st.button("Run"):
		setFaceMask()


# main function

def main():
	st.title("Face Mask Detection App")
	st.write("**Using FaceNet And MobileNet_V2**")
	active = ["Image", "WebCam", "Video", "About"]
	choose = st.sidebar.selectbox("Menu", active)
	if choose == "Image":
		file_image = st.file_uploader("Choose File", type=['jpeg', 'png', 'jpg', 'webp'])
		if file_image is not None and st.button("Run"):
			image = Image.open(file_image)
			result = faceMaskImage(image, faceNet, maskNet)
			if result != "Error!":
				st.image(result[0], use_column_width = True)
				with_mask, without_mask = checkCount(result[1])
				st.success("with_mask: {}".format(with_mask))
				st.error("without_mask: {}".format(without_mask))
			else:
				st.error(result)
	elif choose == "WebCam":
		faceMaskWebCam()
	elif choose == "Video":
		setVideo()
	else:
		st.write("**Author: Vũ Ngọc Minh**")
		st.write("Tech: Python, OpenCV, FaceNet, Tensorflow, Streamlit")
		st.image("author.jpg")
		st.write("Github: https://github.com/minhvu0612")
		st.write("Facebook: https://m.facebook.com/nm.vu.3")

main()