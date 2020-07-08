import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
print("hello")
cap=cv2.VideoCapture(0)
success,img=cap.read()
while success and cv2.waitKey(1)!=27:
	net = cv2.dnn.readNet("C:\\Users\\HP\\Desktop\\project ocr\\frozen_east_text_detection.pb")
	orig = img
	(H, W) = img.shape[:2]

	(newW, newH) = (640, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	img = cv2.resize(img, (newW, newH))

	kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
								 [-1, 2, 2, 2, -1],
								 [-1, 2, 8, 2, -1],
								 [-1, 2, 2, 2, -1],
								 [-1, -1, -1, -1, -1]]) / 8.0
	img = cv2.filter2D(img, -1, kernel_sharpen_3)
	img = cv2.fastNlMeansDenoisingColored(img, 13, 13, 24, 21)
	mean = []
	for k in range(3):
		sum = 0
		for i in range(newH):
			for j in range(newW):
				sum = sum + img[i][j][k]
		sum = float(sum / (newW * newH))
		mean.append(sum)
	(H, W) = img.shape[:2]
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	H = 320
	W = 640
	blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (mean[2], mean[1], mean[0]), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input img
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	for (startX, startY, endX, endY) in boxes:
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the img
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		sub_img = orig[startY:endY, startX:endX]
		cv2.imshow("text detection",orig)
cap.release()
cap.destroyAllWindows()

    
	

	