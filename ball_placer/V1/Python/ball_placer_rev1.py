from imutils.video import VideoStream
import numpy as np 
import cv2
import imutils
import time

cap = cv2.VideoCapture(0)
# time.sleep(2.0)

whiteLower = (0,0,50)
whiteUpper = (150,200,255)

blueLower = (80,100,20)
blueUpper = (140,255,200)

while(True):
	ret, frame = cap.read()

	frame = imutils.resize(frame,width=600)
	blurred = cv2.GaussianBlur(frame, (11,11),0)
	hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

	maskWhite = cv2.inRange(hsv, whiteLower, whiteUpper)
	maskWhite = cv2.erode(maskWhite, None, iterations=2)
	maskWhite = cv2.dilate(maskWhite, None, iterations=2)

	maskBlue = cv2.inRange(hsv, blueLower, blueUpper)
	maskBlue = cv2.erode(maskBlue, None, iterations=2)
	maskBlue = cv2.dilate(maskBlue, None, iterations=2)

	contours, _ = cv2.findContours(maskWhite.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(contours)
	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
		if area > 20000:
			x = approx.ravel()[0]
			y = approx.ravel()[1]
			if len(approx) == 4:
				cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

	contours, _ = cv2.findContours(maskBlue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(contours)
	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
		if area > 4000:
			if len(approx) > 7:
				cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)


	cv2.imshow("Frame", frame)
	cv2.imshow("White Mask", maskWhite)
	cv2.imshow("Blue Mask", maskBlue)

	cv2.imshow('hsv',hsv)
	if cv2.waitKey(20) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()