from imutils.video import VideoStream
import numpy as np 
import cv2
import imutils
import time

cap = cv2.VideoCapture(0)
# time.sleep(2.0)

whiteLower = (0,0,50)
whiteUpper = (120,120,255)

blueLower = (80,100,20)
blueUpper = (140,255,200)

while(True):
	ret, frame = cap.read()

	frame = imutils.resize(frame,width=600)
	blurred = cv2.GaussianBlur(frame, (11,11),0)
	hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

	maskWhite = cv2.inRange(hsv, whiteLower, whiteUpper)
	maskWhite = cv2.erode(maskWhite, None, iterations=4)
	maskWhite = cv2.dilate(maskWhite, None, iterations=2)

	# maskBlue = cv2.inRange(hsv, blueLower, blueUpper)
	# maskBlue = cv2.erode(maskBlue, None, iterations=4)
	# maskBlue = cv2.dilate(maskBlue, None, iterations=2)
	# gray = cv2.cvtColor(maskBlue,cv2.COLOR_HSV2BGR)

	gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

	contours, _ = cv2.findContours(maskWhite.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(contours)
	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
		if area > 20000:
			if len(approx) == 4:
				x1 = approx.ravel()[0]
				y1 = approx.ravel()[1]
				x2 = approx.ravel()[2]
				y2 = approx.ravel()[3]
				x3 = approx.ravel()[4]
				y3 = approx.ravel()[5]
				x4 = approx.ravel()[6]
				y4 = approx.ravel()[7]
				cv2.circle(frame, (x1, y1), 10, (0, 0, 0), -1)
				cv2.circle(frame, (x2, y2), 10, (0, 0, 0), -1)
				cv2.circle(frame, (x3, y3), 10, (0, 0, 0), -1)
				cv2.circle(frame, (x4, y4), 10, (0, 0, 0), -1)
				# cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.9, 100)
	if circles is not None:
		circles = np.round(circles[0,:]).astype("int")

		for (x,y,r) in circles:
			# cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
			cv2.circle(frame, (x,y), 10, (0,255,0), -1)

	# contours, _ = cv2.findContours(maskBlue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# # cnts = imutils.grab_contours(contours)
	# for cnt in contours:
	# 	area = cv2.contourArea(cnt)
	# 	approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
	# 	if area > 4000:
	# 		if len(approx) > 7:
	# 			cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)


	cv2.imshow("Frame", frame)
	# cv2.imshow("White Mask", maskWhite)
	# cv2.imshow("Blue Mask", maskBlue)
	# cv2.imshow("gray", gray)
	# cv2.imshow('hsv',hsv)

	if cv2.waitKey(20) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()