from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pygame


def sound():
	pygame.init()
	pygame.mixer.init()
	pygame.mixer.music.load("alarm.mp3")
	pygame.mixer.music.play()
	time.sleep(5)

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5]) #for vertical
	B = dist.euclidean(eye[2], eye[4]) # points
	C = dist.euclidean(eye[0], eye[3]) #for horizontal points
	ear = (A + B) / (2.0 * C)          #eye aspect ratio i.e EAR
	return ear
 
 

EYE_AR_THRESH = 0.24		#ear threshold
EYE_AR_CONSEC_FRAMES = 10	#for how many frames ear should be less tha threshold for alers

TOTAL = 0

COUNTER = 0	# initialize the frame counter as well as a boolean used to
ALARM_ON = False	# indicate if the alarm is going off



print("Loading facial landmark predicotr")

detector = dlib.get_frontal_face_detector() #dlib's face detector
#predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #facial land mark detector
predictor = dlib.shape_predictor('optimal_eye_predictor.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("Lstart: ", lStart)
print(lEnd)
#alert = cv2.imread('alert.jpg') # image for alert instead of sound

# start the video stream thread
print("Starting video stream thread")
vs = VideoStream(src=1).start()
time.sleep(1.0)


# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		startTime = time.time()
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		stopTime = time.time()
		print("Total time: ", stopTime - startTime)
		
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		#leftEye = shape[lStart:lEnd]
		#rightEye = shape[rStart:rEnd]
		leftEye = shape[0:6]
		rightEye = shape[6:12]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0		
		if leftEAR > 0.3:
			leftEAR = 0.3
		if rightEAR > 0.3:
			rightEAR = 0.3
		if leftEAR < 0.2:
			leftEAR = 0.2
		if rightEAR < 0.2:
			rightEAR = 0.2
		lep = ((leftEAR - 0.2)*(100))/(0.3-0.2)
		rep = ((rightEAR - 0.2)*(100))/(0.3-0.2)


		# average the eye aspect ratio together for both eyes
		

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		#test = cv2.convexHull(shape)
		#cv2.drawContours(frame, [test], -1, (0,255,0), 1)
		
		#_, contours, _ = cv2.findContours(leftEye,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					
					t = Thread(target=sound)
					t.deamon = True
					t.start()

				# draw an alarm on the frame
				
				#cv2.rectangle(alert,(0,0),(1350,750),(0,0,255),-1)
				#cv2.putText(alert, "ALERT!!!", (250,400), cv2.FONT_HERSHEY_SIMPLEX, 7, (0,0,0), 20)


		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			if COUNTER >= 2:
				TOTAL = TOTAL + 1			
			COUNTER = 0
			ALARM_ON = False
			#cv2.rectangle(alert,(0,0),(1350,750),(0,0,0),-1)
			#cv2.putText(alert, "ALERT!!!", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20)

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		#a = (x+w)/2
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (x + 20, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255 ), 1)
		cv2.putText(frame, "BPM: {}".format(TOTAL), (x + w - 120, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255 ), 1)
		#cv2.putText(frame, "LEAR: {:.2f}".format(leftEAR), (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255 ), 1)
		#cv2.putText(frame, "REAR: {:.2f}".format(rightEAR), (30, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255 ), 1)
		cv2.putText(frame, "{}".format(int(lep)), (x - 60, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		cv2.putText(frame, "{}".format(int(rep)), (x + w , y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		#cv2.putText(alert, "EAR: {:.2f}".format(ear), (1100, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0 ), 2)
 	
	# show the frame
	cv2.imshow("Frame", frame)
	#cv2.imshow('alert',alert)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
