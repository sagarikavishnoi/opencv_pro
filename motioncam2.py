import datetime
import imutils
import time
import cv2

cam= cv2.VideoCapture(0)
#time.sleep(0.25)
#fourcc=cv2.VideoWriter_fourcc(*'MJPG')
#out=cv2.VideoWriter('newoutput.avi',fourcc,20.0,(500,500))
 

Initial_frame = None
while True:
	(capture, frame) = cam.read()
	text = "No motion"
 
	# if the frame could not be capture, then we have reached the end
	# of the video
	if not capture:
		break
 
	# frame-resized,converted to grayscale, and then blurred
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if no front frame, initialize it
	if Initial_frame is None:
		Initial_frame = gray
		continue
	# absolute difference between the current frame and
	# first frame
	frame_difference = cv2.absdiff(Initial_frame, gray)
	thresh_difference = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)[1]
	thresh_difference = cv2.dilate(thresh_difference, None, iterations=2)
	_,counts,_ = cv2.findContours(thresh_difference.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
 
	# loop over the contours
	for c in counts:
		# if the contour is too small,not to be considered
		if cv2.contourArea(c) < 500:
			continue
 
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Motion Detected"
	# time and message on screen
	cv2.putText(frame,format(text), (10, 30),
		cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        #out.write(frame)
 
	#display on screen
	cv2.imshow("input", frame)
	cv2.imshow("output", thresh_difference)
	cv2.imshow("Frame", frame_difference)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, get out of the loop
	if key == ord("q"):
		break
 
cam.release()
cv2.destroyAllWindows()
