import cv2
import numpy as np


def find_marker(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	gray = cv2.GaussianBlur(gray,(11,11),0)
	edged = cv2.Canny(gray,40,60)


	(_,cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts,key = cv2.contourArea)

	return  cv2.minAreaRect(c)

def distance_cal(set_wid,focalL,wid):
	return float((set_wid*focalL)/wid)

set_distance = 20
set_wid =3.5


cap=cv2.VideoCapture(0)

ret1,f_frame = cap.read()
focal_frame = cv2.resize(f_frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
f_marker = find_marker(focal_frame)
focalL = (f_marker[1][0]*set_distance)/set_wid


while 1:
	ret,o_frame = cap.read()
	frame = cv2.resize(o_frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
	marker = find_marker(frame)
	distance = distance_cal(set_wid,focalL,marker[2][0])

	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(frame,[box],-1,(0,255,0),2)
	cv2.putText(frame,'%.1fcm'%distance,(frame.shape[1]-200,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
	cv2.imshow('image',frame)

	if cv2.waitKey(1)&0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()




