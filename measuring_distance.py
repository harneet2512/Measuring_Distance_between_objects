# import the necessary packages
from scipy.spatial import distance as dist
from Helpers import *
import numpy as np
import argparse
import cv2



image = cv2.imread('Image for Q5(i).png')
resize = Helpers.resize(image, width=800)
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 0, 70)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = Helpers.grab_contours(cnts)
(cnts, _) = Helpers.sort_contours(cnts)
refObj = None

def draw_circle(xy,r,color):
	cv2.circle(copy, (int(xy[0]), int(xy[1])) , r, color, -1)

def draw_line(a,b):
	cv2.line(copy, (int(a[0]),int(a[1])), (int(b[0]), int(b[1])), (103,224,94), thickness=2)

def distance(a,b):
	return int(dist.euclidean(a,b))

def mid_point(a,b):
	return ((a[0] + b[0]) / 2 , (a[1] + b[1]) / 2)

for c in cnts:
	if cv2.contourArea(c) < 100:
		continue

	box = cv2.minAreaRect(c)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = Helpers.orders(box)
	

	M = cv2.moments(c)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
	    cX, cY = 0, 0

	if refObj is None:
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = mid_point(tl, bl)
		(trbrX, trbrY) = mid_point(tr, br)
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		refObj = (box, (cX, cY), D / 0.955)
		continue

copy = resize.copy()
cv2.drawContours(copy, [box.astype("int")], -1, (227,129,109), 2)
cv2.drawContours(copy, [refObj[0].astype("int")], -1, (255,255,255), 2)

refCoords = np.vstack([refObj[0], refObj[1]])
objCoords = np.vstack([box, (cX, cY)])

for ((xA, yA), (xB, yB)) in zip(refCoords, objCoords):
		draw_circle((int(xA), int(yA)), 5, (0,0,0))
		draw_circle((int(xB), int(yB)), 5, (0,0,0))
		draw_line((int(xA), int(yA)), (int(xB), int(yB)))
		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
		(mX, mY) = mid_point((xA, yA), (xB, yB))
cv2.putText(copy, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), 
cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
cv2.imshow("image", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()