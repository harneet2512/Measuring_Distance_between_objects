import cv2
import numpy as np

class Helpers:
	def __init__(self):
		pass

	def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	    dim = None
	    (h, w) = image.shape[:2]
	    if width is None and height is None:
	        return image
	    if width is None:
	        r = height / float(h)
	        dim = (int(w * r), height)
	    else:
	        r = width / float(w)
	        dim = (width, int(h * r))
	    resized = cv2.resize(image, dim, interpolation=inter)

	    return resized

	def grab_contours(cnts):
		if len(cnts) == 2:
			cnts = cnts[0]
		elif len(cnts) == 3:
			cnts = cnts[1]
		else:
			raise Exception('The length of the contour must be 2 or 3.')
		return cnts

	def sort_contours(cnts, method="left-to-right"):
		# initialize the reverse flag and sort index
		reverse = False

		i = 0
		# handle if we need to sort in reverse
		if method == "right-to-left" or method == "bottom-to-top":
			reverse = True
		# handle if we are sorting against the y-coordinate rather than
		# the x-coordinate of the bounding box
		if method == "top-to-bottom" or method == "bottom-to-top":
			i = 1
		# construct the list of bounding boxes and sort them from top to
		# bottom
		boundingBoxes = [cv2.boundingRect(c) for c in cnts]
		(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
			key=lambda b:b[1][i], reverse=reverse))
		# return the list of sorted contours and bounding boxes
		return (cnts, boundingBoxes)

	def draw_contour(image, c, i):
		# compute the center of the contour area and draw a circle
		# representing the center
		M = cv2.moments(c)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		# draw the countour number on the image
		cv2.putText(image, "{}".format(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.85, (0, 0, 0), 2)
		# return the image with the contour number drawn on it
		return image

	def orders(pts):
		rect = np.zeros((4, 2), dtype = "float32")
		s = pts.sum(axis = 1)

		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]

		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]

		return rect

	def transform(image, pts):
		rect = Helpers.orders(pts)
		(tl, tr, br, bl) = rect

		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

		return warped