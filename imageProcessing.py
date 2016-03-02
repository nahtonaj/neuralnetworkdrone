# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2

# def rotate(angle, src, dst):
#     img = src
#     rows,cols = img.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
#     dst = cv2.warpAffine(img,M,(cols,rows))
class ImProcess(object):
	global img1, img2, grayimg1, grayimg2, cornerMat1, cornerMat2, temp1
	def __init__(self, file1, file2):
		self.img1 = cv2.imread(file1);
		self.img2 = cv2.imread(file2);
		self.grayimg1 = self.initiateimage(self.img1)
		self.grayimg2 = self.initiateimage(self.img2)
	def initiateimage(self, image):
		img1res = cv2.resize(image, None, fx=.2, fy=.2, interpolation = cv2.INTER_AREA) 
		gray = cv2.cvtColor(img1res, cv2.COLOR_BGR2GRAY)
		self.temp1 = gray
		# gray1 = np.float32(gray)
		return gray
	def corner(self, img):
		fast = cv2.FastFeatureDetector_create()
		dst = fast.detect(img,None)
		dst2 = cv2.drawKeypoints(img, dst, flags=0,color=(0,0,255))
		return dst1
	def main(self):
		cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg1.JPG', self.grayimg1)
		cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg2.JPG', self.grayimg2)
		# self.cornerMat1 = self.corner(self.grayimg1)
		# self.cornerMat2 = self.corner(self.grayimg2)
		temp1 = self.corner(self.grayimg1)
		# self.grayimg1[self.cornerMat1>0.01*self.cornerMat1.max()]=[50]
		# self.grayimg2[self.cornerMat2>0.01*self.cornerMat2.max()]=[50]
		# cv2.imshow('image1', self.temp1)
		# cv2.imshow('image2', self.i)

class Main(object):
	file1 = '/home/jonathan/Documents/ImageProcessingImages/img1.JPG';
	file2 = '/home/jonathan/Documents/ImageProcessingImages/img2.JPG';
	Thread1 = ImProcess(file1, file2)
	Thread1.main()



cv2.waitKey(0);
cv2.destroyAllWindows();




# <codecell>


