# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import pybrain.tools.shortcuts as pb


# def rotate(angle, src, dst):
#     img = src
#     rows,cols = img.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
#     dst = cv2.warpAffine(img,M,(cols,rows))


class Corner(object):
    # global img1, img2, grayimg1, grayimg2, cornerMat1, cornerMat2, temp1
        # self.matchPoints = matchPoints
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # # self.img1 = file1  # cv2.imread(file1)
        # # self.img2 = file2  # cv2.imread(file2)
    def setImages(self, file1, file2):
        self.grayimg1 = self.initiateimage(file1)
        self.grayimg2 = self.initiateimage(file2)

    def initiateimage(self, image):
        if image is None:
            print 'Image is null!'
            quit()
        else:
            # imgres = cv2.resize(image, None, fx=.2, fy=.2, interpolation=cv2.INTER_AREA)
            imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # imgcont = cv2.equalizeHist(imggray)
            return imggray

    def FAST(self, img):
        fast = cv2.FastFeatureDetector_create()
        dst, des = fast.detectAndCompute(img, None)
        return cv2.drawKeypoints(img, dst, None, color=(255, 0, 0), flags=0)

    def ORB(self, img):
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        return orb.detectAndCompute(img, None)

    # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    # cv2.drawKeypoints(img,kp, None,color=(0,255,0), flags=0)
    def SURF(self, img):
        surf = cv2.xfeatures2d.SURF_create(400)
        return surf.detectAndCompute(img, None)

    def corner(self, img):
        return self.ORB(img)

    def match(self, img1, img2, draw):
        kp1, des1 = self.corner(img1)
        kp2, des2 = self.corner(img2)
        # if ver2 == 'FLANN':
        # 	# FLANN parameters
        # 	FLANN_INDEX_KDTREE = 0
        # 	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # 	search_params = dict(checks=50)  # or pass empty dictionary
        #
        # 	flann = cv2.FlannBasedMatcher(index_params, search_params)
        #
        # 	matches = flann.knnMatch(des1, des2, k=2)
        #
        # 	# Need to draw only good matches, so create a mask
        # 	matchesMask = [[0, 0] for i in xrange(len(matches))]
        #
        # 	# ratio test as per Lowe's paper
        # 	for i, (m, n) in enumerate(matches):
        # 		if m.distance < 0.7 * n.distance:
        # 			matchesMask[i] = [1, 0]
        #
        # 	draw_params = dict(matchColor=(0, 255, 0),
        # 					   singlePointColor=(255, 0, 0),
        # 					   matchesMask=matchesMask,
        # 					   flags=0)
        #
        # 	return cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:self.matchPoints], None, **draw_params)
        matches = self.bf.knnMatch(des1, des2, k=2)
        # matches = sorted(matches, key = lambda x:x.distance)
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        matches = good
        list_kp1 = []
        list_kp2 = []
        for m in good:
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))
        if draw:
            return cv2.drawMatches(img1, kp1, img2, kp2, good[:self.matchPoints], None, flags=2)
        else:
            return list_kp1, list_kp2
    def getImages(self):
        return self.grayimg1, self.grayimg2
    def getCorners(self):
        return self.corner(self.grayimg1), self.corner(self.grayimg2)
        # cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg1.JPG', self.grayimg1)
        # cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg2.JPG', self.grayimg2)
        # self.cornerMat1 = self.corner(self.grayimg1)
        # self.cornerMat2 = self.corner(self.grayimg2)
        # mat1, mat2 = self.match(self.grayimg1, self.grayimg2, False)
        #
        # print "Coordinate set 1 of ", len(mat1),": ", mat1
        # print "Coordinate set 2 of ", len(mat2),": ", mat2
        # cv2.imshow('img', self.match(self.grayimg1, self.grayimg2, True))
    def reset(self):
        self.grayimg1 = None
        self.grayimg2 = None

class Matcher(object):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, corner1, corner2, matchPoints, draw=False):
        kp1, des1 = corner1
        kp2, des2 = corner2
        matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
        list_kp1 = []
        list_kp2 = []
        for m in good:
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))
        if draw:
            return kp1, kp2, good[:matchPoints], list_kp1, list_kp2
        else:
            return list_kp1, list_kp2


class VideoStream(object):
    def __init__(self, capid):
        self.cap = cv2.VideoCapture(capid)
        self.cap.set(5,24)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        # self.cap.open(capid)

    def getFrame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()


class NNet(object):
    def __init__(self, points1, points2):
        self.inputarr = points1
        self.expected = points2
        self.net = pb.buildNetwork(2, 4, 2)

    # def


class Main(object):
    Camera1 = VideoStream(1)
    Camera2 = VideoStream(0)
    Matcher1 = Matcher()
    Thread1 = Corner()
    while(True):
        file1 = Camera1.getFrame()
        file2 = Camera2.getFrame()
        Thread1.setImages(file1, file2)
        corner1, corner2 = Thread1.getCorners()
        grayimg1, grayimg2 = Thread1.getImages()
        try:
            # Matcher1.match(corner1, corner2, 50, True)
            # mat1, mat2 = Matcher1.match(corner1, corner2, 50, False)
            kp1, kp2, good, mat1, mat2 = Matcher1.match(corner1, corner2, 50, True)
            print "Coordinate set 1 of ", len(mat1),": ", mat1
            print "Coordinate set 2 of ", len(mat2),": ", mat2
            cv2.imshow('img', cv2.drawMatches(grayimg1, kp1, grayimg2, kp2, good, None, flags=2))
        except:
            Thread1.reset()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Camera1.release()
            Camera2.release()
            Thread1.reset()
            cv2.destroyAllWindows()
            break

# <codecell>
