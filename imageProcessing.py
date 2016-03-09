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


class ImProcess(object):
    # global img1, img2, grayimg1, grayimg2, cornerMat1, cornerMat2, temp1
    def __init__(self, file1, file2, cornerAlg='orb', matchAlg='BF', matchPoints=50):
        self.cornerAlg = cornerAlg
        self.matchAlg = matchAlg
        self.matchPoints = matchPoints
        self.img1 = file1#cv2.imread(file1)
        self.img2 = file2#cv2.imread(file2)
        self.grayimg1 = self.initiateimage(self.img1)
        self.grayimg2 = self.initiateimage(self.img2)

    def initiateimage(self, image):
        if image is None:
            print 'Image is null!'
            break
        else:
            imgres = cv2.resize(image, None, fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)

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

    def corner(self, img, ver):
        if ver == 'fast':
            return self.FAST(img)
        if ver == 'orb':
            return self.ORB(img)
        if ver == 'surf':
            return self.SURF(img)

    def match(self, img1, img2, ver, ver2, draw):
        kp1, des1 = self.corner(img1, ver)
        kp2, des2 = self.corner(img2, ver)
        if ver2 == 'FLANN':
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in xrange(len(matches))]

            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            return cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:self.matchPoints], None, **draw_params)
        if ver2 == 'BF':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            # matches = sorted(matches, key = lambda x:x.distance)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            # matches = good
            list_kp1 = []
            list_kp2 = []
            for m in good:
                img1_idx = m.queryIdx
                img2_idx = m.trainIdx
                (x1,y1) = kp1[img1_idx].pt
                (x2,y2) = kp2[img2_idx].pt
                list_kp1.append((x1, y1))
                list_kp2.append((x2, y2))
            if draw:
                return cv2.drawMatches(img1, kp1, img2, kp2, good[:self.matchPoints], None, flags=2)
            else:
                return list_kp1, list_kp2
    def main(self):
        cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg1.JPG', self.grayimg1)
        cv2.imwrite('/home/jonathan/Documents/ImageProcessingImages/grayimg2.JPG', self.grayimg2)
        # self.cornerMat1 = self.corner(self.grayimg1)
        # self.cornerMat2 = self.corner(self.grayimg2)
        mat1, mat2 = self.match(self.grayimg1, self.grayimg2, self.cornerAlg, self.matchAlg, False)

        print "Coordinate set 1 of ", len(mat1),": ", mat1
        print "Coordinate set 2 of ", len(mat2),": ", mat2
        cv2.imshow('img', self.match(self.grayimg1, self.grayimg2, self.cornerAlg, self.matchAlg, True))

    # self.grayimg1[self.cornerMat1>0.01*self.cornerMat1.max()]=[50]
    # self.grayimg2[self.cornerMat2>0.01*self.cornerMat2.max()]=[50]
    # cv2.imshow('image1', temp1)
    # cv2.imshow('image2', temp2)
    # cv2.imshow('image2', self.i)

class VideoStream(object):
    def __init__(self, capid):
        self.cap = cv2.VideoCapture(capid)
        self.cap.open(capid)
    def getFrame(self):
        ret, frame = self.cap.read()
      	return frame
    def release(self):
        self.cap.release()


class NNet(object):
    def __init__(self, points1, points2):
        self.inputarr = points1
        self.expected = points2
        self.net = pb.buildNetwork(2,4,2)
    # def




class Main(object):
    Camera1 = VideoStream(0)
    while(True):
	    file1 = Camera1.getFrame()#'/home/jonathan/Documents/ImageProcessingImages/img1.JPG'
	    file2 = cv2.imread('/home/jonathan/Documents/ImageProcessingImages/img2.JPG')
	    Thread1 = ImProcess(file1, file2, 'orb', 'BF', 50)
	    Thread1.main()
	    if cv2.waitKey(2) & 0xFF == ord('q'):
	        Camera1.release()
	        cv2.destroyAllWindows()
	        break


# <codecell>
