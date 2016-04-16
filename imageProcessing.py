# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

import cPickle as pickle



class Corner(object):
    def __init__(self):
        self.orb = cv2.ORB_create()
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

        # find the keypoints with ORB
        return self.orb.detectAndCompute(img, None)

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
    def getImages(self):# self.net.randomize()
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
            if m.distance < 0.65 * n.distance:
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
        # print list_kp1
        dim = range(matchPoints)
        if draw:
            if matchPoints>len(good):
                return np.asarray(kp1), np.asarray(kp2), np.asarray(good), np.asarray(list_kp1), np.asarray(list_kp2)
            else:
                return np.asarray(kp1)[dim], np.asarray(kp2)[dim], np.asarray(good)[dim], np.asarray(list_kp1)[dim], np.asarray(list_kp2)[dim]
        else:
            return list_kp1[dim], list_kp2[dim]

class VideoStream(object):
    def __init__(self, capid):
        self.cap = cv2.VideoCapture(capid)
        self.cap.set(5, 60)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        # self.cap.open(capid)

    def getFrame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()

class NNet(object):
    def __init__(self, inpNeurons, hiddenNeurons, outNeurons):
        self.net = buildNetwork(inpNeurons, hiddenNeurons, outNeurons, hiddenclass=TanhLayer, bias=True)
        if raw_input('Recover Network?: y/n\n')=='y':
            print 'Recovering Network'
            net = NetworkReader.readFrom('Network1.xml')
        else:
            print 'New Network'
            self.net.randomize()
        print self.net
        self.ds = SupervisedDataSet(inpNeurons,outNeurons)
        self.trainer = BackpropTrainer(self.net, self.ds, learningrate = 0.01, momentum=0.99)
    def addTrainDS(self, data1, data2, max):
        norm1 = self.normalize(data1, max)
        # n1x, n1y = np.hsplit(norm1, 2)
        # n1x = n1x.flatten()
        # n1y = n1y.flatten()
        norm2 = self.normalize(data2, max)
        # n2x, n2y = np.hsplit(norm2, 2)
        # n2x = n2x.flatten()
        # n2y = n2y.flatten()
        # print 'Normalized set 1:', norm1
        # print 'Normalized set 2:', norm2
        for x in range(len(norm1)):
            self.ds.addSample(norm1[x], norm2[x])
    def train(self):
        print "Training"
        # print self.trainer.train()
        trndata, tstdata = self.ds.splitWithProportion(.1)
        self.trainer.trainUntilConvergence(verbose=True,
                                           trainingData=trndata,
                                           maxEpochs=1000)
        # self.trainer.trainOnDataset(trndata,500)
        self.trainer.testOnData(tstdata, verbose= True)
        # if raw_input('Save Network?: y/n\n')=='y':
        NetworkWriter.writeToFile(self.net, 'Network1.xml')
        print 'Saving network'

    def activate(self, data):
        for x in data:
            self.net.activate(x)

    def normalize(self, data, max):
        # if type(data)==int or type(data)==np.uint8:
        #     return data/max
        # # if type(data) == int and type(max) == int:
        # #     return data/max
        # else:
        #     normData = np.empty((len(data), 2))
        #     for x in [0,1]:
        #         for y in range(len(data)):
        #             val = data[y][x]
        #             normData[y][x] = (val)/(max[x])
        #     # print normData
        #     return np.asarray(normData)
        inp = np.asarray(data, dtype=np.float32)
        out = inp/np.asarray(max, dtype=np.float32)
        return out

    def denormalize(self, data, max):
        # if type(data) == int or type(data) == np.uint8 or type(data)==np.float64:
        #     return data*max
        # else:
        #     deNorm = np.empty((len(data), 2))
        #     for x in [0,1]:
        #         for y in range(len(data)):
        #             val = data[y][x]
        #             if val>1:
        #                 val=1
        #             else:
        #                 deNorm[y][x] = val*max[x]
        #     return np.asarray(deNorm)
        inp = np.asarray(data, dtype=np.float32)
        out = inp*np.asarray(max, dtype=np.float32)
        return out
    def getOutput(self, mat, max):
        norm = self.normalize(mat,max)
        # print 'Normalized value: ', norm
        if len(mat.shape)>1:
            for i in range(mat.shape[0]):
                out = self.net.activate(norm[0])
        else:
            out=self.net.activate(norm)
        # print 'Normalized output: ', out
        denorm = self.denormalize(out,max)
        # print 'Denormalized: ', denorm
        return denorm
        # if type(mat) == tuple and type(max) == type:
        # norm = (self.normalize(mat[0],max[0]), self.normalize(mat[1],max[1]))
        # print norm
        # out = self.net.activate(norm)
        # denorm = (self.denormalize(out[0],max[0]), self.denormalize(out[1],max[1]))
        # return denorm
        # else:
            # norm = self.normalize(mat, max)
            # out = []
            # for val in norm:
            #     out.append(self.net.activate(val))
            # return np.asarray(self.denormalize(out, max))
    def getRemap(self, sizex, sizey):
        print 'Mapping...'

        oldx = np.arange(sizex)
        oldy = np.arange(sizey)
        mapx = np.zeros(shape=[sizex, sizey], dtype=np.float32)
        mapy = np.zeros(shape=[sizex, sizey], dtype=np.float32)

        for x in oldx:
            for y in oldy:
                # mapx[x][y] = 2*x
                # mapy[x][y] = .5*y
                newx, newy = self.getOutput(np.asarray((x,y)), np.asarray((sizex, sizey)))
                mapx[x][y] = newx
                mapy[x][y] = newy
        # newx, newy = self.getOutput(oldxy, np.asarray((sizex, sizey)))
        print 'Mapping Done'
        i = open('mapx.xml', 'wb')
        j = open('mapy.xml', 'wb')
        pickle.dump(mapx, i)
        pickle.dump(mapy, j)

        print 'Saving maps'
        return mapx, mapy


                # try:
                #     newxy = self.getOutput(np.asarray((x,y)),mat1.shape[:2])
                    # print 'Old:', (x,y), 'new coord', newxy
                    # outImage[newxy[0]][newxy[1]] = mat1[x][y]
                # except:
                #     print 'Error'
                #     pass


    def getDifference(self, image, remap):
        diff = np.empty_like(image)
        if image.shape[:2]==remap.shape[:2]:
            # diff = np.absolute((remap-image))
            diff = np.absolute(remap-image)
            return diff
        else:
            print 'Image and remap size unequal'

if __name__ == "__main__":

    Camera1 = VideoStream(1)
    Camera2 = VideoStream(2)
    file1dir = '/home/jonathan/Documents/ImageProcessingImages/left5.jpg'
    file2dir = '/home/jonathan/Documents/ImageProcessingImages/right5.jpg'
    Matcher1 = Matcher()
    Corner1 = Corner()
    NNet1 = NNet(2,6,2)
    try:
        mapxdir = open('mapx.xml', 'rb')
        mapydir = open('mapy.xml', 'rb')
        matchpointsdir = open('matchpoints.xml', 'rb')
        resizedir = open('resize.xml', 'rb')
        mapx = pickle.load(mapxdir)
        mapy = pickle.load(mapydir)
        matchpoints = pickle.load(matchpointsdir)
        resize = pickle.load(resizedir)
        print 'Recovering maps'
    except:
        print 'New maps'
        mapx = None
        mapy = None
        matchpoints = 50
        resize = .5
    while True:
        m = raw_input("Select Task \n 1. Train\n 2. Run\n 3. Cameras\n 4. Test Module\n 5. Exit\n >>:")
        if m == '0':
            pass
        # Camera1.release()
        # Camera2.release()
        if m == '1':
            FrameMatch=True
            p = open('matchpoints.xml', 'wb')
            q = open('resize.xml', 'wb')
            matchpoints = int(raw_input("Number of Match Points: "))
            resize = float(raw_input("Image scale: "))
            pickle.dump(matchpoints, p)
            pickle.dump(resize, q)
            file1 = None
            file2 = None
            while FrameMatch:
                print 'Getting frames...'
                Camera1.getFrame()
                Camera2.getFrame()
                while (file1 is None or file2 is None):
                    file1 = Camera1.getFrame()
                    file2 = Camera2.getFrame()
                print 'done'
                # file1 = cv2.resize(cv2.imread(file1dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                # file2 = cv2.resize(cv2.imread(file2dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                # cv2.imshow('left', file1)
                # cv2.imshow('right', file2)
                # if file1 is None or file2 is None:
                #     print 'Camera Error: Failed to get image'
                #     # Camera1.release()
                #     # Camera2.release()
                #     break
                # while(True):
                height, width = file1.shape[:2]
                Corner1.setImages(file1, file2)
                corner1, corner2 = Corner1.getCorners()
                grayimg1, grayimg2 = Corner1.getImages()
                # try:
                # Matcher1.match(corner1, corner2, 50, True)
                # mat1, mat2 = Matcher1.match(corner1, corner2, 50, False)
                kp1, kp2, good, mat1, mat2 = Matcher1.match(corner1, corner2, matchpoints, True)
                # cv2.imshow('img', cv2.drawMatches(grayimg1, kp1, grayimg2, kp2, good, None, flags=2))
                if len(mat1)==0:
                    break
                print "Coordinate set 1 of ", len(mat1),": ", mat1, "\nCoordinate set 2 of ", len(mat2),": ", mat2
                NNet1.addTrainDS(mat1, mat2, (height, width))
                NNet1.train()
                # pickle
                if raw_input('Retrain network?: y/n\n')=='n':
                    mapx, mapy = NNet1.getRemap(height, width)
                    FrameMatch=False

                    # print NNet1.getOutput(mat1, (height, width))
                    # except:

                    #     Corner1.reset()
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     Camera1.release()
                    #     Camera2.release()
                    #     # Corner1.reset()
                    #     cv2.destroyAllWindows()
                    # break
        if m == '2':
            while(True):
                file1 = Camera1.getFrame()
                file2 = Camera2.getFrame()
                # file1 = cv2.resize(cv2.imread(file1dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                # file2 = cv2.resize(cv2.imread(file2dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                height, width = file1.shape[:2]
                Corner1.setImages(file1, file2)
                corner1, corner2 = Corner1.getCorners()
                grayimg1, grayimg2 = Corner1.getImages()
                # try:
                # Matcher1.match(corner1, corner2, 50, True)
                # mat1, mat2 = Matcher1.match(corner1, corner2, 50, False)
                try:
                    kp1, kp2, good, mat1, mat2 = Matcher1.match(corner1, corner2, matchpoints, True)
                except:
                    pass
                # print "Coordinate set 1 of ", len(mat1),": \n", mat1
                # print "Coordinate set 2 of ", len(mat2),": \n", mat2
                # NNet1.addTrainDS(mat1, mat2, (height, width))
                # NNet1.train()
                # newx, newy = NNet1.transformImage(width, height)
                if mapx==None or mapy==None:
                    print 'Please train network first'
                    m = '0'
                    break
                else:
                    # print 'Remapping...'
                    remap = cv2.remap(grayimg1, mapy, mapx, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
                    # print remap.shape[:2]
                    # print grayimg2.shape[:2]
                    # print 'Getting difference image...'
                    # diff = NNet1.getDifference(grayimg1, remap)
                    diff = cv2.absdiff(remap, grayimg2)
                    print cv2.matchTemplate(remap,grayimg2, method=cv2.TM_CCORR_NORMED)
                    # diff = cv2.adaptiveThreshold(diff, 80, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=7, C=5)
                cv2.imshow('right', grayimg2)
                cv2.imshow('remapped', remap)
                cv2.imshow('diff', diff)
                # cv2.imwrite("~/home/jonathan/Documents/ImageProcessingImages/diff.jpg", diff)
                # cv2.imwrite("~/home/jonathan/Documents/ImageProcessingImages/right.jpg", grayimg2)
                # print "remap: ", remap
                # cv2.imshow('img', cv2.drawMatches(grayimg1, kp1, grayimg2, kp2, good, None, flags=2))
                # except:

                #     Corner1.reset()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    Camera1.release()
                    Camera2.release()
                    # Corner1.reset()
                    cv2.destroyAllWindows()
                    # cv2.waitKey(0)
                    m='0'
                    break

        if m=='3':
            while(True):
                file1 = Camera1.getFrame()
                file2 = Camera2.getFrame()
                # file1 = cv2.imread(file1dir)
                # file2 = cv2.imread('/home/jonathan/Documents/ImageProcessingImages/imRung2.JPG')
                height, width = file1.shape[:2]
                Corner1.setImages(file1, file2)
                corner1, corner2 = Corner1.getCorners()
                grayimg1, grayimg2 = Corner1.getImages()
                # try:
                # Matcher1.match(corner1, corner2, 50, True)
                # mat1, mat2 = Matcher1.match(corner1, corner2, 50, False)
                try:
                    kp1, kp2, good, mat1, mat2 = Matcher1.match(corner1, corner2, matchpoints, True)
                    print "Coordinate set 1 of ", len(mat1),": \n", mat1
                    print "Coordinate set 2 of ", len(mat2),": \n", mat2
                    cv2.imshow('img', cv2.drawMatches(grayimg1, kp1, grayimg2, kp2, good, None, flags=2))
                except:
                    pass
                # except:

                #     Corner1.reset()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    Camera1.release()
                    Camera2.release()
                    # Corner1.reset()
                    cv2.destroyAllWindows()
                    # cv2.waitKey(0)
                    m='0'
                    break
        if m=='4':
            pass
        if m=='5':
            print 'Exiting'
            break
        else:
            print 'Please select a valid choice'
            # return
# <codecell>
