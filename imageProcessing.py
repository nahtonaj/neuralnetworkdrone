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
        return self.orb.detectAndCompute(img, None)

    def SURF(self, img):
        surf = cv2.xfeatures2d.SURF_create(400)
        return surf.detectAndCompute(img, None)

    def corner(self, img):
        return self.ORB(img)

    def match(self, img1, img2, draw):
        kp1, des1 = self.corner(img1)
        kp2, des2 = self.corner(img2)
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
        print 'Total matched points: ', len(good)
        # dim = np.random.randint(matchPoints, size=matchPoints)
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
        norm2 = self.normalize(data2, max)
        # print 'Normalized set 1:', norm1
        # print 'Normalized set 2:', norm2
        for x in range(len(norm1)):
            self.ds.addSample(norm1[x], norm2[x])
    def train(self):
        print "Training"
        trndata, tstdata = self.ds.splitWithProportion(.1)
        self.trainer.trainUntilConvergence(verbose=True,
                                           trainingData=trndata,
                                           maxEpochs=1000)
        self.trainer.testOnData(tstdata, verbose= True)
        # if raw_input('Save Network?: y/n\n')=='y':
        NetworkWriter.writeToFile(self.net, 'Network1.xml')
        print 'Saving network'

    def activate(self, data):
        for x in data:
            self.net.activate(x)

    def normalize(self, data, max):
        inp = np.asarray(data, dtype=np.float32)
        out = inp/np.asarray(max, dtype=np.float32)
        return out

    def denormalize(self, data, max):
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

    def getRemap(self, sizex, sizey):
        print 'Mapping...'

        oldx = np.arange(sizex)
        oldy = np.arange(sizey)
        mapx = np.zeros(shape=[sizex, sizey], dtype=np.float32)
        mapy = np.zeros(shape=[sizex, sizey], dtype=np.float32)

        for x in oldx:
            for y in oldy:
                newx, newy = self.getOutput(np.asarray((x,y)), np.asarray((sizex, sizey)))
                mapx[x][y] = newx
                mapy[x][y] = newy
        print 'Mapping Done'
        i = open('mapx.xml', 'wb')
        j = open('mapy.xml', 'wb')
        pickle.dump(mapx, i)
        pickle.dump(mapy, j)

        print 'Saving maps'
        return mapx, mapy

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
    # file1dir = '/mnt/pi/left.jpg'
    # file2dir = '/mnt/pi/right.jpg'
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
                # Camera1.getFrame()
                # Camera2.getFrame()
                while (file1 is None or file2 is None):
                    file1 = Camera1.getFrame()
                    file2 = Camera2.getFrame()
                    # file1 = cv2.resize(cv2.imread(file1dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    # file2 = cv2.resize(cv2.imread(file2dir), None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

                print 'done'
                height, width = file1.shape[:2]
                Corner1.setImages(file1, file2)
                corner1, corner2 = Corner1.getCorners()
                grayimg1, grayimg2 = Corner1.getImages()
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
                    remap = cv2.remap(grayimg1, mapy, mapx, cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT)
                    # print remap.shape[:2]
                    # print grayimg2.shape[:2]
                    # print 'Getting difference image...'
                    # diff = NNet1.getDifference(grayimg1, remap)
                    diff = cv2.absdiff(remap, grayimg2)
                    print cv2.matchTemplate(remap,grayimg2, method=cv2.TM_CCORR_NORMED)
                    remapsections = np.hsplit(remap, 4)
                    rightsections = np.hsplit(grayimg2, 4)
                    for i in range(4):
                        print 'Section ',i, ': ', cv2.matchTemplate(remapsections[i], rightsections[i], method=cv2.TM_CCORR_NORMED)
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
                try:
                    file1 = Camera1.getFrame()
                    file2 = Camera2.getFrame()
                    # file1 = cv2.imread(file1dir)
                    # file2 = cv2.imread(file2dir)
                    height, width = file1.shape[:2]
                    Corner1.setImages(file1, file2)
                    corner1, corner2 = Corner1.getCorners()
                    grayimg1, grayimg2 = Corner1.getImages()
                    kp1, kp2, good, mat1, mat2 = Matcher1.match(corner1, corner2, matchpoints, True)
                    print "Coordinate set 1 of ", len(mat1),": \n", mat1
                    print "Coordinate set 2 of ", len(mat2),": \n", mat2
                    cv2.imshow('img', cv2.drawMatches(grayimg1, kp1, grayimg2, kp2, good, None, flags=2))
                except:
                    pass
                # except:

                #     Corner1.reset()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Camera1.release()
                    # Camera2.release()
                    # Corner1.reset()
                    cv2.destroyAllWindows()
                    # cv2.waitKey(0)
                    m='0'
                    break
        if m=='4':
            print mapx, mapy
        if m=='5':
            print 'Exiting'
            break
        else:
            print 'Please select a valid choice'
            # return
# <codecell>
