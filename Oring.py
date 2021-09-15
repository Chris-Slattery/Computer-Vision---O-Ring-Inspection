import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt



def threshold (img, thresh):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
        	if img[x,y] > thresh:
        		img[x,y] = 255
        	else:
        		img[x,y] = 0
    return img

def imhist(img):
	hist = np.zeros(256)
	for x in range(0, img.shape[0]):
		for y in range(0, img.shape[1]):
			hist[img[x,y]]+=1
	return hist

def findT(hist):
    max = 0
    for i in range(256):
        if hist[i] > max:
            max = hist[i]
            peak = i
    T = peak - 50
    return T

def imageDilation(img):
    dilatedImage = img.copy()
    for x in range(1, img.shape[0]-1):
        for y in range(1, img.shape[1]-1):
            xKernel = [
            x-1, x, x+1,
            x-1, x, x+1,
            x-1, x, x+1]

            yKernel = [
            y-1, y-1, y-1,
            y, y, y,
            y+1, y+1, y+1]
            for n in range(len(xKernel)):
                if img[xKernel[n], yKernel[n]] == 255:
                    dilatedImage[x,y] = 255
    return dilatedImage
   


def imageErosion(img):
    erodedImage = img.copy()
    for x in range(1, img.shape[0]-1):
        for y in range(1, img.shape[1]-1):
            yKernel = [
            y-1, y-1, y-1,
            y, y, y,
            y+1, y+1, y+1]

            xKernel = [
            x-1, x, x+1,
            x-1, x, x+1,
            x-1, x, x+1]
            for n in range(len(xKernel)):
                if img[xKernel[n], yKernel[n]] == 0:
                    erodedImage[x,y] = 0
    return erodedImage


def componentLabel(img):
    q = Queue()
    curlab = 0
    blackLabel = 2
    connectedImage = img.copy()
    labels = img.copy()
    for x in range(1, img.shape[0]-1):
        for y in range(1, img.shape[1]-1):
            yKernel = [
            y-1, y-1, y-1,
            y, y, y,
            y+1, y+1, y+1]

            xKernel = [
            x-1, x, x+1,
            x-1, x, x+1,
            x-1, x, x+1]
            for n in range(len(xKernel)):
                if img[xKernel[n], yKernel[n]] == 255 & labels[xKernel[n], yKernel[n]] == 0:
                	curlab +=1
                	labels[n] = curlab
                	q.enqueue(n)

                	while not q.isEmtpy():
                		xKernel = q.dequeue()
                		for n in range(len(xKernel)):
                			#if img[xKernel[n], yKernel[n] == 255 & labels[xKernel[n], yKernel[n]] == 0:
                				labels[n] = curlab
                				q.enqueue(xKernel[n])

class Queue:
	def __init__(self):
		self.items = []


	def isEmtpy(self):
		return self.items == []


	def enqueue(self, item):
		self.items.insert(0, item)


	def dequeue(self):
		return self.items.pop()		



def passFail(img):
	margin = 0
	startXandY = 1
	aboveY = 0
	aboveX = 0
	belowY = 0
	belowX = 0
	botHalf = 0
	topHalf = 0
	for x in range(0, img.shape[0]):
		for y in range(0, img.shape[1]):
			yKernel = [
			y-1, y-1, y-1,
			y, y, y,
			y+1, y+1, y+1]

			xKernel = [
			x-1, x, x+1,
			x-1, x, x+1,
			x-1, x, x+1]
			if img[x, y] == 0:
				if x >= aboveX:
					aboveX = x
				if y >= aboveY:
					aboveY = y
				if startXandY == 1:
					belowX = x
					belowY = y
				startXandY = 2
				if y <= belowY:
					belowY = y
				if x <= belowX:
					belowX = x



	highAndLow = aboveX + belowX
	halfWay = highAndLow / 2



	for x in range(belowX, int(halfWay)):
		for y in range(belowY, aboveY+1):
			if img[x][y] == 0:
				topHalf +=1



	for x in range(int(halfWay), aboveX+1):
		for y in range(belowY, aboveY+1):
			if img[x][y] == 0:
				botHalf +=1

	if topHalf > botHalf:
		margin = topHalf - botHalf
	elif botHalf > topHalf:
		margin = botHalf - topHalf
	else:
		margin = topHalf - botHalf

	if margin <=90:
		print("Pass")
	else:
		print("Fail")









#read in images
for i in range(15):
    #read in an image into memory
    img = cv.imread('c:/Users/chris/Pictures/Orings/Oring' + str(i+1) + '.jpg',0)
    copy = img.copy()
    before = time.time()
    hist = imhist(img)
    T = findT(hist) 
    plt.plot(hist)
    plt.plot((T, T), (0, 500), 'g')
    plt.show()

    img = threshold(copy, T)
    img = cv.putText(img, 'Hello', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0))
    after = time.time()

    print("Time taken to process hand coded thresholding: " + str(after-before))


    dilatedImage = imageDilation(img)
    erodedImage = imageErosion(dilatedImage)
    erodedImage1 = imageErosion(erodedImage)
    erodedImage2 = imageErosion(erodedImage1)
    erodedImage3 = imageErosion(erodedImage2)
    cv.imshow('thresholded image 2',copy)
    cv.imshow('Dilation and Erosion', erodedImage3)
    passOrFail = passFail(img)
    
    
    cv.waitKey(0)
cv.destroyAllWindows()