import cv2 as cv

def rescale(img, scale=0.5):
    height = int(img.shape[0]*scale)
    width = int(img.shape[1]*scale)
    dimensions = (width, height)

    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height, capture):
    #Live videos
    capture.set(3, width)
    capture.set(4, width)

img = cv.imread('resources\\photos\\lady.jpg')

#cv.imshow('Cats', img)
cv.imshow('Cats', rescale(img))
cv.waitKey(0)