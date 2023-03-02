import cv2 as cv

img = cv.imread('.\\resources\\photos\\cats.jpg')

cv.imshow('Cats', img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("Cats grey", grey)

#1. Simple tresholding

treshold, tresh = cv.threshold(grey, 150, 255, cv.THRESH_BINARY)
cv.imshow("Cats grey", tresh)

treshold, tresh = cv.threshold(grey, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Cats grey", tresh)

#2. Adaptive Tresholding

adaptive_tresh = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 1)
cv.imshow('ad', adaptive_tresh)

cv.waitKey(0)