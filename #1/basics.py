import cv2 as cv

img=cv.imread('resources\\photos\\lady.jpg')

cv.imshow('Cats', img)

#1. Converting ti grayscale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grey', grey)


#2. # Blur 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#3. # Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# 4.Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# 5.Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow('Eroded', eroded)

# 6.Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# 7.Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)




cv.waitKey(0)