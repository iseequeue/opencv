import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


# p=[]
# for i in os.listdir(DIR):
#     p.append(i)
# print(p)

features=[]
labels=[]

# a = cv.imread('.\\resources\\photos\\faces\\train\\Ben Afflek\\4.jpg')
# cv.imshow('a', a)

# b = cv.imread('faces\\train\\Ben Afflek\\4.jpg')
# cv.imshow('a', b)
haar_cascade = cv.CascadeClassifier(".\\resources\\photos\\haar.xml")

def create_train():
    for person in people:
        path = f".\\resources\\photos\\faces\\train\\{person}\\"
        print(1)
        label = people.index(person)
    
        for img in os.listdir(path):
            print(img)
            img_path = f".\\resources\\photos\\faces\\train\\{person}\\{img}"
            img_array = cv.imread(img_path)

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)
create_train()
print('train done---')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

cv.waitKey(0)