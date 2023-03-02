import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier(".\\resources\\photos\\haar.xml")

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']

DIR = r'C:\Users\permo\OneDrive\Рабочий стол\opencv\resources\photos\faces\val'
p=[]
for i in os.listdir(DIR):
    p.append(i)
print(p)

features = np.load('.\\features.npy', allow_pickle=True)
labels = np.load('.\\labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('.\\face_trained.yml')

for person in people:
        path = f".\\resources\\photos\\faces\\val\\{person}\\"
    
        for i in os.listdir(path):
            img_path = f".\\resources\\photos\\faces\\val\\{person}\\{i}"
            img = cv.imread(img_path)

            #cv.imshow(f'{person} + {i}', img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]

                label, confidence = face_recognizer.predict(faces_roi)
                print(f'Label = {people[label]} with a confidence of {confidence}')

                cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
                cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

            cv.imshow(f'{person} + {i}', img)
            #break



# img = cv.imread(f".\\resources\\photos\\faces\\val\\elton_john\\1.jpg")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# features = np.load('.\\features.npy', allow_pickle=True)
# labels = np.load('.\\labels.npy')

# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('.\\face_trained.yml')



# # # Detect the face in the image
# faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# for (x,y,w,h) in faces_rect:
#     faces_roi = gray[y:y+h,x:x+w]

#     label, confidence = face_recognizer.predict(faces_roi)
#     print(f'Label = {people[label]} with a confidence of {confidence}')

#     cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
#     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Face', img)

cv.waitKey(0)