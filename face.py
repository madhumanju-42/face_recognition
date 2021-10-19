import cv2

#trained data set
trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read a image
img = cv2.imread('images/dhoni.jpeg')
#cv2.imshow('dhoni',img)
#cv2.waitKey()

#convert into grayscale
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
faces = trainedDataset.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('dhoni',img)
#cv2.imshow('gray',gray)
cv2.waitKey()