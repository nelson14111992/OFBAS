__author__ = 'MY'
import cv2
import face_recognition
import numpy as np
import sys
import pickle
from datetime import datetime
import os
import tensorflow as tf
import dlib
from mtcnn.mtcnn import MTCNN
from sklearn.externals import joblib

#Conversion to RGB
#imgmark_bgr = face_recognition.load_image_file('C:\\Users\\MY\\PycharmProjects\\untitled1\\mark.jpg')
#imgmark_rgb =  cv2.cvtColor(imgmark_bgr,cv2.COLOR_BGR2RGB)

#cv2.imshow('bgr',imgmark_bgr)
#cv2.imshow('rgb',imgmark_rgb)
#cv2.waitKey(0)

#Find Face & Draw Box
#face = face_recognition.face_locations(imgmark_rgb)[0]
#copy = imgmark_rgb.copy()

#cv2.rectangle(copy,(face[3],face[0]),(face[1],face[2]),(255,0,255),2)
#cv2.imshow('copy',copy)
#cv2.imshow('imgmark_rgb',imgmark_rgb)
#cv2.waitKey(0)

#Encoding Images
#Encoded_imgmark = face_recognition.face_encodings(imgmark_rgb)[0]

#Encoding and comparing faces
#imgmark2_bgr = face_recognition.load_image_file('sunder.webp')
#imgmark2_rgb = cv2.cvtColor(imgmark2_bgr,cv2.COLOR_BGR2RGB)
#Encoded_imgmark2 = face_recognition.face_encodings(imgmark2_rgb)[0]
#result = face_recognition.compare_faces([Encoded_imgmark],Encoded_imgmark2)

#print("Result:", result)


#Dataset path

path = 'C:\\Users\\MY\\PycharmProjects\\untitled1\\dataset'

#Make list of Dataset
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    # absolute_path = os.path.join(os.getcwd(), cl);
    absolute_path = os.path.join(path, cl);
    #img = cv2.imread(absolute_path)
    #curImg = cv2.imread('{path}/{cl}')
    images.append(absolute_path)
    classNames.append(os.path.splitext(cl)[0])

#Encode all dataset
def findencodings(images):

    encodeList = []
    for img in images:
        imgmark_bgr = face_recognition.load_image_file(img)
        img_rgb = cv2.cvtColor(imgmark_bgr,cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img_rgb)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findencodings(images)


#face_liveliness


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('models/colorspace_ycrcbluv_print.pkl')
sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    faces3 = net.forward()

    measures[count%sample_number]=0
    height, width = img.shape[:2]
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.7:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 5)
            roi = img[y:y1, x:x1]

            point = (0,0)

            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]

            measures[count % sample_number] = prob

            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

            point = (x, y-5)

            print (measures, np.mean(measures))


            if 0 not in measures:
                text = "True"
                if np.mean(measures) >= 0.7:
                    text = "False"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        count+=1
        cv2.imshow('img_rgb', img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break




cap.release()
cv2.destroyAllWindows()


#Create attendance csv file
#def markattendance(name):
 #   with open('Attendance.csv','r+') as f:
  #      myDataList = f.readlines()
   #     nameList = []
    #    for line in myDataList:
     #       entry = line.split(',')
      #      nameList.append(entry[0])
       # if name not in nameList:
        #    now = datetime.now()
         #   time = now.strftime('%I:%M:%S:%p')
          #  date = now.strftime('%d-%B-%Y')
           # f.writelines('n{name}, {time}, {date}')
    #capture.release()
    #quit()
#compare realtime Camera



#capture = cv2.VideoCapture(0)

#while True:
    #success, img = capture.read()
    #imgS=cv2.resize(img, (0,0), None, 0.25,0.25)
    #imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #faces_in_frame = face_recognition.face_locations(imgS)
    #encoded_faces = face_recognition.face_encodings(imgS,faces_in_frame)
    #liveness(img)
    #for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        #matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        #faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        #matchIndex = np.argmin(faceDist)
        #print(matchIndex)
        #if matches[matchIndex]:
            #name = classNames[matchIndex].upper().lower()
            #y1,x2,y2,x1 = faceloc
             #since we scaled down by 4 times
            #y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            #cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
          #  markattendance(name)
            #cv2.imshow('webcam', img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break;
#capture.release()
#cv2.destroyAllWindows()




