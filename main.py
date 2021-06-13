import os
import csv
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from datetime import date

#read chinese path name
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img


def encodeFaces(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(face_recognition.face_encodings(img))
        encode_img = face_recognition.face_encodings(img)[0]
        # print(type(encode_img))
        encodelist.append(encode_img)
    return encodelist

def markAttendance(name):
    with open('attendance.csv','r+',newline='')as file:
        writer = csv.writer(file)
        namelist = []
        datalist = csv.reader(file)
        for line in datalist:
            # print(line)
            namelist.append(line[0])
        if name not in namelist:
            time = datetime.now().strftime('%H:%M:%S')
            print(time)
            writer.writerow([name] + [time])
            
def main():
    #prep
    # today = date.today().strftime('%d/%m/%Y')
    # newfile = f'attendance {today}.csv'
    open( 'attendance.csv' , 'wb')
    path = 'Images'
    images = []
    stu_names = []
    os.chdir('D:\python_projects\Attendance project')
    print(os.getcwd())
    imglist = os.listdir(path)
    for stu in imglist:
        curImg = cv_imread(f'{path}/{stu}')
        images.append(curImg)
        stu_names.append(os.path.splitext(stu)[0])
        print(os.path.splitext(stu)[0])
        # print(curImg)

    # encode known faces
    known_encoded = encodeFaces(images)
    print(len(known_encoded))
    print('encoding complete')

    # webcam feed
    cap = cv2.VideoCapture(0)

    while True:
        read,imgF = cap.read()
        # shrink image to speed up process
        imgS = cv2.resize(imgF,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
        facelocs = face_recognition.face_locations(imgS)
        curFaces = face_recognition.face_encodings(imgS,facelocs)
        
        for face,loc in zip(curFaces,facelocs):
            matches = face_recognition.compare_faces(known_encoded,face,0.4)
            dist = face_recognition.face_distance(known_encoded,face)
            print(stu_names)
            print(dist)
            print(matches)
            min_index = np.argmin(dist)
            print(min_index)
            if matches[min_index]:
                name = stu_names[min_index]
                y1,x2,y2,x1 = loc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(imgF,(x1,y1),(x2,y2),(255,255,0),2)
                cv2.rectangle(imgF,(x1,y2),(x2,y2-30),(255,255,0),cv2.FILLED)
                cv2.putText(imgF,name,(x1+5,y2-5),cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255,255,255),1)
                markAttendance(name)

        cv2.imshow('Webcam',imgF)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()