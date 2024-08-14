from flask import Flask, render_template,redirect,url_for
import os
import cv2
import attend
import face_recognition
import numpy as np

app = Flask(__name__, template_folder='templates',static_folder='templates/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-python-app', methods=['POST'])
def start_python_app():
    # Code to start your Python app goes here
    print("Starting Python app from Flask")
    path = r'C:\Users\Lenovo\Desktop\Smart Attendance System (3)\Smart Attendance System\Attendance system\Images'
    images = []
    image_names=[]
    classNames = []
    List1 = os.listdir(path)
    flag=False
    global name
    k = 0

    for cl in List1:
        image_names.append(cl)
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
        #print(curImg)
    print(image_names)
    encodeListKnown = attend.findEncodings(images,path,List1)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:

        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # print(encodeListKnown)
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            k+=1
            # print(faceDis)
            if any(matches):
                print(k)
                matchIndex = np.argmin(faceDis)

            # print("Entering0")

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                attend.markAttendance(name)
                #print(name)
                flag=True


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
        if(flag==True):
            return redirect(url_for('success'))

@app.route('/success')
def success():

    return render_template('attendence.html',name=name)

if __name__ == '__main__':
    app.run(debug=True)
