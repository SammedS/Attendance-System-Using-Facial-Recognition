import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing images of students
path = r'C:\Users\Sanjith\Documents\Sjbit\Hackathon project Horizon\Smart Attendance System\Attendance system\Images'
images = []
classNames = []
List1 = os.listdir(path)

# Load images and their corresponding names
for cl in List1:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces from loaded images
def findEncodings(images):
    encodeList = []
    for img in images:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(imgRGB)
        encodings = face_recognition.face_encodings(imgRGB)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print(f"No face found in {img}")
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dtString}\n')

# Encoding faces from loaded images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Compare faces in the current frame with known faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Display the webcam feed with recognized faces
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
