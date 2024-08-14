import cv2
import numpy as np
import face_recognition
import os
import webbrowser
from datetime import datetime

path = r'C:\Users\Sanjith\Documents\Sjbit\Hackathon project Horizon\Smart Attendance System\Attendance system\Images'
images = []
classNames = []
List1 = os.listdir(path)

for cl in List1:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            encode = face_recognition.face_encodings(imgRGB)[0]  # Encode face
            encodeList.append(encode)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    return encodeList


def markAttendance(name):
    with open('attend1.csv', 'a+') as k:  # Use 'a+' to append instead of 'w+' to overwrite
        myDataList = k.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%D')
            k.write(f'\n{name},{dtString}')
        # Assuming this part is correct, keep it as-is


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Check if there are faces detected
        if len(encodeListKnown) == 0:
            continue  # Skip processing if no known encodings

        # Compare face encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

        # Handle the case where no matches are found
        if not any(matches):
            continue  # Skip this face if no matches found

        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
