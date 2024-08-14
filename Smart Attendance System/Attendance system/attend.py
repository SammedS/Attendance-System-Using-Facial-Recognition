import cv2
import face_recognition
from datetime import datetime
import webbrowser

def findEncodings(images,path,List1):
    encodeList = []
    for img in List1:
        try:
            imgcode = cv2.imread(f'{path}/{img}')  # Load image
            imgRGB = cv2.cvtColor(imgcode, cv2.COLOR_BGR2RGB)  # Convert to RGB
            encode = face_recognition.face_encodings(imgRGB)[0]  # Encode face
            encodeList.append(encode)
            #print(encodeList)
        except Exception as e:
            print(f"Error processing image {img}: {str(e)}")
    return encodeList

def markAttendance(name):
    with open('attend1.csv','a+') as k:
        myDataList = k.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%D')
            k.writelines(f'\n{name},{dtString}')
