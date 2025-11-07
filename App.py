import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

st.set_page_config(page_title="Face Attendance System", layout="wide")

path = 'Training_images'
if not os.path.exists(path):
    st.error("❌ 'Training_images' folder not found!")
    st.stop()

images = []
classNames = []
marked_names = set()

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

def markUniqueAttendance(name):
    filename = 'Attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")

    global marked_names
    if not marked_names:
        with open(filename, 'r') as f:
            marked_names = set(line.strip().split(',')[0] for line in f.readlines()[1:])

    if name not in marked_names:
        marked_names.add(name)
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        with open(filename, 'a') as f:
            f.write(f"{name},{dtString}\n")

st.sidebar.header("Setup")
st.sidebar.write("Encoding known faces...")
encodeListKnown = findEncodings(images)
st.sidebar.success("✅ Encoding complete")
st.sidebar.write(f"Known people: {', '.join(classNames)}")

st.title("📸 Face Recognition Attendance System")

FRAME_WINDOW = st.image([])
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎯 Unique people detected this session:")
    name_display = st.empty()
with col2:
    if os.path.exists("Attendance.csv"):
        with open("Attendance.csv", "r") as f:
            csv_data = f.read()
        st.download_button("📥 Download Attendance CSV", csv_data, "Attendance.csv", mime="text/csv")
    else:
        st.info("No attendance file yet. Start camera to mark attendance.")


camera = cv2.VideoCapture(0)
frame_count = 0

run = st.checkbox('Start Camera')

while run:
    success, img = camera.read()
    if not success:
        st.error("⚠️ Unable to access camera.")
        break

    frame_count += 1

    if frame_count % 10 == 0: 
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            name = "UNKNOWN"
            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if name != "UNKNOWN":
                markUniqueAttendance(name)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img)

    if marked_names:
        name_display.markdown(f"**{', '.join(marked_names)}**")
    else:
        name_display.text("No one detected yet.")

camera.release()

