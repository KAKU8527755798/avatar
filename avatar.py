# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:13:04 2023

@author: User
"""
import cv2

# Load the video
video = cv2.VideoCapture(0)

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("/Users/sahilaggarwal/Desktop/sahil aggarwal/haarcascade_frontalface_default.xml")

# Define the avatar image
avatar = cv2.imread("/Users/sahilaggarwal/Downloads/india-652857.png",-1)

while True:
    # Read the next frame from the video
    ret, frame = video.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the faces
    for (x, y, w, h) in faces:
        # Resize the avatar to match the size of the face
        avatar_resized = cv2.resize(avatar, (w, h), interpolation=cv2.INTER_AREA)

        # Get the alpha channel of the avatar
        alpha = avatar_resized[:, :, 3] / 255.0

        # Copy the avatar onto the frame
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = alpha * avatar_resized[:, :, c] + (1 - alpha) * frame[y:y+h, x:x+w, c]

    # Display the frame
    cv2.imshow("Video", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
   

# Release the video and destroy the windows
video.release()
cv2.destroyAllWindows()