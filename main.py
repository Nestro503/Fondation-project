import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)   #pour choisir la source de capture video : 0 pour camera PC et 1 pour IV_Cam

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', gray)

    # Check if the user pressed 'q'
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
