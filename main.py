import numpy as np
import time
import cv2

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}



# youhou ca marche
def aruco_display(corners, ids, rejected, image):

    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, "ID: {}".format(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Map ArUco marker coordinates to game card coordinates
            if markerID == 0:
                mapped_coordinates = (0, 0)
            elif markerID == 1:
                mapped_coordinates = (1000, 0)
            elif markerID == 2:
                mapped_coordinates = (1000, 1000)
            elif markerID == 3:
                mapped_coordinates = (0, 1000)
            else:
                mapped_coordinates = (cX, cY)

            # Display mapped coordinates on the image
            cv2.putText(image, "Mapped: ({},{})".format(mapped_coordinates[0], mapped_coordinates[1]),
                        (topLeft[0], topLeft[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print mapped coordinates in the console
            print("[Inference] ArUco marker ID: {}, Mapped Coordinates: {}".format(markerID, mapped_coordinates))

            # Check for ArUco marker with ID 137
            if markerID == 137 and 0 < cX < 1000 and 0 < cY < 1000:
                # Calculate relative coordinates within the rectangle
                relative_coordinates = (cX - topLeft[0], cY - topLeft[1])
                print(topLeft[0], topLeft[1])

                # Display coordinates of the ArUco marker with ID 137
                cv2.putText(image, "ID 137: ({},{})".format(relative_coordinates[0], relative_coordinates[1]),
                            (topLeft[0], topLeft[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID 137 found at Coordinates: ({},{})".format(relative_coordinates[0],
                                                                                             relative_coordinates[1]))

    return image


aruco_type = "DICT_ARUCO_ORIGINAL"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)




def replace_color_image(image, mask, replacement_color):
    result = image.copy()
    result[mask > 0] = replacement_color
    return result


while cap.isOpened():

    ret, img = cap.read()

    h, w, _ = img.shape

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 90, 100])
    upper_yellow = np.array([50, 255, 255])

    mask_yellow = cv2.inRange(HSV, lower_yellow, upper_yellow)

    replacement_color_green = [0, 255, 0]

    img = replace_color_image(img, mask_yellow, replacement_color_green)

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers = aruco_display(corners, ids, rejected, img)

    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
