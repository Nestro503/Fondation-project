
import numpy as np
import time
import cv2

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


#Constantes
DISTANCE_COIN_ARUCO_X_CM = 294
DISTANCE_COIN_Y_CM = 194

#LONGEUR_CALIB_CM = 9



# Function to calculate cm_nbr_pixel and pixel_nbr_cm
def calculate_pixel_distance(corners, ids):
    if len(corners) > 1:
        index_aruco_0 = np.where(ids == 0)[0]
        index_aruco_1 = np.where(ids == 1)[0]
        index_aruco_2 = np.where(ids == 2)[0]
        index_aruco_3 = np.where(ids == 3)[0]

        # Initialize default values
        x_aruco_0, y_aruco_0 = 0, 0
        x_aruco_1, y_aruco_1 = 0, 0
        x_aruco_2, y_aruco_2 = 0, 0
        x_aruco_3, y_aruco_3 = 0, 0

        # Check if marker 0 is detected
        if index_aruco_0.size > 0:
            x_aruco_0 = corners[index_aruco_0[0]][0][0][0] # Use the top-left corner
            y_aruco_0 = corners[index_aruco_0[0]][0][0][1]
            print("x_aruco_0:{}, y_aruco_0:{}".format(x_aruco_0, y_aruco_0))

        # Check if marker 1 is detected
        if index_aruco_1.size > 0:
            x_aruco_1 = corners[index_aruco_1[0]][0][1][0] # Use the top-right corner
            y_aruco_1 = corners[index_aruco_1[0]][0][1][1]
            print("x_aruco_1:{}, y_aruco_1:{}".format(x_aruco_1, y_aruco_1))

        # Check if marker 2 is detected
        if index_aruco_2.size > 0:
            x_aruco_2 = corners[index_aruco_2[0]][0][2][0]  # Use the bottom-right corner
            y_aruco_2 = corners[index_aruco_2[0]][0][2][1]
            print("x_aruco_2:{}, y_aruco_2:{}".format(x_aruco_2, y_aruco_2))

        # Check if marker 3 is detected
        if index_aruco_3.size > 0:
            x_aruco_3 = corners[index_aruco_3[0]][0][3][0]  # Use the bottom-left corner
            y_aruco_3 = corners[index_aruco_3[0]][0][3][1]
            print("x_aruco_3:{}, y_aruco_3:{}".format(x_aruco_3, y_aruco_3))

        # Calculate the pixel distance in both x and y directions
        pixel_distance_x = abs(x_aruco_2 - x_aruco_0)

        # Check if pixel_distance_x is zero to avoid division by zero
        if pixel_distance_x == 0:
            print("[Inference] Warning: pixel_distance_x is zero. Cannot calculate cm_nbr_pixel_x and pixel_nbr_cm_x.")
            return None, None, None, None, None

        # Calculate cm_nbr_pixel and pixel_nbr_cm for both x and y directions
        cm_nbr_pixel_x = float(pixel_distance_x) / DISTANCE_COIN_ARUCO_X_CM
        pixel_nbr_cm_x = float(DISTANCE_COIN_ARUCO_X_CM) / pixel_distance_x

        print("[Inference] pixeldistance: {}, cm_nbr_pixel_x: {}, pixel_nbr_cm_x: {}".format(pixel_distance_x, cm_nbr_pixel_x, pixel_nbr_cm_x))

        return pixel_distance_x, cm_nbr_pixel_x, pixel_nbr_cm_x, x_aruco_0, y_aruco_0, x_aruco_1, y_aruco_1, x_aruco_2, y_aruco_2, x_aruco_3, y_aruco_3

    return None, None, None, None, None, None, None, None, None, None



def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()

        pixel_distance_x, cm_nbr_pixel, pixel_nbr_cm, x_aruco_0, y_aruco_0, x_aruco_1, y_aruco_1, x_aruco_2, y_aruco_2, x_aruco_3, y_aruco_3 = calculate_pixel_distance(corners, ids)

        if pixel_distance_x is not None and cm_nbr_pixel is not None:
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

                # Map ArUco marker coordinates to game card coordinates
                if markerID == 0:
                    mapped_coordinates = (topLeft[0], topLeft[1])
                    cv2.circle(image, mapped_coordinates, 4, (0, 0, 255), -1)
                elif markerID == 1:
                    mapped_coordinates = (topRight[0], topRight[1])
                    cv2.circle(image, mapped_coordinates, 4, (0, 0, 255), -1)
                elif markerID == 2:
                    mapped_coordinates = (bottomRight[0], bottomRight[1])
                    cv2.circle(image, mapped_coordinates, 4, (0, 0, 255), -1)
                elif markerID == 3:
                    mapped_coordinates = (bottomLeft[0], bottomLeft[1])
                    cv2.circle(image, mapped_coordinates, 4, (0, 0, 255), -1)
                else:
                    mapped_coordinates = (cX, cY)
                    cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                if cm_nbr_pixel is not None:

                    # Dessin calib blanc
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 9 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 9 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)


                    # Dessin calib jaune
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 18 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 18 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib cyan
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 27 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 27 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib vert
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 36 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 36 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib magenta
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 45 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 45 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib rouge
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 54 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 54 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib bleu
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 63 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 63 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    # Dessin calib noir
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 72 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                    int(x_aruco_3 + 5 * cm_nbr_pixel + 72 * cm_nbr_pixel), int(y_aruco_3 + 2 * cm_nbr_pixel)), 2,
                               (0, 0, 0), -1)





                cv2.putText(image, "ID: {}".format(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)



                # Display mapped coordinates on the image
                cv2.putText(image, "Mapped: ({},{})".format(mapped_coordinates[0], mapped_coordinates[1]),
                            (topLeft[0], topLeft[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Print mapped coordinates in the console
                print("[Inference] ArUco marker ID: {}, Mapped Coordinates: {}".format(markerID, mapped_coordinates))

    return image


aruco_type = "DICT_ARUCO_ORIGINAL"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)




def replace_color_image(image, mask, replacement_color):
    result = image.copy()
    result[mask > 0] = replacement_color
    return result


while cap.isOpened():

    ret, img = cap.read()

    h, w, _ = img.shape

    #HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #lower_yellow = np.array([20, 90, 100])
    #upper_yellow = np.array([50, 255, 255])

    #mask_yellow = cv2.inRange(HSV, lower_yellow, upper_yellow)

    #replacement_color_green = [0, 255, 0]

    #img = replace_color_image(img, mask_yellow, replacement_color_green)

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(img)

    # Call the aruco_display function
    detected_markers = aruco_display(corners, ids, rejected, img)

    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()
cap.release()


