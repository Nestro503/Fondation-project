import numpy as np
import time
import cv2

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Constantes
DISTANCE_COIN_ARUCO_X_MM = 2940
DISTANCE_COIN_Y_MM = 1940




# Function to calculate mm_nbr_pixel and pixel_nbr_mm
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
            x_aruco_0 = corners[index_aruco_0[0]][0][0][0]  # Use the top-left corner
            y_aruco_0 = corners[index_aruco_0[0]][0][0][1]
            print("x_aruco_0:{}, y_aruco_0:{}".format(x_aruco_0, y_aruco_0))

        # Check if marker 1 is detected
        if index_aruco_1.size > 0:
            x_aruco_1 = corners[index_aruco_1[0]][0][1][0]  # Use the top-right corner
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
            print("[Inference] Warning: pixel_distance_x is zero. Cannot calculate mm_nbr_pixel_x and pixel_nbr_mm_x.")
            return None, None, None, None, None

        # Calculate mm_nbr_pixel and pixel_nbr_mm for both x and y directions
        mm_nbr_pixel_x = float(pixel_distance_x) / DISTANCE_COIN_ARUCO_X_MM
        pixel_nbr_mm_x = float(DISTANCE_COIN_ARUCO_X_MM) / pixel_distance_x

        print(
            "[Inference] pixeldistance: {:.2f}, mm_nbr_pixel_x: {:.2f}, pixel_nbr_mm_x: {:.2f}".format(pixel_distance_x,
                                                                                                       mm_nbr_pixel_x,
                                                                                                       pixel_nbr_mm_x))

    return pixel_distance_x, mm_nbr_pixel_x, pixel_nbr_mm_x, x_aruco_0, y_aruco_0, x_aruco_1, y_aruco_1, x_aruco_2, y_aruco_2, x_aruco_3, y_aruco_3


# Function to display ArUco markers and calibration circles
def aruco_display(corners, ids, image):
    if len(corners) > 0:
        ids = ids.flatten()

        pixel_distance_x, mm_nbr_pixel, pixel_nbr_mm, x_aruco_0, y_aruco_0, x_aruco_1, y_aruco_1, x_aruco_2, y_aruco_2, x_aruco_3, y_aruco_3 = calculate_pixel_distance(
            corners, ids)

        if pixel_distance_x is not None and mm_nbr_pixel is not None:
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

                if mm_nbr_pixel is not None:
                    ####################Dessins calib couleurs !!
                    '''
                    # Dessin calib jaune
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 180 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 180 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1) 
                    # Dessin calib cyan
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 270 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 270 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib vert
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 360 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 360 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib magenta
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 450 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 450 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib rouge
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 540 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 540 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib bleu
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 630 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_3 + 50 * mm_nbr_pixel + 630 * mm_nbr_pixel), int(y_aruco_3 + 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)


                    ####################Dessins calib blancs !!
                    # Dessin calib blanc 0
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 90 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 90 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 1
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 180 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 180 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 2
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 270 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 270 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 3
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 360 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 360 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 4
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 450 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 450 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 5
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 540 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 540 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 6
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 630 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 630 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 7
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 720 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 720 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 8
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 810 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 810 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)
                    # Dessin calib blanc 9
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 900 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 4,
                               (255, 255, 255), -1)  # -1 pour remplir le cercle
                    cv2.circle(image, (
                        int(x_aruco_1 - 50 * mm_nbr_pixel - 900 * mm_nbr_pixel), int(y_aruco_1 - 10 * mm_nbr_pixel)), 2,
                               (0, 0, 0), -1)

                    '''

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

cap = cv2.VideoCapture("model.jpg")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while True:

    ret, img = cap.read()

    # Vérifier si la lecture de l'image a réussi
    if not ret:
        time.sleep(0.5)  # Pause d'une demi-seconde
        print("Reprise après pause.")
        continue  # Continue à la prochaine itération de la boucle

    h, w, _ = img.shape
    print("Lecture d'image réussie. Shape: {}".format((h, w)))



    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(img)
    detected_markers = aruco_display(corners, ids, img)


    # Perform the conversion and print results
    pixel_distance_x, mm_nbr_pixel_x, pixel_nbr_mm_x, x_aruco_0, y_aruco_0, x_aruco_1, y_aruco_1, x_aruco_2, y_aruco_2, x_aruco_3, y_aruco_3 = calculate_pixel_distance(
        corners, ids)
    if pixel_distance_x is not None and mm_nbr_pixel_x is not None:
        print(
            "[Inference] pixeldistance: {:.2f}, mm_nbr_pixel_x: {:.2f}, pixel_nbr_mm_x: {:.2f}".format(pixel_distance_x,
                                                                                                       mm_nbr_pixel_x,
                                                                                                       pixel_nbr_mm_x))






    # Coordonnées des pixels à vérifier
    pixel_jaune = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 180 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_jaune = img[pixel_jaune[1], pixel_jaune[0]]
    # Dessinez un cercle avec la couleur du pixel
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 180 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_jaune[0]), int(bgr_pixel_jaune[1]), int(bgr_pixel_jaune[2])), -1)

    pixel_cyan = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 270 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_cyan = img[pixel_cyan[1], pixel_cyan[0]]
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 270 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_cyan[0]), int(bgr_pixel_cyan[1]), int(bgr_pixel_cyan[2])), -1)

    pixel_vert = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 360 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_vert = img[pixel_vert[1], pixel_vert[0]]
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 360 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_vert[0]), int(bgr_pixel_vert[1]), int(bgr_pixel_vert[2])), -1)

    pixel_magenta = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 450 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_magenta = img[pixel_magenta[1], pixel_magenta[0]]
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 450 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_magenta[0]), int(bgr_pixel_magenta[1]), int(bgr_pixel_magenta[2])), -1)

    pixel_rouge = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 540 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_rouge = img[pixel_rouge[1], pixel_rouge[0]]
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 540 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_rouge[0]), int(bgr_pixel_rouge[1]), int(bgr_pixel_rouge[2])), -1)

    pixel_bleu = (
        int(x_aruco_3 + 50 * mm_nbr_pixel_x + 630 * mm_nbr_pixel_x),
        int(y_aruco_3 + 10 * mm_nbr_pixel_x)
    )
    # Obtenez la valeur BGR du pixel
    bgr_pixel_bleu = img[pixel_bleu[1], pixel_bleu[0]]
    cv2.circle(img, (int(x_aruco_3 + 50 * mm_nbr_pixel_x + 630 * mm_nbr_pixel_x), int(y_aruco_3 + 70 * mm_nbr_pixel_x)),
               10,
               (int(bgr_pixel_bleu[0]), int(bgr_pixel_bleu[1]), int(bgr_pixel_bleu[2])), -1)

    cv2.imshow("Original Image", detected_markers)



    # Définir la plage de tolérance pour la couleur de référence
    tolerance = 20
    # Créer un masque pour la couleur de référence
    lower_bound = np.array(bgr_pixel_jaune) - tolerance
    upper_bound = np.array(bgr_pixel_jaune) + tolerance
    mask = cv2.inRange(detected_markers, lower_bound, upper_bound)
    # Remplacer la couleur de référence par du noir
    detected_markers[mask > 0] = [0, 0, 0]

    # Afficher l'image modifiée
    cv2.imshow("Image avec couleur de référence remplacée par du noir", detected_markers)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
    cap.release()

