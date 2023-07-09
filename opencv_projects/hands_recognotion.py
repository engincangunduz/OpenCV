import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# for leave the space
offset = 20 
imgSize = 300

folder = "Data/I Love You"
counter = 0

while True:
    ret, frame = cap.read()
    # kamerayı düz yapmak için
    # frame = cv2.flip(frame,1)
    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"] #bounding box kısaltması

        # np.ones ile 1'den oluşan matris döndürür
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = frame[y - offset :y + h + offset, x - offset: x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w #en boy oranı

        if aspectRatio >1:
            k = imgSize/h # boyu uzatıcağız gericez
            wCal = math.ceil(k*w) # genişlik hesaplama maksimum alması için ceil
        
            imgResize = cv2.resize(imgCrop,(wCal, imgSize)) # yeniden boyutlandırmak için ve height sabit
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize/w # boyu uzatıcağız gericez
            hCal = math.ceil(k * h) # genişlik hesaplama maksimum alması için ceil
            imgResize = cv2.resize(imgCrop,(imgSize, hCal)) # bu sefer width sabit 
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize 

        cv2.imshow("ImageCrop", imgCrop) #for crop the image
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()