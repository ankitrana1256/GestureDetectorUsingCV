import cv2
import mediapipe as mp
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
print(volume.GetVolumeRange())

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


def position(l, point):
    if len(lmlist) == 20:
        x1, y1 = lmlist[point][1], lmlist[point][2]
        cv2.circle(output, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        return x1, y1

while True:
    success, img = cap.read()
    output = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rotatedImg = cv2.flip(imgRGB, 1)
    result1 = hands.process(rotatedImg)
    lmlist = []

    if result1.multi_hand_landmarks:
        for handLms in result1.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                x1 = position(lmlist, 4)
                x2 = position(lmlist, 8)
                cv2.line(output, x1, x2, (255, 0, 0), 3)
                if x1 and x2:
                    lcx = ((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)**1/2
                    point1, point2 = (x2[0]+x1[0])//2, (x2[1]+x1[1])//2
                    cv2.circle(output, (point1, point2), 10, (255, 0, 255), cv2.FILLED)
                    print("Distance :", lcx)
                    vol = np.interp(lcx, [12,29000], [-65.25, 0.0])
                    volume.SetMasterVolumeLevel(vol, None)
                    if lcx < 400:
                        cv2.circle(output, (point1, point2), 10, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(output, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(output, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Assisstant", output)

    if cv2.waitKey(20) == 27:
        break