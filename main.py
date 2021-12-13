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

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

finger_status = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}

gestures = {'Victory': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'ThumbsUp': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'Z_alphabet': {'thumb': False, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'Y-alphabet': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': True},
            'Hand': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}}


def find_gesture(val):
    for key, value in gestures.items():
        if val == value:
            cv2.putText(output, f"Gesture: {key}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


def update_finger_status(lmlist):
    if len(lmlist) == 21:

        # position for index finger
        x1, x2 = position(lmlist, 8), position(lmlist, 5)
        x3, x4 = position(lmlist, 7), position(lmlist, 6)

        # position for middle finger
        y1, y2 = position(lmlist, 12), position(lmlist, 9)
        y3, y4 = position(lmlist, 11), position(lmlist, 10)

        # position for ring finger
        z1, z2 = position(lmlist, 16), position(lmlist, 13)
        z3, z4 = position(lmlist, 15), position(lmlist, 14)

        # position for pinky finger
        p1, p2 = position(lmlist, 20), position(lmlist, 17)
        p3, p4 = position(lmlist, 19), position(lmlist, 18)

        # position for pinky finger
        t1, t2 = position(lmlist, 4), position(lmlist, 1)
        # t3, t4 = position(lmlist, 2), position(lmlist, 1)

        # condition for index finger
        if x1 and x2 and x1[1] < x2[1] and x3[1] < x4[1]:
            finger_status['index'] = True
        else:
            finger_status['index'] = False

        # condition of middle finger
        if y1 and y2 and y1[1] < y2[1] and y3[1] < y4[1]:
            finger_status['middle'] = True
        else:
            finger_status['middle'] = False

        # condition for ring finger
        if z1 and z2 and z1[1] < z2[1] and z3[1] < z4[1]:
            finger_status['ring'] = True
        else:
            finger_status['ring'] = False

        # condition for pinky finger
        if p1 and p2 and p1[1] < p2[1] and p3[1] < p4[1]:
            finger_status['pinky'] = True
        else:
            finger_status['pinky'] = False

        if t1[1] >= x2[1]:
            finger_status['thumb'] = False
        else:
            finger_status['thumb'] = True

        detect = find_gesture(finger_status)


def position(l, point):
    if len(lmlist) == 21:
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

                # detecting different gestures
                update_finger_status(lmlist)

            mpDraw.draw_landmarks(output, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(output, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Assisstant", output)

    if cv2.waitKey(20) == 27:
        break