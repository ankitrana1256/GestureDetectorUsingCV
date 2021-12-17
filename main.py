import cv2
import mediapipe as mp
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

text_color = (255, 255, 255)
circle_color = (255, 0, 255)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8,
                      min_tracking_confidence=0.5)

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
            cv2.putText(output, f"Gesture: {key}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, circle_color, 3)
            highlight_finger(val)
            return val


def update_finger_status(lmlist):
    if len(lmlist) == 21:

        # position for index finger
        IFT, IFM = position(lmlist, 8), position(lmlist, 5)
        IFD, IFP = position(lmlist, 7), position(lmlist, 6)

        # position for middle finger
        MFT, MFM = position(lmlist, 12), position(lmlist, 9)
        MFD, MFP = position(lmlist, 11), position(lmlist, 10)

        # position for ring finger
        RFT, RFM = position(lmlist, 16), position(lmlist, 13)
        RFD, RFP = position(lmlist, 15), position(lmlist, 14)

        # position for pinky finger
        PFT, PFM = position(lmlist, 20), position(lmlist, 17)
        PFD, PFP = position(lmlist, 19), position(lmlist, 18)

        # position for thumb finger
        TFT, TFC = position(lmlist, 4), position(lmlist, 1)
        TFM, TFI = position(lmlist, 2), position(lmlist, 3)

        # condition for index finger
        if IFT and IFM and IFT[1] < IFM[1] and IFD[1] < IFP[1]:
            finger_status['index'] = True
            cv2.putText(output, "I", (95, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (100, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['index'] = False
            cv2.putText(output, "I", (95, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (100, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition of middle finger
        if MFT and MFM and MFT[1] < MFM[1] and MFD[1] < MFP[1]:
            finger_status['middle'] = True
            cv2.putText(output, "M", (130, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (150, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['middle'] = False
            cv2.putText(output, "M", (130, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (150, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for ring finger
        if RFT and RFM and RFT[1] < RFM[1] and RFD[1] < RFP[1]:
            finger_status['ring'] = True
            cv2.putText(output, "R", (190, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (200, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['ring'] = False
            cv2.putText(output, "R", (190, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (200, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for pinky finger
        if PFT and PFM and PFT[1] < PFM[1] and PFD[1] < PFP[1]:
            finger_status['pinky'] = True
            cv2.putText(output, "P", (245, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (250, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['pinky'] = False
            cv2.putText(output, "P", (245, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (250, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for thumb
        if TFT[1] >= IFM[1] and TFT[0] > IFM[0]:
            finger_status['thumb'] = False
            cv2.putText(output, "T", (35, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (50, 490), 10, (0, 0, 255), cv2.FILLED)
        else:
            finger_status['thumb'] = True
            cv2.putText(output, "T", (35, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (50, 490), 10, circle_color, cv2.FILLED)

        detect = find_gesture(finger_status)


def position(l, point, mark=False):
    if len(lmlist) == 21:
        x, y = lmlist[point][1], lmlist[point][2]
        if mark == True:
            cv2.circle(output, (x, y), 10, circle_color, cv2.FILLED)
        return x, y


def highlight_finger(status):
    if status['index']:
        IFT, IFM = position(lmlist, 8, True), position(lmlist, 5, True)
        IFD, IFP = position(lmlist, 7, True), position(lmlist, 6, True)
    if status['middle']:
        MFM, MFT = position(lmlist, 9, True), position(lmlist, 12, True)
        MFP, MFD = position(lmlist, 10, True), position(lmlist, 11, True)
    if status['ring']:
        RFM, RFT = position(lmlist, 13, True), position(lmlist, 16, True)
        RFP, RFD = position(lmlist, 14, True), position(lmlist, 15, True)
    if status['pinky']:
        PFT, PFM = position(lmlist, 20, True), position(lmlist, 17, True)
        PFD, PFP = position(lmlist, 19, True), position(lmlist, 18, True)
    if status['thumb']:
        TFT, TFC = position(lmlist, 4, True), position(lmlist, 1, True)
        TFM, TFI = position(lmlist, 2, True), position(lmlist, 3, True)
        WRIST = position(lmlist, 0, True)


while True:
    success, img = cap.read()
    output = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rotatedImg = cv2.flip(imgRGB, 1)
    result = hands.process(rotatedImg)
    lmlist = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
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
    cv2.imshow("Visualiser", output)

    if cv2.waitKey(20) == 27:
        break
