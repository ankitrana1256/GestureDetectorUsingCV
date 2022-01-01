import math
import cv2
import mediapipe as mp
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

text_color = (255, 255, 255)
circle_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8,
                      min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

finger_status = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}

gestures = {'Victory': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'ThumbsUp': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'Z_alphabet': {'thumb': False, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'Y-alphabet': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': True},
            'Hand': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
            'Volume': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False}}


class process_data:
    def __init__(self, lmlist):
        self.lmlist = lmlist

        if len(self.lmlist) == 21:
            # position for index finger
            self.IFT, self.IFM = position(self.lmlist, 8), position(self.lmlist, 5)
            self.IFD, self.IFP = position(self.lmlist, 7), position(self.lmlist, 6)

            # position for middle finger
            self.MFT, self.MFM = position(self.lmlist, 12), position(self.lmlist, 9)
            self.MFD, self.MFP = position(self.lmlist, 11), position(self.lmlist, 10)

            # position for ring finger
            self.RFT, self.RFM = position(self.lmlist, 16), position(self.lmlist, 13)
            self.RFD, self.RFP = position(self.lmlist, 15), position(self.lmlist, 14)

            # position for pinky finger
            self.PFT, self.PFM = position(self.lmlist, 20), position(self.lmlist, 17)
            self.PFD, self.PFP = position(self.lmlist, 19), position(self.lmlist, 18)

            # position for thumb finger
            self.TFT, self.TFC = position(self.lmlist, 4), position(self.lmlist, 1)
            self.TFM, self.TFI = position(self.lmlist, 2), position(self.lmlist, 3)
            self.WRIST = position(self.lmlist,0)

            self.condition()

    def condition(self):
        # condition for index finger
        if self.IFT and self.IFM and self.IFT[1] < self.IFM[1] and self.IFD[1] < self.IFP[1]:
            finger_status['index'] = True
            cv2.putText(output, "I", (95, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (100, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['index'] = False
            cv2.putText(output, "I", (95, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (100, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for middle finger
        if self.MFT and self.MFM and self.MFT[1] < self.MFM[1] and self.MFD[1] < self.MFP[1]:
            finger_status['middle'] = True
            cv2.putText(output, "M", (130, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (150, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['middle'] = False
            cv2.putText(output, "M", (130, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (150, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for ring finger
        if self.RFT and self.RFM and self.RFT[1] < self.RFM[1] and self.RFD[1] < self.RFP[1]:
            finger_status['ring'] = True
            cv2.putText(output, "R", (190, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (200, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['ring'] = False
            cv2.putText(output, "R", (190, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (200, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for pinky finger
        if self.PFT and self.PFM and self.PFT[1] < self.PFM[1] and self.PFD[1] < self.PFP[1]:
            finger_status['pinky'] = True
            cv2.putText(output, "P", (245, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (250, 490), 10, circle_color, cv2.FILLED)
        else:
            finger_status['pinky'] = False
            cv2.putText(output, "P", (245, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (250, 490), 10, (0, 0, 255), cv2.FILLED)

        # condition for thumb
        if self.TFT[1] >= self.IFM[1] and self.TFT[0] > self.IFM[0]:
            finger_status['thumb'] = False
            cv2.putText(output, "T", (35, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (50, 490), 10, (0, 0, 255), cv2.FILLED)
        else:
            finger_status['thumb'] = True
            cv2.putText(output, "T", (35, 470), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 3)
            cv2.circle(output, (50, 490), 10, circle_color, cv2.FILLED)

        detect = self.find_gesture(finger_status)
        return detect

    def find_gesture(self, val):
        for key, value in gestures.items():
            if val == value:
                self.highlight_finger(val)
                cv2.putText(output, f"Gesture: {key}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, circle_color, 3)
                if key == "Volume":
                    set_volume(self.lmlist)
                return val

    def highlight_finger(self, status):
        if status['index']:
            self.IFT, self.IFM = position(self.lmlist, 8, True), position(self.lmlist, 5, True)
            self.IFD, self.IFP = position(self.lmlist, 7, True), position(self.lmlist, 6, True)
        if status['middle']:
            self.MFM, self.MFT = position(self.lmlist, 9, True), position(self.lmlist, 12, True)
            self.MFP, self.MFD = position(self.lmlist, 10, True), position(self.lmlist, 11, True)
        if status['ring']:
            self.RFM, self.RFT = position(self.lmlist, 13, True), position(self.lmlist, 16, True)
            self.RFP, self.RFD = position(self.lmlist, 14, True), position(self.lmlist, 15, True)
        if status['pinky']:
            self.PFT, self.PFM = position(self.lmlist, 20, True), position(self.lmlist, 17, True)
            self.PFD, self.PFP = position(self.lmlist, 19, True), position(self.lmlist, 18, True)
        if status['thumb']:
            self.TFT, self.TFC = position(self.lmlist, 4, True), position(self.lmlist, 1, True)
            self.TFM, self.TFI = position(self.lmlist, 2, True), position(self.lmlist, 3, True)
            self.WRIST = position(self.lmlist, 0, True)


def position(l, point, mark=False):
    if len(lmlist) == 21:
        x, y = lmlist[point][1], lmlist[point][2]
        if mark:
            cv2.circle(output, (x, y), 10, circle_color, cv2.FILLED)
        return x, y


def set_volume(lmlist):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    IFT, TFT = position(lmlist, 8), position(lmlist, 4)
    cv2.line(output, IFT, TFT, (255, 225, 255), 3)
    distance = math.hypot(TFT[0] - IFT[0], TFT[1] - IFT[1])
    cx, cy = (TFT[0] + IFT[0]) // 2, (TFT[1] + IFT[1]) // 2
    if distance > 30:
        cv2.circle(output, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
    else:
        cv2.circle(output, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    vol = np.interp(distance, [20, 300], [-65, 0])
    volBar = np.interp(distance, [20, 300], [500, 100])
    volume.SetMasterVolumeLevel(vol, None)
    cv2.rectangle(output, (860, 100), (920, 500), (0, 0, 0), 3)
    cv2.rectangle(output, (860, int(volBar)), (920, 500), (255, 255, 255), cv2.FILLED)
    p = (500 - int(volBar)) / 400 * 100
    cv2.putText(output, f"Volume: {int(p)}%", (10, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)


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
                a = process_data(lmlist)

            mpDraw.draw_landmarks(output, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(output, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("Visualiser", output)

    if cv2.waitKey(20) == 27:
        break
