import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
import numpy
import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

dragging = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                z = int(landmark.z * 100)

                if id == 8:
                    cv2.circle(img=frame, center=(x,y), radius=11, color=(0,255,255), thickness=2)
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                    pyautogui.moveTo(index_x, index_y)
                    print(f"ID {id}, X: {index_x}, Y: {index_y}, Z: {z}")

                elif id == 0:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 1:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 2:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 3:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 5:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 6:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 7:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 9:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 10:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 11:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 12:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 13:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 14:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 15:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 16:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 17:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 18:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 19:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)

                elif id == 20:
                    cv2.circle(img=frame, center=(x, y), radius=11, color=(233,123,134), thickness=1)


                elif id == 4:
                    cv2.circle(img=frame, center=(x,y), radius=11, color=(0,255,255), thickness=2)
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y

                    distance = abs(index_y - thumb_y)



                    if distance < 30:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                        else:
                            pyautogui.click()
                            pyautogui.sleep(1)

                    #elif 30 <= distance < 80:
                        #if not dragging:
                            #pyautogui.mouseDown()
                            #dragging = True
                        #pyautogui.moveTo(index_x, index_y)


                    elif distance >= 100 and dragging:
                        pyautogui.mouseUp()
                        dragging = False

    cv2.imshow('virtual Mouse', frame)
    cv2.waitKey(1)

elif id == 0:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 1:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 2:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 3:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 5:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 6:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 7:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 9:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 10:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 11:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 12:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 13:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 14:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 15:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 16:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 17:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 18:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 19:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)

elif id == 20:
cv2.circle(img=frame, center=(x, y), radius=11, color=(233, 123, 134), thickness=1)