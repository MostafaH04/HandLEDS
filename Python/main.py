# Imports necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import pyfirmata as firm
import time


board = firm.Arduino('COM3')

hands = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7, min_tracking_confidence= 0.7) #Uses default confidence tracking and detection arguments
drawings = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0) # Gets Camera feed

# While the camera is opened, this checks for any hands detected in the video feed
while cam.isOpened():
    ret, img = cam.read() # Gets a fram from the camera feed and saves it to the variable img
    prevFingNum = 0
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converts the frame to RGB from BGR

    results = hands.process(imgRGB) # Stores results in variable results after running mediapipe on the RGB frame

    handAnnotations = results.multi_hand_landmarks # stores the hand landmarks

    fing = [[None,None],[None,None],[None,None],[None,None],[None,None]]

    if handAnnotations: 
        for handLandmarks in handAnnotations:
            
            fingNum = 5
            fing[0][0] = handLandmarks.landmark[4].x*img.shape[1]
            fing[0][1] = handLandmarks.landmark[3].x*img.shape[1]

            fing[1][0] = handLandmarks.landmark[8].y*img.shape[0]
            fing[1][1] = handLandmarks.landmark[7].y*img.shape[0]

            fing[2][0] = handLandmarks.landmark[12].y*img.shape[0]
            fing[2][1] = handLandmarks.landmark[11].y*img.shape[0]

            fing[3][0] = handLandmarks.landmark[16].y*img.shape[0]
            fing[3][1] = handLandmarks.landmark[15].y*img.shape[0]

            fing[4][0] = handLandmarks.landmark[20].y*img.shape[0]
            fing[4][1] = handLandmarks.landmark[19].y*img.shape[0]            

            for i in fing[1:]:
                if i[0] > i[1]:
                    fingNum -= 1
            
            if handLandmarks.landmark[1].x*img.shape[1] <  handLandmarks.landmark[0].x*img.shape[1]: #checks if this is the right hand (ish)
                if fing[0][0] > fing[0][1]:
                    fingNum -=1
            else: #its the left hand (ish)
                if fing[0][0] < fing[0][1]:
                    fingNum -=1

            
            if fingNum != prevFingNum:
                for i in range(5):
                    board.digital[i+3].write(0)
                board.digital[fingNum+2].write(1)

            prevFingNum = fingNum
            
            cv2.putText(img, str(fingNum), (20,50), cv2.FONT_HERSHEY_PLAIN, 4, color = (255,20,20))
            # Adds the hand makers onto the image if they exist
            drawings.draw_landmarks(img, handLandmarks, mp.solutions.hands.HAND_CONNECTIONS)
    else:
        for i in range(5):
            board.digital[i+3].write(0)
            
    # Displays the image in a window labeled Output
    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break

#when the program is done, this stops using the Camera feed
cam.release()
