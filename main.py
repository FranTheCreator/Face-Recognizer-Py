import cv2
import os
import turtle
import numpy as np
import asyncio



def main():
    webcam = cv2.VideoCapture(0)

    deviceId = 0
    apiId = cv2.CAP_ANY

    webcam.open(deviceId, apiId)


    if (not (webcam.isOpened())):
        print("Camera not opened");
        return

    frame = None
    while True: 
        ret, frame = webcam.read()

        if (not ret): 
            print("Blank frame")
            break
            
        cv2.imshow("Live", frame)

        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break



if (__name__ == "__main__"): main()
