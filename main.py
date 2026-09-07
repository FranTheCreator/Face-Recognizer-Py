import cv2
import time
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode


captured_frames = None

def print_res(result, output: mp.Image, ts_ms: int):
    first_detection = result.detections[0] if result.detections else None 
    if not first_detection or captured_frames is None: return

    rect_x = first_detection.bounding_box.origin_x
    rect_y = first_detection.bounding_box.origin_y
    rect_width = first_detection.bounding_box.width
    rect_height = first_detection.bounding_box.height

    window_h, window_w = captured_frames.shape[:2]


    for keypoint in first_detection.keypoints:
        circle_x = int(keypoint.x * window_w)
        circle_y = int(keypoint.y * window_h)

        cv2.circle(captured_frames, (circle_x, circle_y), 3, (0, 0, 255), -1)

    
    cv2.rectangle(
        captured_frames,
        (rect_x, rect_y),
        (rect_x + rect_width, rect_y + rect_height),
        (0, 255, 0),
        2
    )
    cv2.imshow("Face Detection", captured_frames)


    if (cv2.waitKey(1) & 0xFF == ord("q")):
        return exit(0)



options = FaceDetectorOptions(
    BaseOptions("./models/blaze_face_short_range.tflite"),
    VisionRunningMode.LIVE_STREAM,
    0.5,
    0.5,
    print_res
)

with FaceDetector.create_from_options(options) as detector: 
    webcam = cv2.VideoCapture(0)

    deviceId = 0
    apiId = cv2.CAP_ANY

    webcam.open(deviceId, apiId)

    if (not (webcam.isOpened())):
        print("Camera not opened")
        exit(0)


    frame = None
    while True: 
        ret, frame = webcam.read()

        if (not ret): 
            print("Blank frame")
            break

        captured_frames = frame
        frame_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(mp.ImageFormat.SRGB, frame)

        detector.detect_async(mp_image, frame_timestamp_ms)
