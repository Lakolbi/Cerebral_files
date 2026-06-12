import cv2
import numpy as np

# Читаем первый кадр
cap = cv2.VideoCapture(r'D:\Загрузки Яндекс\моё_церебральный\Cerebral-main\calibration_data\intrinsic\port_0.mp4')
ret, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Проверяем все словари
dicts_to_try = [
    (cv2.aruco.DICT_4X4_50, "DICT_4X4_50"),
    (cv2.aruco.DICT_4X4_100, "DICT_4X4_100"),
    (cv2.aruco.DICT_4X4_250, "DICT_4X4_250"),
    (cv2.aruco.DICT_5X5_50, "DICT_5X5_50"),
    (cv2.aruco.DICT_5X5_100, "DICT_5X5_100"),
    (cv2.aruco.DICT_6X6_50, "DICT_6X6_50"),
    (cv2.aruco.DICT_6X6_100, "DICT_6X6_100"),
]

params = cv2.aruco.DetectorParameters()
for dict_id, dict_name in dicts_to_try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    n = len(ids) if ids is not None else 0
    print(f"{dict_name}: найдено маркеров = {n}")
    if n > 0:
        print(f"  --> IDs: {ids.flatten().tolist()}")