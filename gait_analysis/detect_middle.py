import cv2

cap = cv2.VideoCapture(r'D:\Загрузки Яндекс\моё_церебральный\Cerebral-main\calibration_data\intrinsic\port_0.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Всего кадров: {total}')

# Берём кадр из середины
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

cv2.imwrite('test_middle.png', frame)
print('Сохранён кадр из середины: test_middle.png')

# Пробуем найти маркеры
import cv2
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
corners, ids, _ = detector.detectMarkers(gray)
print(f'DICT_4X4_50 на среднем кадре: {len(ids) if ids is not None else 0} маркеров')