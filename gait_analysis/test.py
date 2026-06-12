import cv2
cap = cv2.VideoCapture(r'D:\Загрузки Яндекс\моё_церебральный\Cerebral-main\calibration_data\intrinsic\port_0.mp4')
ret, frame = cap.read()
print('Открыто:', ret)
print('Размер:', frame.shape if ret else 'нет')
cv2.imwrite('test_frame.png', frame)
cap.release()
print('Кадр сохранён в test_frame.png')