# Читаем файл
with open('calibration.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Исправляем параметры доски
text = text.replace('BOARD_COLS = 4', 'BOARD_COLS = 3')
text = text.replace('BOARD_ROWS = 3', 'BOARD_ROWS = 4')

# Записываем обратно
with open('calibration.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Готово! BOARD_COLS=3, BOARD_ROWS=4')