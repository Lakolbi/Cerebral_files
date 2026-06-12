"""
Извлечение 2D координат суставов из видеофайлов с помощью MediaPipe Pose.

Использование:
    python pose_estimator.py --session ./my_session --model SIMPLE_HOLISTIC --out ./my_session/SIMPLE_HOLISTIC

Ожидаемая структура session:
    my_session/
        port_0.mp4
        port_1.mp4
        port_2.mp4

Результат сохраняется в:
    my_session/SIMPLE_HOLISTIC/
        xy_SIMPLE_HOLISTIC.csv   <- 2D детекции в формате Caliscope

Формат выходного CSV:
    port, frame_index, point_id, img_loc_x, img_loc_y
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

# Точки MediaPipe Pose которые используются в системе (нижняя часть тела + плечи)
GAIT_POINT_IDS = {
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Параметры детектора MediaPipe
# min_detection_confidence — минимальная уверенность для первичного обнаружения человека
# min_tracking_confidence  — минимальная уверенность для трекинга между кадрами
# model_complexity         — сложность модели: 0 (быстро), 1 (баланс), 2 (точно)
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE  = 0.5
MODEL_COMPLEXITY     = 1


def process_video(video_path: str, port: int) -> list[dict]:
    """
    Запускает MediaPipe Pose на одном видеофайле и извлекает 2D координаты
    суставов для каждого кадра.

    Алгоритм на каждом кадре:
    1. Конвертирует BGR -> RGB (MediaPipe требует RGB).
    2. Запускает детектор позы.
    3. Если поза найдена — считывает нормализованные координаты (0..1)
       и переводит их в пиксели умножением на ширину/высоту кадра.
    4. Сохраняет только точки из GAIT_POINT_IDS.
    5. Если поза не найдена — пропускает кадр (точки не записываются,
       что даст NaN при последующей обработке).

    Возвращает список строк для итогового CSV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть видео: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"  port_{port}: {width}x{height}, {fps:.1f} fps, {total} кадров")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    rows = []
    frame_idx = 0
    detected = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe требует RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            detected += 1
            landmarks = result.pose_landmarks.landmark
            for point_id in GAIT_POINT_IDS:
                lm = landmarks[point_id]
                # visibility — уверенность что точка видна (0..1)
                # Пропускаем точки с низкой уверенностью
                if lm.visibility < 0.3:
                    continue
                rows.append({
                    "port":        port,
                    "frame_index": frame_idx,
                    "point_id":    point_id,
                    "img_loc_x":   round(lm.x * width,  2),
                    "img_loc_y":   round(lm.y * height, 2),
                    "visibility":  round(lm.visibility, 3),
                })

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    кадр {frame_idx}/{total}...")

    cap.release()
    pose.close()

    pct = 100 * detected / frame_idx if frame_idx > 0 else 0
    print(f"  port_{port}: поза найдена в {detected}/{frame_idx} кадрах ({pct:.1f}%)")
    return rows


def run_pose_estimation(session_dir: str, out_dir: str, ports: list[int] = None) -> pd.DataFrame:
    """
    Запускает детекцию позы для всех камер сессии.

    session_dir : папка с видеофайлами port_0.mp4, port_1.mp4, port_2.mp4
    out_dir     : папка для сохранения результата
    ports       : список номеров камер (по умолчанию [0, 1, 2])

    Возвращает итоговый DataFrame и сохраняет xy_SIMPLE_HOLISTIC.csv.
    """
    if ports is None:
        ports = []
        for p in sorted(Path(session_dir).glob("port_*.mp4")):
            num = int(p.stem.split("_")[1])
            ports.append(num)
        if not ports:
            raise FileNotFoundError(
                f"Не найдено видеофайлов port_*.mp4 в папке {session_dir}"
            )

    print(f"Найдено камер: {len(ports)} -> {ports}")

    all_rows = []
    for port in ports:
        video_path = Path(session_dir) / f"port_{port}.mp4"
        if not video_path.exists():
            print(f"  ВНИМАНИЕ: файл не найден: {video_path}, пропускаю")
            continue
        print(f"\nОбработка port_{port}...")
        rows = process_video(str(video_path), port)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("Не удалось извлечь ни одной точки ни с одной камеры.")

    df = pd.DataFrame(all_rows)

    # Сохраняем
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "xy_SIMPLE_HOLISTIC.csv"
    df.to_csv(out_path, index=False)

    total_frames = df["frame_index"].nunique()
    print(f"\nГотово: {len(df)} строк, {total_frames} уникальных кадров")
    print(f"Сохранено в {out_path}")
    return df


def check_detection_quality(df: pd.DataFrame):
    """Выводит сводку качества детекции по каждой камере и точке."""
    print("\n=== Качество детекции ===")
    for port in sorted(df["port"].unique()):
        sub = df[df["port"] == port]
        n_frames = sub["frame_index"].nunique()
        print(f"\nКамера port_{port}: {n_frames} кадров с детекцией")
        for pid, name in GAIT_POINT_IDS.items():
            pts = sub[sub["point_id"] == pid]
            if "visibility" in pts.columns:
                mean_vis = pts["visibility"].mean()
                coverage = 100 * len(pts) / n_frames if n_frames > 0 else 0
                status = "OK" if coverage > 70 else "ПЛОХО"
                print(f"  {name:20s}: покрытие {coverage:5.1f}%, "
                      f"видимость {mean_vis:.2f}  [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение 2D координат суставов из видео (MediaPipe Pose)"
    )
    parser.add_argument("--session", required=True,
                        help="Папка с видеофайлами port_0.mp4 / port_1.mp4 / port_2.mp4")
    parser.add_argument("--out", required=True,
                        help="Папка для сохранения xy_SIMPLE_HOLISTIC.csv")
    parser.add_argument("--ports", default=None,
                        help="Номера камер через запятую, например 0,1,2 (по умолчанию — все)")
    parser.add_argument("--detection-conf", type=float, default=DETECTION_CONFIDENCE,
                        help=f"Порог уверенности детекции (по умолчанию {DETECTION_CONFIDENCE})")
    parser.add_argument("--tracking-conf", type=float, default=TRACKING_CONFIDENCE,
                        help=f"Порог уверенности трекинга (по умолчанию {TRACKING_CONFIDENCE})")
    parser.add_argument("--complexity", type=int, default=MODEL_COMPLEXITY, choices=[0, 1, 2],
                        help=f"Сложность модели MediaPipe: 0/1/2 (по умолчанию {MODEL_COMPLEXITY})")
    args = parser.parse_args()

    global DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, MODEL_COMPLEXITY
    DETECTION_CONFIDENCE = args.detection_conf
    TRACKING_CONFIDENCE  = args.tracking_conf
    MODEL_COMPLEXITY     = args.complexity

    ports = [int(p) for p in args.ports.split(",")] if args.ports else None

    df = run_pose_estimation(args.session, args.out, ports)
    check_detection_quality(df)


if __name__ == "__main__":
    main()
