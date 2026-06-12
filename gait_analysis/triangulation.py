"""
Триангуляция 3D координат суставов из 2D детекций с 3 камер.

Использование:
    python triangulation.py \
        --session ./my_session \
        --calib calibration_result.json \
        --model SIMPLE_HOLISTIC \
        --out xyz_triangulated.csv

Ожидаемая структура session:
    my_session/
        SIMPLE_HOLISTIC/
            xy_SIMPLE_HOLISTIC.csv    <- 2D детекции от Caliscope

Формат xy_SIMPLE_HOLISTIC.csv:
    port, frame_index, point_id, img_loc_x, img_loc_y

Результат сохраняется в том же формате что xyz_SIMPLE_HOLISTIC.csv:
    sync_index, point_id, x_coord, y_coord, z_coord
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import load_calibration

# Минимум камер которые должны видеть точку чтобы триангулировать
MIN_CAMERAS = 2

# MediaPipe номера точек которые нас интересуют (нижняя часть тела + плечи)
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


def build_projection_matrices(cameras: list) -> list:
    """
    Строит матрицы проекции P = K @ [R | T] для каждой камеры.
    P (3x4) переводит 3D точку в гомогенные 2D координаты.
    """
    projections = []
    for cam in cameras:
        K = np.array(cam["camera_matrix"])   # 3x3
        R = np.array(cam["R"])               # 3x3
        T = np.array(cam["T"]).reshape(3, 1) # 3x1
        RT = np.hstack([R, T])               # 3x4
        P = K @ RT                           # 3x4
        projections.append(P)
    return projections


def triangulate_dlt(projections: list, points_2d: list) -> np.ndarray:
    """
    Метод DLT (Direct Linear Transform) — самый простой и надёжный способ
    триангуляции из N камер.

    Для каждой камеры i и точки (u, v) составляем два уравнения:
        u * (P[2] @ X) = P[0] @ X
        v * (P[2] @ X) = P[1] @ X

    Получаем систему AX = 0, решаем через SVD.

    projections : список матриц P (3x4), по одной на камеру
    points_2d   : список (u, v) — 2D координаты на каждой камере
                  None если камера не видит точку

    Возвращает (x, y, z) в метрах или (nan, nan, nan) если не хватает камер.
    """
    rows = []
    for P, pt in zip(projections, points_2d):
        if pt is None:
            continue
        u, v = pt
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    if len(rows) < 4:  # нужно минимум 4 строки (2 камеры по 2 уравнения)
        return np.array([np.nan, np.nan, np.nan])

    A = np.vstack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]           # последняя строка Vt — решение
    X = X / X[3]         # делим на гомогенную координату
    return X[:3]


def undistort_points(points_px: dict, cameras: list) -> dict:
    """
    Корректирует дисторсию линзы для 2D точек перед триангуляцией.

    points_px : {port: {point_id: (u, v)}}
    Возвращает тот же формат с исправленными координатами.
    """
    corrected = {}
    for cam in cameras:
        port = cam["port"]
        if port not in points_px:
            continue
        K = np.array(cam["camera_matrix"])
        D = np.array(cam["dist_coeffs"])
        pts = points_px[port]
        if not pts:
            corrected[port] = {}
            continue

        # Собираем точки в массив для cv2.undistortPoints
        import cv2
        ids = list(pts.keys())
        arr = np.array([[pts[i]] for i in ids], dtype=np.float32)  # (N,1,2)
        undist = cv2.undistortPoints(arr, K, D, P=K)               # (N,1,2)
        corrected[port] = {i: (float(undist[j, 0, 0]), float(undist[j, 0, 1]))
                           for j, i in enumerate(ids)}
    return corrected


def load_2d_detections(session_dir: str, model: str) -> pd.DataFrame:
    """Загружает CSV с 2D детекциями от Caliscope."""
    path = Path(session_dir) / model / f"xy_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Файл 2D детекций не найден: {path}")
    df = pd.read_csv(path)
    print(f"Загружено 2D детекций: {len(df)} строк, "
          f"кадров: {df['frame_index'].nunique()}, "
          f"камер: {df['port'].nunique()}")
    return df


def triangulate_session(xy_df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    Основная функция: для каждого кадра и каждой точки собирает
    наблюдения со всех камер и триангулирует 3D координату.

    Возвращает DataFrame в формате xyz_<model>_labelled.csv.
    """
    cameras = calib["cameras"]
    projections = build_projection_matrices(cameras)
    ports = [cam["port"] for cam in cameras]

    frames = sorted(xy_df["frame_index"].unique())
    point_ids = sorted(xy_df["point_id"].unique())
    # Оставляем только нужные точки
    point_ids = [p for p in point_ids if p in GAIT_POINT_IDS]

    results = []
    n_total = 0
    n_ok = 0

    for frame_idx in frames:
        frame_rows = xy_df[xy_df["frame_index"] == frame_idx]

        # Собираем 2D точки по камерам: {port: {point_id: (u, v)}}
        points_px = {}
        for _, row in frame_rows.iterrows():
            port = int(row["port"])
            pid = int(row["point_id"])
            if pid not in GAIT_POINT_IDS:
                continue
            x, y = row["img_loc_x"], row["img_loc_y"]
            if pd.isna(x) or pd.isna(y):
                continue
            points_px.setdefault(port, {})[pid] = (float(x), float(y))

        # Корректируем дисторсию
        points_undist = undistort_points(points_px, cameras)

        # Триангулируем каждую точку
        for pid in point_ids:
            n_total += 1
            pts_for_triangulation = []
            n_visible = 0
            for port, P in zip(ports, projections):
                pt = points_undist.get(port, {}).get(pid)
                pts_for_triangulation.append(pt)
                if pt is not None:
                    n_visible += 1

            if n_visible < MIN_CAMERAS:
                xyz = np.array([np.nan, np.nan, np.nan])
            else:
                xyz = triangulate_dlt(projections, pts_for_triangulation)
                if not np.isnan(xyz).any():
                    n_ok += 1

            results.append({
                "sync_index": frame_idx,
                "point_id": pid,
                "x_coord": round(float(xyz[0]), 6) if not np.isnan(xyz[0]) else np.nan,
                "y_coord": round(float(xyz[1]), 6) if not np.isnan(xyz[1]) else np.nan,
                "z_coord": round(float(xyz[2]), 6) if not np.isnan(xyz[2]) else np.nan,
            })

    pct = 100 * n_ok / n_total if n_total > 0 else 0
    print(f"Триангулировано: {n_ok}/{n_total} точек ({pct:.1f}%)")
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, out_path: str):
    """Сохраняет результаты в CSV."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    print(f"Результат сохранён в {out_path}")
    print(f"  Строк: {len(df)}, кадров: {df['sync_index'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Триангуляция 3D поз из 2D детекций с 3 камер (DLT)"
    )
    parser.add_argument("--session", required=True,
                        help="Папка с данными сессии Caliscope")
    parser.add_argument("--calib", required=True,
                        help="JSON файл с результатами калибровки")
    parser.add_argument("--model", default="SIMPLE_HOLISTIC",
                        help="Модель Caliscope (по умолчанию SIMPLE_HOLISTIC)")
    parser.add_argument("--out", required=True,
                        help="Путь для сохранения CSV с 3D координатами")
    parser.add_argument("--min-cameras", type=int, default=MIN_CAMERAS,
                        help=f"Минимум камер для триангуляции (по умолчанию {MIN_CAMERAS})")
    args = parser.parse_args()

    global MIN_CAMERAS
    MIN_CAMERAS = args.min_cameras

    print("Загрузка калибровки...")
    calib = load_calibration(args.calib)
    print(f"  Камер в калибровке: {len(calib['cameras'])}")

    print("\nЗагрузка 2D детекций...")
    xy_df = load_2d_detections(args.session, args.model)

    print("\nТриангуляция...")
    result_df = triangulate_session(xy_df, calib)

    save_results(result_df, args.out)


if __name__ == "__main__":
    main()
