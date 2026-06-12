"""
Калибровка многокамерной системы по ChArUco доске.

Использование:
    python calibration.py --intrinsic ./intrinsic --extrinsic ./extrinsic --out calibration_result.json

Структура папок:
    intrinsic/
        port_0.mp4, port_1.mp4, port_2.mp4   <- видео каждой камеры отдельно
    extrinsic/
        port_0.mp4, port_1.mp4, port_2.mp4   <- видео всех камер одновременно

Доска: ChArUco, 4 столбца x 3 строки, размер клетки ~3 см.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---------- Параметры доски ----------
# Меняйте здесь если доска другого размера
BOARD_COLS = 3        # клеток по горизонтали
BOARD_ROWS = 4        # клеток по вертикали
SQUARE_SIZE = 0.03    # метры (3 см — уточните по реальной доске)
MARKER_SIZE = 0.022   # метры (маркер обычно ~0.75 от клетки)
ARUCO_DICT  = cv2.aruco.DICT_4X4_50

# Каждый N-й кадр берём для калибровки (ускоряет обработку)
FRAME_STEP = 15


def make_board():
    """Создаёт объект ChArUco доски."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS),
        SQUARE_SIZE,
        MARKER_SIZE,
        aruco_dict,
    )
    return board, aruco_dict


def detect_charuco_in_video(video_path: str, board, aruco_dict, frame_step=FRAME_STEP):
    """
    Проходит по видео и собирает обнаружения ChArUco доски.
    Возвращает (all_corners, all_ids, image_size).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть видео: {video_path}")

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    all_corners = []
    all_ids = []
    image_size = None
    frame_idx = 0
    found_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is not None and len(marker_ids) >= 4:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            if ret and charuco_corners is not None and len(charuco_corners) >= 4:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                found_count += 1

        frame_idx += 1

    cap.release()
    print(f"  {Path(video_path).name}: найдено кадров с доской = {found_count}")
    return all_corners, all_ids, image_size


def calibrate_intrinsics(video_path: str, board, aruco_dict):
    """
    Калибрует внутренние параметры одной камеры.
    Возвращает (camera_matrix, dist_coeffs, image_size, rms_error).
    """
    print(f"Внутренняя калибровка: {Path(video_path).name}")
    corners, ids, image_size = detect_charuco_in_video(video_path, board, aruco_dict)

    if len(corners) < 5:
        raise RuntimeError(
            f"Слишком мало кадров с доской ({len(corners)}) для калибровки {video_path}. "
            "Убедитесь что доска хорошо видна и занимает значительную часть кадра."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        corners, ids, board, image_size, None, None
    )
    print(f"  RMS ошибка репроекции = {rms:.4f} пикселей")
    return camera_matrix, dist_coeffs, image_size, rms


def calibrate_extrinsics(video_paths: list, camera_matrices: list,
                         dist_coeffs_list: list, board, aruco_dict):
    """
    Вычисляет взаимное расположение камер (экзтринсики).
    Камера 0 принимается за мировую систему координат (R=I, T=0).
    Для каждой пары (0, i) находится относительная поза через общие
    наблюдения доски в одном кадре.

    Возвращает список (R, T) для каждой камеры.
    """
    n_cams = len(video_paths)
    print("\nВнешняя калибровка (взаимное расположение камер):")

    # Открываем все видео одновременно
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise RuntimeError(f"Не могу открыть видео: {video_paths[i]}")

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Собираем rvecs/tvecs для каждой камеры отдельно
    # rvecs[cam] = список векторов поворота для каждого кадра
    rvecs_per_cam = [[] for _ in range(n_cams)]
    tvecs_per_cam = [[] for _ in range(n_cams)]
    frame_valid = []   # индексы кадров где ВСЕ камеры видят доску

    frame_idx = 0
    while True:
        frames = []
        any_failed = False
        for cap in caps:
            ok, frame = cap.read()
            if not ok:
                any_failed = True
                break
            frames.append(frame)
        if any_failed:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        poses = []
        all_visible = True
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            pose = None
            if marker_ids is not None and len(marker_ids) >= 4:
                ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )
                if ret and ch_corners is not None and len(ch_corners) >= 4:
                    ok2, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        ch_corners, ch_ids, board,
                        camera_matrices[i], dist_coeffs_list[i],
                        None, None
                    )
                    if ok2:
                        pose = (rvec, tvec)
            if pose is None:
                all_visible = False
                break
            poses.append(pose)

        if all_visible:
            for i, (rv, tv) in enumerate(poses):
                rvecs_per_cam[i].append(rv)
                tvecs_per_cam[i].append(tv)
            frame_valid.append(frame_idx)

        frame_idx += 1

    for cap in caps:
        cap.release()

    print(f"  Кадров где все 3 камеры видят доску: {len(frame_valid)}")
    if len(frame_valid) < 3:
        raise RuntimeError(
            "Слишком мало общих кадров для внешней калибровки. "
            "Убедитесь что все камеры одновременно видят доску в extrinsic-видео."
        )

    # Камера 0 — мировая система координат
    results = []
    for i in range(n_cams):
        if i == 0:
            R = np.eye(3)
            T = np.zeros((3, 1))
        else:
            # Находим R, T такие что: P_world = R_i @ P_cam_i + T_i
            # Через цепочку: мир -> доска (из cam0) -> cam_i
            R_list, T_list = [], []
            for k in range(len(frame_valid)):
                rv0, tv0 = rvecs_per_cam[0][k], tvecs_per_cam[0][k]
                rvi, tvi = rvecs_per_cam[i][k], tvecs_per_cam[i][k]

                R0, _ = cv2.Rodrigues(rv0)
                Ri, _ = cv2.Rodrigues(rvi)

                # R_rel = R0 @ Ri^T,  T_rel = tv0 - R_rel @ tvi
                R_rel = R0 @ Ri.T
                T_rel = tv0 - R_rel @ tvi
                R_list.append(R_rel)
                T_list.append(T_rel)

            # Усредняем по всем кадрам
            R_mean = np.mean(R_list, axis=0)
            # Ортогонализируем через SVD
            U, _, Vt = np.linalg.svd(R_mean)
            R = U @ Vt
            T = np.mean(T_list, axis=0)

        results.append((R, T))
        print(f"  Камера {i}: T = {T.flatten().round(4)} м")

    return results


def save_calibration(camera_matrices, dist_coeffs_list, extrinsics,
                     image_sizes, rms_list, out_path: str):
    """Сохраняет результаты калибровки в JSON."""
    data = {
        "board": {
            "cols": BOARD_COLS,
            "rows": BOARD_ROWS,
            "square_size_m": SQUARE_SIZE,
            "marker_size_m": MARKER_SIZE,
        },
        "cameras": []
    }
    for i, (K, D, (R, T), sz, rms) in enumerate(
            zip(camera_matrices, dist_coeffs_list, extrinsics, image_sizes, rms_list)):
        data["cameras"].append({
            "port": i,
            "image_size": list(sz),
            "rms_intrinsic": round(float(rms), 6),
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.flatten().tolist(),
            "R": R.tolist(),
            "T": T.flatten().tolist(),
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"\nКалибровка сохранена в {out_path}")


def load_calibration(path: str) -> dict:
    """Загружает результаты калибровки из JSON."""
    with open(path) as f:
        data = json.load(f)
    cameras = []
    for cam in data["cameras"]:
        cameras.append({
            "port": cam["port"],
            "image_size": tuple(cam["image_size"]),
            "camera_matrix": np.array(cam["camera_matrix"]),
            "dist_coeffs": np.array(cam["dist_coeffs"]),
            "R": np.array(cam["R"]),
            "T": np.array(cam["T"]).reshape(3, 1),
        })
    return {"board": data["board"], "cameras": cameras}


def check_calibration_quality(calib: dict):
    """Выводит краткую сводку качества калибровки."""
    print("\n=== Качество калибровки ===")
    for cam in calib["cameras"]:
        K = cam["camera_matrix"]
        print(f"Камера {cam['port']}:")
        print(f"  Фокусное расстояние: fx={K[0,0]:.1f}, fy={K[1,1]:.1f} пикс")
        print(f"  Главная точка: cx={K[0,2]:.1f}, cy={K[1,2]:.1f} пикс")
        print(f"  Дисторсия: {np.array(cam['dist_coeffs']).round(5)}")
        T = np.array(cam['T']).flatten()
        print(f"  Позиция камеры: {T.round(3)} м")


def main():
    parser = argparse.ArgumentParser(description="Калибровка ChArUco 3-камерной системы")
    parser.add_argument("--intrinsic", required=True,
                        help="Папка с видео для внутренней калибровки")
    parser.add_argument("--extrinsic", required=True,
                        help="Папка с синхронными видео для внешней калибровки")
    parser.add_argument("--out", default="calibration_result.json",
                        help="Путь для сохранения результата")
    parser.add_argument("--square-size", type=float, default=0.03,
                        help="Размер клетки доски в метрах (по умолчанию 0.03)")
    args = parser.parse_args()

    global SQUARE_SIZE
    SQUARE_SIZE = args.square_size

    board, aruco_dict = make_board()
    ports = [0, 1, 2]

    # --- Внутренняя калибровка ---
    print("=" * 50)
    print("ШАГ 1: Внутренняя калибровка каждой камеры")
    print("=" * 50)
    camera_matrices, dist_coeffs_list, image_sizes, rms_list = [], [], [], []
    for port in ports:
        video = Path(args.intrinsic) / f"port_{port}.mp4"
        K, D, sz, rms = calibrate_intrinsics(str(video), board, aruco_dict)
        camera_matrices.append(K)
        dist_coeffs_list.append(D)
        image_sizes.append(sz)
        rms_list.append(rms)

    # --- Внешняя калибровка ---
    print("\n" + "=" * 50)
    print("ШАГ 2: Внешняя калибровка (расположение камер)")
    print("=" * 50)
    extrinsic_videos = [Path(args.extrinsic) / f"port_{p}.mp4" for p in ports]
    extrinsics = calibrate_extrinsics(
        extrinsic_videos, camera_matrices, dist_coeffs_list, board, aruco_dict
    )

    # --- Сохранение и проверка ---
    save_calibration(camera_matrices, dist_coeffs_list, extrinsics,
                     image_sizes, rms_list, args.out)
    calib = load_calibration(args.out)
    check_calibration_quality(calib)


if __name__ == "__main__":
    main()
