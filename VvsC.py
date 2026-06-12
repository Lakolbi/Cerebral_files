import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ---------- Загрузка и подготовка данных Vicon ----------

def load_vicon_points_with_pelvis(filepath: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Читает экспорт Vicon Trajectories (Excel) и возвращает:
      frames: (N,) массив номеров кадров
      points: dict[str -> (N,3)] в метрах
        включает LASI, RASI, LKNE, RKNE, LANK, RANK и
        производные LHIP, RHIP, PELVIS.
    """
    df_raw = pd.read_excel(filepath)

    # Строки с заголовками
    header_marker = df_raw.iloc[1]  # имена маркеров (LASI, RASI, ...)
    axis_row = df_raw.iloc[2]       # 'X', 'Y', 'Z' или 'Frame', 'Sub Frame'

    new_cols: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    current_marker: Optional[str] = None

    for j, col in enumerate(df_raw.columns):
        if j == 0:
            new_cols[col] = ("meta", "Frame")
            continue
        if j == 1:
            new_cols[col] = ("meta", "SubFrame")
            continue

        marker_cell = header_marker[col]
        axis = axis_row[col]

        # Если в ячейке есть имя маркера, запоминаем его
        if isinstance(marker_cell, str) and ":" in marker_cell:
            current_marker = marker_cell.split(":", 1)[1].strip()

        # Если столбец с X/Y/Z и у нас есть текущий маркер – сохраняем
        if isinstance(axis, str) and axis.strip() in ("X", "Y", "Z") and current_marker is not None:
            new_cols[col] = (current_marker, axis.strip())
        else:
            new_cols[col] = (None, None)

    # Данные начинаются с 5-й строки (индекс 4)
    data = df_raw.iloc[4:].reset_index(drop=True)

    # Номера кадров
    frames = data[df_raw.columns[0]].astype(float).to_numpy()

    # Собираем координаты по маркерам
    markers: Dict[str, Dict[str, np.ndarray]] = {}
    for col, (name, axis) in new_cols.items():
        if name is None or axis is None or name == "meta":
            continue
        markers.setdefault(name, {})[axis] = data[col].astype(float).to_numpy()

    points: Dict[str, np.ndarray] = {}
    for name, comps in markers.items():
        if all(a in comps for a in ("X", "Y", "Z")):
            arr_mm = np.stack([comps["X"], comps["Y"], comps["Z"]], axis=1)
            arr_m = arr_mm / 1000.0  # mm -> m
            points[name] = arr_m

    # Строим LHIP, RHIP и PELVIS из LASI/RASI
    points_out = dict(points)
    if "LASI" in points and "RASI" in points:
        points_out["LHIP"] = points["LASI"]
        points_out["RHIP"] = points["RASI"]
        points_out["PELVIS"] = 0.5 * (points["LASI"] + points["RASI"])
    elif "LASI" in points or "RASI" in points:
        base = points.get("LASI", points.get("RASI"))
        points_out["LHIP"] = base
        points_out["RHIP"] = base
        points_out["PELVIS"] = base

    return frames, points_out


# ---------- Загрузка и подготовка данных Caliscope ----------

def load_caliscope_points_with_pelvis(filepath: str, fs: float = 20.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Читает CSV Caliscope / MediaPipe Holistic и возвращает:
      time: (N,) массив времени в секундах
      points: dict[str -> (N,3)]
        LHIP, RHIP, LKNE, RKNE, LANK, RANK, PELVIS
    """
    df = pd.read_csv(filepath)

    if "sync_index" in df.columns:
        time = df["sync_index"].to_numpy().astype(float) / fs
    else:
        time = np.arange(len(df), dtype=float) / fs

    points: Dict[str, np.ndarray] = {}

    def get_as(name: str, prefix: str) -> None:
        cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        if all(c in df.columns for c in cols):
            points[name] = df[cols].to_numpy().astype(float)

    get_as("LANK", "left_ankle")
    get_as("RANK", "right_ankle")
    get_as("LKNE", "left_knee")
    get_as("RKNE", "right_knee")
    get_as("LHIP", "left_hip")
    get_as("RHIP", "right_hip")

    if "LHIP" in points and "RHIP" in points:
        points["PELVIS"] = 0.5 * (points["LHIP"] + points["RHIP"])

    return time, points


# ---------- Поиск прыжка по скорости ----------

# ---------- Поиск события (прыжка) ----------

def detect_event_index(
    time: np.ndarray,
    point: np.ndarray,
    max_search_time: Optional[float] = None,
    mode: str = "speed",
) -> int:
    """
    Находит индекс кадра события в начале записи.

    mode:
      - "speed": максимум 3D-скорости (как было раньше);
      - "zmax": максимум координаты Z (апекс прыжка).
    """
    n = len(time)
    if n < 2:
        return 0

    # Ограничиваемся началом записи, если задано
    if max_search_time is not None:
        idx_use = np.where(time <= max_search_time)[0]
        if len(idx_use) < 2:
            idx_use = np.arange(n)
    else:
        idx_use = np.arange(n)

    t_use = time[idx_use]
    p_use = point[idx_use]

    if mode == "speed":
        dt = np.diff(t_use)
        if np.any(dt == 0):
            non_zero = dt[dt > 0]
            if len(non_zero) == 0:
                dt[dt == 0] = 1.0
            else:
                dt[dt == 0] = np.min(non_zero)

        vel = np.diff(p_use, axis=0) / dt[:, None]
        speed = np.linalg.norm(vel, axis=1)

        if len(speed) == 0:
            return int(idx_use[0])

        rel = int(np.argmax(speed))
        event_idx = int(idx_use[rel + 1])  # скорость между кадрами -> относим к следующему
        return event_idx

    elif mode == "zmax":
        # Предполагаем, что вертикальная ось – Z (индекс 2)
        if p_use.shape[1] < 3:
            raise ValueError("У точки нет Z-координаты для поиска максимума")
        z = p_use[:, 2]
        rel = int(np.argmax(z))  # максимум Z в окне
        event_idx = int(idx_use[rel])
        return event_idx

    else:
        raise ValueError(f"Неизвестный режим события: {mode}")

def find_z_peak_index_in_window(
    time_al: np.ndarray,
    point: np.ndarray,
    window: Tuple[float, float],
) -> int:
    """
    Ищет индекс максимума Z для точки point в заданном окне времени.

    time_al : (N,) текущая временная шкала (после грубого выравнивания)
    point   : (N,3) координаты точки (например, PELVIS)
    window  : (t_start, t_end) в секундах относительно time_al

    Если в окне нет точек, возвращает глобальный максимум Z.
    """
    t_start, t_end = window
    if t_end <= t_start:
        raise ValueError("Некорректное окно для поиска пика по Z")

    if point.shape[1] < 3:
        raise ValueError("У точки нет Z-координаты")

    z = point[:, 2]
    mask = (time_al >= t_start) & (time_al <= t_end)

    if not np.any(mask):
        # Нет ни одной точки в окне – берём глобальный максимум
        return int(np.argmax(z))

    # Локальный максимум в окне
    idx_local = int(np.argmax(z[mask]))
    idx_global = int(np.arange(len(time_al))[mask][idx_local])
    return idx_global

def detect_jump_index(time: np.ndarray, point: np.ndarray, max_search_time: Optional[float] = None) -> int:
    """
    Обёртка для совместимости: старое поведение (по скорости).
    """
    return detect_event_index(time, point, max_search_time=max_search_time, mode="speed")


# ---------- Пространственное подобие (масштаб + поворот + сдвиг) ----------

def compute_similarity_transform(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Находит подобие (s, R, t) такое, что B ≈ s * A @ R.T + t.
    A, B: (N,3)
    """
    assert A.shape == B.shape
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    A0 = A - mu_A
    B0 = B - mu_B

    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_A = (A0 ** 2).sum()
    s = S.sum() / var_A
    t = mu_B - s * (R @ mu_A)
    return s, R, t


def apply_similarity_transform(A: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Применяет найденное подобие к массиву точек A (N,3).
    """
    return s * (A @ R.T) + t


# ---------- Временное выравнивание и интерполяция ----------

def align_time_and_interpolate(
    vicon_time: np.ndarray,
    vicon_points: Dict[str, np.ndarray],
    cal_time: np.ndarray,
    cal_points: Dict[str, np.ndarray],
    joint_names: List[str],
    event_mode: str = "speed",
    z_peak_window: Optional[Tuple[float, float]] = (0.0, 5.0),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    event_mode:
      - "speed": синхронизация по максимуму 3D-скорости;
      - "zmax": по максимуму Z;
      - "xcorr": по кросс-корреляции скорости PELVIS.

    z_peak_window:
      - если None, дополнительная подстройка по пику Z не выполняется;
      - иначе (t_start, t_end) – окно, в котором ищем максимум Z PELVIS
        и ещё раз выравниваем обе шкалы так, чтобы этот максимум был в t=0.
    """
    if "PELVIS" in vicon_points and "PELVIS" in cal_points:
        sync_joint = "PELVIS"
    elif joint_names:
        sync_joint = joint_names[0]
    else:
        raise ValueError("Нет сустава для синхронизации")

    # --- Шаг 1. Грубое выравнивание по event_mode ---
    if event_mode in ("speed", "zmax"):
        idx_v = detect_event_index(
            vicon_time, vicon_points[sync_joint], max_search_time=2.0, mode=event_mode
        )
        idx_c = detect_event_index(
            cal_time, cal_points[sync_joint], max_search_time=2.0, mode=event_mode
        )

        t_v_al = vicon_time - vicon_time[idx_v]
        t_c_al = cal_time - cal_time[idx_c]

    elif event_mode == "xcorr":
        shift_sec = estimate_time_shift_by_xcorr(
            vicon_time,
            vicon_points[sync_joint],
            cal_time,
            cal_points[sync_joint],
            fs_grid=100.0,
            max_shift=0.5,
        )
        # Vicon оставляем как есть, Caliscope сдвигаем
        t_v_al = vicon_time.copy()
        t_c_al = cal_time - shift_sec
    else:
        raise ValueError(f"Неизвестный event_mode: {event_mode}")

    # --- Шаг 2. Тонкая подстройка по пику Z в заданном окне ---
    if z_peak_window is not None:
        idx_v_peak = find_z_peak_index_in_window(
            t_v_al, vicon_points[sync_joint], z_peak_window
        )
        idx_c_peak = find_z_peak_index_in_window(
            t_c_al, cal_points[sync_joint], z_peak_window
        )

        t0_v = t_v_al[idx_v_peak]
        t0_c = t_c_al[idx_c_peak]

        # Сдвигаем шкалы так, чтобы их пики оказались в t=0
        t_v_al = t_v_al - t0_v
        t_c_al = t_c_al - t0_c

    # --- Шаг 3. Общий интервал и интерполяция ---
    start_t = max(t_v_al[0], t_c_al[0])
    end_t = min(t_v_al[-1], t_c_al[-1])

    mask_c = (t_c_al >= start_t) & (t_c_al <= end_t)
    t_common = t_c_al[mask_c]

    if t_common.size < 2:
        raise ValueError("Нет пересечения по времени между Vicon и Caliscope")

    def interp(time_src: np.ndarray, values_src: np.ndarray, time_tgt: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.interp(time_tgt, time_src, values_src[:, 0]),
            np.interp(time_tgt, time_src, values_src[:, 1]),
            np.interp(time_tgt, time_src, values_src[:, 2]),
        ])

    vicon_res: Dict[str, np.ndarray] = {}
    cal_res: Dict[str, np.ndarray] = {}

    for name in joint_names:
        if name in vicon_points:
            vicon_res[name] = interp(t_v_al, vicon_points[name], t_common)
        if name in cal_points:
            cal_res[name] = cal_points[name][mask_c]

    return t_common, vicon_res, cal_res

# ---------- Расчёт ошибок для одного испытания ----------

def compute_errors_for_trial(
    vicon_path: str,
    cal_path: str,
    vicon_fps: float = 100.0,
    cal_fps: float = 20.0,
    joint_names: Optional[List[str]] = None,
    event_mode: str = "speed",
) -> Dict[str, object]:
    """
    Обрабатывает одну пару файлов (Vicon + Caliscope) и возвращает:
      - t: общая временная сетка
      - joints_used: список суставов, реально использованных
      - vicon_transformed: словарь сустав -> матрица (T,3) после подобия
      - caliscope: словарь сустав -> матрица (T,3)
      - joint_metrics: метрики по суставам
          * 3D-ошибка (rmse_m, mae_m, max_m, median_m)
          * ошибка по осям (rmse_x_m, rmse_y_m, rmse_z_m)
          * 3D-ошибка в системе координат таза (rmse_rel_m)
            и по осям (rmse_rel_x_m, rmse_rel_y_m, rmse_rel_z_m)
      - global_metrics: глобальные метрики по всем суставам
          * 3D-ошибка (rmse_m, mae_m, max_m, median_m)
          * 3D-ошибка в системе координат таза (rmse_rel_m)
      - similarity: параметры подобия (s, R, t)
    """
    if joint_names is None:
        joint_names = ["LHIP", "RHIP", "LKNE", "RKNE", "LANK", "RANK", "PELVIS"]

    # Загрузка данных
    frames, vicon_pts = load_vicon_points_with_pelvis(vicon_path)
    vicon_time = frames.astype(float) / vicon_fps

    cal_time, cal_pts = load_caliscope_points_with_pelvis(cal_path, fs=cal_fps)

    # Временное выравнивание и интерполяция
    t_common, vicon_al, cal_al = align_time_and_interpolate(
        vicon_time,
        vicon_pts,
        cal_time,
        cal_pts,
        joint_names,
        event_mode=event_mode,
        z_peak_window=(0.0, 5.0),  # окно поиска первого большого пика по Z
    )

    joints_used = [j for j in joint_names if j in vicon_al and j in cal_al]
    if not joints_used:
        raise ValueError("Нет общих суставов в Vicon и Caliscope")

    T = t_common.size

    # Формируем массивы для оценки подобия
    A_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []

    # Общая маска кадров без NaN для всех суставов
    mask_all = np.ones(T, dtype=bool)
    for j in joints_used:
        v = vicon_al[j]
        c = cal_al[j]
        mask_joint = (~np.isnan(v).any(axis=1)) & (~np.isnan(c).any(axis=1))
        mask_all &= mask_joint

    if not mask_all.any():
        raise ValueError("Все кадры содержат пропуски, нельзя оценить трансформацию")

    for j in joints_used:
        v = vicon_al[j][mask_all]
        c = cal_al[j][mask_all]
        A_list.append(v)
        B_list.append(c)

    A_all = np.concatenate(A_list, axis=0)
    B_all = np.concatenate(B_list, axis=0)

    # Подобие (масштаб + поворот + сдвиг)
    s, R, t = compute_similarity_transform(A_all, B_all)

    # Применяем подобие к Vicon
    vicon_transformed: Dict[str, np.ndarray] = {}
    for j in joints_used:
        vicon_transformed[j] = apply_similarity_transform(vicon_al[j], s, R, t)

    # --- Относительные координаты в системе таза ---
    vicon_rel: Dict[str, np.ndarray] = {}
    cal_rel: Dict[str, np.ndarray] = {}
    has_pelvis = "PELVIS" in vicon_transformed and "PELVIS" in cal_al

    if has_pelvis:
        pelvis_v = vicon_transformed["PELVIS"]
        pelvis_c = cal_al["PELVIS"]
        for j in joints_used:
            vicon_rel[j] = vicon_transformed[j] - pelvis_v
            cal_rel[j] = cal_al[j] - pelvis_c

    # --- Метрики ---
    joint_metrics: Dict[str, Dict[str, float]] = {}
    all_dists_abs: List[np.ndarray] = []
    all_dists_rel: List[np.ndarray] = []

    for j in joints_used:
        v = vicon_transformed[j]
        c = cal_al[j]

        diff = v - c
        mask = ~np.isnan(diff).any(axis=1)
        if not mask.any():
            continue

        diff_valid = diff[mask]
        dists = np.linalg.norm(diff_valid, axis=1)
        all_dists_abs.append(dists)

        ex = diff_valid[:, 0]
        ey = diff_valid[:, 1]
        ez = diff_valid[:, 2]

        rmse = float(np.sqrt(np.mean(dists ** 2)))
        mae = float(np.mean(np.abs(dists)))
        max_err = float(np.max(dists))
        median_err = float(np.median(dists))

        rmse_x = float(np.sqrt(np.mean(ex ** 2)))
        rmse_y = float(np.sqrt(np.mean(ey ** 2)))
        rmse_z = float(np.sqrt(np.mean(ez ** 2)))

        metrics_j: Dict[str, float] = {
            "n_samples": int(dists.size),
            "rmse_m": rmse,
            "mae_m": mae,
            "max_m": max_err,
            "median_m": median_err,
            "rmse_x_m": rmse_x,
            "rmse_y_m": rmse_y,
            "rmse_z_m": rmse_z,
        }

        # Относительные координаты (если есть таз)
        if has_pelvis:
            v_r = vicon_rel[j]
            c_r = cal_rel[j]
            diff_r = v_r - c_r
            mask_r = ~np.isnan(diff_r).any(axis=1)
            if mask_r.any():
                diff_r_valid = diff_r[mask_r]
                dists_r = np.linalg.norm(diff_r_valid, axis=1)
                all_dists_rel.append(dists_r)

                ex_r = diff_r_valid[:, 0]
                ey_r = diff_r_valid[:, 1]
                ez_r = diff_r_valid[:, 2]

                rmse_rel = float(np.sqrt(np.mean(dists_r ** 2)))
                rmse_rel_x = float(np.sqrt(np.mean(ex_r ** 2)))
                rmse_rel_y = float(np.sqrt(np.mean(ey_r ** 2)))
                rmse_rel_z = float(np.sqrt(np.mean(ez_r ** 2)))

                metrics_j["rmse_rel_m"] = rmse_rel
                metrics_j["rmse_rel_x_m"] = rmse_rel_x
                metrics_j["rmse_rel_y_m"] = rmse_rel_y
                metrics_j["rmse_rel_z_m"] = rmse_rel_z
            else:
                metrics_j["rmse_rel_m"] = float("nan")
                metrics_j["rmse_rel_x_m"] = float("nan")
                metrics_j["rmse_rel_y_m"] = float("nan")
                metrics_j["rmse_rel_z_m"] = float("nan")

        joint_metrics[j] = metrics_j

    # --- Глобальные метрики ---
    if all_dists_abs:
        all_concat_abs = np.concatenate(all_dists_abs)
        global_metrics: Dict[str, float] = {
            "n_samples": int(all_concat_abs.size),
            "rmse_m": float(np.sqrt(np.mean(all_concat_abs ** 2))),
            "mae_m": float(np.mean(np.abs(all_concat_abs))),
            "max_m": float(np.max(all_concat_abs)),
            "median_m": float(np.median(all_concat_abs)),
        }
    else:
        global_metrics = {
            "n_samples": 0,
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "max_m": float("nan"),
            "median_m": float("nan"),
        }

    if all_dists_rel:
        all_concat_rel = np.concatenate(all_dists_rel)
        global_metrics["rmse_rel_m"] = float(np.sqrt(np.mean(all_concat_rel ** 2)))
    else:
        global_metrics["rmse_rel_m"] = float("nan")

    return {
        "t": t_common,
        "joints_used": joints_used,
        "vicon_transformed": vicon_transformed,
        "caliscope": cal_al,
        "joint_metrics": joint_metrics,
        "global_metrics": global_metrics,
        "similarity": {"s": s, "R": R, "t": t},
    }


# ---------- Обработка всех пар файлов и запись сводки ----------

def process_all_pairs(
    pairs: List[Tuple[str, str]],
    vicon_fps: float = 100.0,
    cal_fps: float = 20.0,
    output_summary_path: str = "results_summary.csv",
) -> pd.DataFrame:
    """
    Обрабатывает все пары файлов и сохраняет сводку в CSV:
      trial, joint, n_samples, rmse_m, mae_m, max_m, median_m,
      rmse_x_m, rmse_y_m, rmse_z_m, rmse_rel_m, rmse_rel_x_m, rmse_rel_y_m, rmse_rel_z_m (если есть таз)
    """
    summary_rows: List[Dict[str, object]] = []

    for idx, (vicon_path, cal_path) in enumerate(pairs, start=1):
        print(f"Обработка пары {idx}:")
        result = compute_errors_for_trial(
            vicon_path,
            cal_path,
            vicon_fps=vicon_fps,
            cal_fps=cal_fps,
            event_mode="xcorr",
        )

        trial_name = f"trial_{idx}"

        # Метрики по суставам
        for joint, metrics in result["joint_metrics"].items():
            row: Dict[str, object] = {"trial": trial_name, "joint": joint}
            row.update(metrics)
            summary_rows.append(row)

        # Глобальные метрики
        g = result["global_metrics"]
        row_g: Dict[str, object] = {"trial": trial_name, "joint": "ALL"}
        row_g.update(g)
        summary_rows.append(row_g)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(output_summary_path, index=False)
    print(f"Итоговая сводка сохранена в {output_summary_path}")
    return df_sum

def debug_plot_jump(
    vicon_path: str,
    cal_path: str,
    vicon_fps: float = 100.0,
    cal_fps: float = 20.0,
    max_search_time: float = 2.0,
) -> None:
    """
    Визуализирует момент прыжка для одной пары Vicon + Caliscope:
      - вертикальное положение PELVIS (или LHIP) вокруг прыжка
      - модуль скорости вокруг прыжка
    Помогает проверить, корректно ли выбраны индексы прыжка.
    """
    # Загрузка данных
    frames, vicon_pts = load_vicon_points_with_pelvis(vicon_path)
    vicon_time = frames.astype(float) / vicon_fps

    cal_time, cal_pts = load_caliscope_points_with_pelvis(cal_path, fs=cal_fps)

    # Выбираем сустав для синхронизации
    if "PELVIS" in vicon_pts and "PELVIS" in cal_pts:
        sync_joint = "PELVIS"
    elif "LHIP" in vicon_pts and "LHIP" in cal_pts:
        sync_joint = "LHIP"
    else:
        raise ValueError("Нет общей точки PELVIS или LHIP для визуализации прыжка")

    p_v = vicon_pts[sync_joint]
    p_c = cal_pts[sync_joint]

    # Индексы прыжка по текущему алгоритму
    idx_v = detect_jump_index(vicon_time, p_v, max_search_time=max_search_time)
    idx_c = detect_jump_index(cal_time, p_c, max_search_time=max_search_time)

    print(f"[debug_plot_jump] Vicon: jump_idx={idx_v}, time={vicon_time[idx_v]:.3f} s")
    print(f"[debug_plot_jump] Caliscope: jump_idx={idx_c}, time={cal_time[idx_c]:.3f} s")

    # Переводим время так, чтобы прыжок был в 0
    t_v = vicon_time - vicon_time[idx_v]
    t_c = cal_time - cal_time[idx_c]

    # Окно просмотра вокруг прыжка
    win = 1.0  # секунды
    mask_v = (t_v >= -win) & (t_v <= win)
    mask_c = (t_c >= -win) & (t_c <= win)

    # ----- Фигура 1: вертикальная координата вокруг прыжка -----
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    # предполагаем, что вертикальная ось - Z (индекс 2)
    ax1.plot(t_v[mask_v], p_v[mask_v, 2], label="Vicon Z", linewidth=1.0)
    ax1.plot(t_c[mask_c], p_c[mask_c, 2], label="Caliscope Z", linewidth=1.0, linestyle="--")
    ax1.axvline(0.0, color="gray", linestyle=":")
    ax1.set_xlabel("Время относительно прыжка, с")
    ax1.set_ylabel(f"{sync_joint} Z, м")
    ax1.set_title(f"Вертикальное движение {sync_joint} вокруг прыжка")
    ax1.legend()
    ax1.grid(True)

    # ----- Фигура 2: модуль скорости вокруг прыжка -----
    # Скорости для Vicon
    dt_v = np.diff(vicon_time)
    dt_v[dt_v == 0] = np.min(dt_v[dt_v > 0]) if np.any(dt_v > 0) else 1.0
    vel_v = np.diff(p_v, axis=0) / dt_v[:, None]
    speed_v = np.linalg.norm(vel_v, axis=1)
    t_v_mid = 0.5 * (vicon_time[:-1] + vicon_time[1:]) - vicon_time[idx_v]

    # Скорости для Caliscope
    dt_c = np.diff(cal_time)
    dt_c[dt_c == 0] = np.min(dt_c[dt_c > 0]) if np.any(dt_c > 0) else 1.0
    vel_c = np.diff(p_c, axis=0) / dt_c[:, None]
    speed_c = np.linalg.norm(vel_c, axis=1)
    t_c_mid = 0.5 * (cal_time[:-1] + cal_time[1:]) - cal_time[idx_c]

    mask_v2 = (t_v_mid >= -win) & (t_v_mid <= win)
    mask_c2 = (t_c_mid >= -win) & (t_c_mid <= win)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t_v_mid[mask_v2], speed_v[mask_v2], label="Vicon speed", linewidth=1.0)
    ax2.plot(t_c_mid[mask_c2], speed_c[mask_c2], label="Caliscope speed", linewidth=1.0, linestyle="--")
    ax2.axvline(0.0, color="gray", linestyle=":")
    ax2.set_xlabel("Время относительно прыжка, с")
    ax2.set_ylabel(f"Скорость {sync_joint}, м/с")
    ax2.set_title(f"Модуль скорости {sync_joint} вокруг прыжка")
    ax2.legend()
    ax2.grid(True)

    plt.show()

def debug_plot_aligned_joint(
    vicon_path: str,
    cal_path: str,
    joint: str = "RANK",
    vicon_fps: float = 100.0,
    cal_fps: float = 20.0,
    event_mode: str = "speed",   # <--- НОВЫЙ параметр
) -> None:
    """
    Визуализирует выровненные траектории выбранного сустава (по умолчанию RANK):
      - Vicon (после подобия)
      - Caliscope
    по осям X, Y, Z во времени.
    Это помогает понять, нет ли фазового сдвига (разъезда по времени).
    """
    # Те же суставы, что и в основном коде
    joint_names = ["LHIP", "RHIP", "LKNE", "RKNE", "LANK", "RANK", "PELVIS"]

    # Загрузка данных
    frames, vicon_pts = load_vicon_points_with_pelvis(vicon_path)
    vicon_time = frames.astype(float) / vicon_fps
    cal_time, cal_pts = load_caliscope_points_with_pelvis(cal_path, fs=cal_fps)

    # Временное выравнивание и интерполяция
    t_common, vicon_al, cal_al = align_time_and_interpolate(
        vicon_time,
        vicon_pts,
        cal_time,
        cal_pts,
        joint_names,
        event_mode=event_mode,
        z_peak_window=(0.0, 5.0),
    )

    if joint not in vicon_al or joint not in cal_al:
        raise ValueError(f"Сустав {joint} не найден одновременно в Vicon и Caliscope")

    # Подготовка для подобия (как в compute_errors_for_trial)
    joints_used = [j for j in joint_names if j in vicon_al and j in cal_al]
    T = t_common.size

    A_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []

    mask_all = np.ones(T, dtype=bool)
    for j in joints_used:
        v = vicon_al[j]
        c = cal_al[j]
        mask_joint = (~np.isnan(v).any(axis=1)) & (~np.isnan(c).any(axis=1))
        mask_all &= mask_joint

    if not mask_all.any():
        raise ValueError("Все кадры содержат пропуски, нельзя оценить трансформацию")

    for j in joints_used:
        v = vicon_al[j][mask_all]
        c = cal_al[j][mask_all]
        A_list.append(v)
        B_list.append(c)

    A_all = np.concatenate(A_list, axis=0)
    B_all = np.concatenate(B_list, axis=0)

    s, R, t = compute_similarity_transform(A_all, B_all)

    # Применяем подобие только к выбранному суставу
    v_joint = vicon_al[joint]
    v_joint_tr = apply_similarity_transform(v_joint, s, R, t)
    c_joint = cal_al[joint]

    # Маска без NaN для конкретного сустава
    mask_joint = (~np.isnan(v_joint_tr).any(axis=1)) & (~np.isnan(c_joint).any(axis=1))

    t_plot = t_common[mask_joint]
    v_plot = v_joint_tr[mask_joint]
    c_plot = c_joint[mask_joint]

    # Визуализируем по осям X, Y, Z
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axis_labels = ["X", "Y", "Z"]
    for i in range(3):
        axes[i].plot(t_plot, v_plot[:, i], label="Vicon (transformed)", linewidth=1.0)
        axes[i].plot(t_plot, c_plot[:, i], label="Caliscope", linewidth=1.0, linestyle="--")
        axes[i].set_ylabel(f"{joint} {axis_labels[i]}, м")
        axes[i].grid(True)
        if i == 0:
            axes[i].set_title(f"Сравнение траекторий {joint} по осям X/Y/Z (после выравнивания)")

    axes[-1].set_xlabel("Время, с")
    axes[0].legend()

    plt.tight_layout()
    plt.show()

def estimate_time_shift_by_xcorr(
    vicon_time: np.ndarray,
    vicon_point: np.ndarray,
    cal_time: np.ndarray,
    cal_point: np.ndarray,
    fs_grid: float = 100.0,
    max_shift: float = 0.5,
) -> float:
    """
    Оценивает глобальный временной сдвиг между Vicon и Caliscope по
    кросс-корреляции модуля скорости выбранной точки (обычно PELVIS).

    Возвращает shift_sec:
      если shift_sec > 0, то Caliscope ЗАДЕРЖИВАЕТСЯ относительно Vicon
      на shift_sec секунд и его нужно сдвинуть НАЗАД:
        t_c_aligned = cal_time - shift_sec
    """
    def compute_speed(time: np.ndarray, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(time)
        if n < 2:
            raise ValueError("Недостаточно точек для вычисления скорости")

        dt = np.diff(time)
        if np.any(dt == 0):
            non_zero = dt[dt > 0]
            if len(non_zero) == 0:
                dt[dt == 0] = 1.0
            else:
                dt[dt == 0] = np.min(non_zero)

        vel = np.diff(point, axis=0) / dt[:, None]
        speed = np.linalg.norm(vel, axis=1)
        t_mid = 0.5 * (time[:-1] + time[1:])
        return t_mid, speed

    # 1) Скорости для обеих систем
    t_v_speed, s_v = compute_speed(vicon_time, vicon_point)
    t_c_speed, s_c = compute_speed(cal_time, cal_point)

    # 2) Общий интервал
    start = max(t_v_speed[0], t_c_speed[0])
    end = min(t_v_speed[-1], t_c_speed[-1])
    if end - start <= 2.0 / fs_grid:
        raise ValueError("Слишком маленькое пересечение для оценки сдвига")

    t_grid = np.arange(start, end, 1.0 / fs_grid)

    s_vg = np.interp(t_grid, t_v_speed, s_v)
    s_cg = np.interp(t_grid, t_c_speed, s_c)

    # 3) Нормируем сигналы
    s_vg = s_vg - np.mean(s_vg)
    s_cg = s_cg - np.mean(s_cg)
    std_v = np.std(s_vg)
    std_c = np.std(s_cg)
    if std_v > 0:
        s_vg = s_vg / std_v
    if std_c > 0:
        s_cg = s_cg / std_c

    # 4) Кросс-корреляция: correlate( caliscope , vicon )
    corr = np.correlate(s_cg, s_vg, mode="full")
    lags = np.arange(-len(s_vg) + 1, len(s_cg))

    max_lag_samples = int(max_shift * fs_grid)
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    if not np.any(mask):
        raise ValueError("Нет лагов в заданном окне max_shift")

    corr_window = corr[mask]
    lags_window = lags[mask]

    best_idx = int(np.argmax(corr_window))
    best_lag = int(lags_window[best_idx])

    shift_sec = best_lag / fs_grid
    print(f"[xcorr] best_lag={best_lag} samples, shift_sec={shift_sec:.4f} s")
    return shift_sec

def plot_z_for_joint_all_trials(
    pairs: List[Tuple[str, str]],
    joint: str,
    vicon_fps: float,
    cal_fps: float,
    event_mode: str,
    output_dir: str,
) -> None:
    """
    Для каждой пары файлов строит график Z-координаты выбранного сустава:
      Vicon (после подобия) vs Caliscope
    и сохраняет в PNG-файлы в output_dir.

    joint: например, "PELVIS" или "RANK".
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (vicon_path, cal_path) in enumerate(pairs, start=1):
        print(f"[plot_z] Пара {idx}: {os.path.basename(vicon_path)} vs {os.path.basename(cal_path)}")

        # Берём готовый пайплайн, чтобы не дублировать логику
        result = compute_errors_for_trial(
            vicon_path=vicon_path,
            cal_path=cal_path,
            vicon_fps=vicon_fps,
            cal_fps=cal_fps,
            event_mode=event_mode,
        )

        t = result["t"]
        vicon_transformed = result["vicon_transformed"]
        caliscope = result["caliscope"]

        if joint not in vicon_transformed or joint not in caliscope:
            print(f"  [plot_z] Сустав {joint} отсутствует в этой паре, пропускаю.")
            continue

        v = vicon_transformed[joint]
        c = caliscope[joint]

        # Маска без NaN
        mask = (~np.isnan(v).any(axis=1)) & (~np.isnan(c).any(axis=1))
        if not mask.any():
            print(f"  [plot_z] Все значения для {joint} содержат NaN, пропускаю.")
            continue

        t_plot = t[mask]
        v_plot = v[mask]
        c_plot = c[mask]

        # Строим график Z
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_plot, v_plot[:, 2], label="Vicon (transformed) Z", linewidth=1.0)
        ax.plot(t_plot, c_plot[:, 2], label="Caliscope Z", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Время, с")
        ax.set_ylabel(f"{joint} Z, м")
        ax.set_title(f"{joint} Z во времени, пара {idx}")
        ax.grid(True)
        ax.legend()

        fname = f"{joint}_Z_trial_{idx}.png"
        out_path = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"  [plot_z] Сохранён файл {out_path}")


def detect_heel_strikes_from_ankle_z(
    time: np.ndarray,
    ankle_z: np.ndarray,
    min_interval_s: float = 0.5,
) -> np.ndarray:
    """
    Грубый детектор моментов контакта стопы с опорой по локальным минимумам Z голеностопа.

    time        : (T,) массив времени (t_common)
    ankle_z     : (T,) массив Z-координаты голеностопа
    min_interval_s : минимальный интервал между событиями (сек), чтобы не ловить шум

    Возвращает массив индексов time[], соответствующих контактам.
    """
    T = time.size
    if T < 3:
        return np.array([], dtype=int)

    dz = np.diff(time)
    dt_median = float(np.median(dz[dz > 0])) if np.any(dz > 0) else 0.05
    min_distance_frames = max(1, int(min_interval_s / max(dt_median, 1e-6)))

    z = ankle_z
    events: List[int] = []
    last_idx = -min_distance_frames

    for i in range(1, T - 1):
        if np.isnan(z[i]) or np.isnan(z[i - 1]) or np.isnan(z[i + 1]):
            continue
        if z[i] <= z[i - 1] and z[i] < z[i + 1]:
            if i - last_idx >= min_distance_frames:
                events.append(i)
                last_idx = i

    return np.array(events, dtype=int)

def compute_spatiotemporal_for_trial(result: Dict[str, object]) -> Dict[str, float]:
    """
    Вычисляет пространственно-временные параметры шага для одной пары:
      - средняя длина шага по главной оси движения (Vicon и Caliscope)
      - ошибка по длине шага (абсолютная и относительная)
      - среднее время шага (Vicon и Caliscope)
      - каденс (Vicon и Caliscope) и ошибка по каденсу
    Использует RANK как опорную ногу.
    """
    t = result["t"]
    vicon_transformed: Dict[str, np.ndarray] = result["vicon_transformed"]  # type: ignore
    caliscope: Dict[str, np.ndarray] = result["caliscope"]  # type: ignore

    metrics: Dict[str, float] = {
        "mean_step_length_vicon_m": float("nan"),
        "mean_step_length_caliscope_m": float("nan"),
        "step_length_error_m": float("nan"),
        "step_length_error_rel_percent": float("nan"),
        "mean_step_time_vicon_s": float("nan"),
        "mean_step_time_caliscope_s": float("nan"),
        "cadence_vicon_steps_per_min": float("nan"),
        "cadence_caliscope_steps_per_min": float("nan"),
        "cadence_error_steps_per_min": float("nan"),
    }

    if "RANK" not in vicon_transformed or "RANK" not in caliscope:
        return metrics

    ankle_v = vicon_transformed["RANK"]
    ankle_c = caliscope["RANK"]

    z_v = ankle_v[:, 2]
    z_c = ankle_c[:, 2]

    # События шага по Vicon и Caliscope отдельно
    events_v = detect_heel_strikes_from_ankle_z(t, z_v, min_interval_s=0.5)
    events_c = detect_heel_strikes_from_ankle_z(t, z_c, min_interval_s=0.5)

    if events_v.size < 2 or events_c.size < 2:
        return metrics

    # Ось движения – та, по которой таз смещается сильнее всего
    if "PELVIS" in vicon_transformed:
        pelvis_v = vicon_transformed["PELVIS"]
    else:
        pelvis_v = ankle_v
    pelvis_range = np.ptp(pelvis_v, axis=0)
    forward_axis = int(np.argmax(np.abs(pelvis_range)))

    # --- Vicon: длина и время шага ---
    step_lengths_v: List[float] = []
    step_times_v: List[float] = []
    for k in range(events_v.size - 1):
        i0 = int(events_v[k])
        i1 = int(events_v[k + 1])
        if i1 <= i0:
            continue
        if np.isnan(ankle_v[i0, forward_axis]) or np.isnan(ankle_v[i1, forward_axis]):
            continue
        step_len = float(abs(ankle_v[i1, forward_axis] - ankle_v[i0, forward_axis]))
        step_time = float(t[i1] - t[i0])
        step_lengths_v.append(step_len)
        step_times_v.append(step_time)

    # --- Caliscope: длина и время шага ---
    step_lengths_c: List[float] = []
    step_times_c: List[float] = []
    for k in range(events_c.size - 1):
        i0 = int(events_c[k])
        i1 = int(events_c[k + 1])
        if i1 <= i0:
            continue
        if np.isnan(ankle_c[i0, forward_axis]) or np.isnan(ankle_c[i1, forward_axis]):
            continue
        step_len = float(abs(ankle_c[i1, forward_axis] - ankle_c[i0, forward_axis]))
        step_time = float(t[i1] - t[i0])
        step_lengths_c.append(step_len)
        step_times_c.append(step_time)

    if not step_lengths_v or not step_lengths_c or not step_times_v or not step_times_c:
        return metrics

    mean_step_length_v = float(np.mean(step_lengths_v))
    mean_step_length_c = float(np.mean(step_lengths_c))
    mean_step_time_v = float(np.mean(step_times_v))
    mean_step_time_c = float(np.mean(step_times_c))

    cadence_v = 60.0 / mean_step_time_v if mean_step_time_v > 0.0 else float("nan")
    cadence_c = 60.0 / mean_step_time_c if mean_step_time_c > 0.0 else float("nan")

    length_error = mean_step_length_c - mean_step_length_v
    length_error_rel = 100.0 * length_error / mean_step_length_v if mean_step_length_v > 0.0 else float("nan")
    cadence_error = cadence_c - cadence_v

    metrics["mean_step_length_vicon_m"] = mean_step_length_v
    metrics["mean_step_length_caliscope_m"] = mean_step_length_c
    metrics["step_length_error_m"] = length_error
    metrics["step_length_error_rel_percent"] = length_error_rel
    metrics["mean_step_time_vicon_s"] = mean_step_time_v
    metrics["mean_step_time_caliscope_s"] = mean_step_time_c
    metrics["cadence_vicon_steps_per_min"] = cadence_v
    metrics["cadence_caliscope_steps_per_min"] = cadence_c
    metrics["cadence_error_steps_per_min"] = cadence_error

    return metrics

def process_spatiotemporal_all_pairs(
    pairs: List[Tuple[str, str]],
    vicon_fps: float = 100.0,
    cal_fps: float = 20.0,
    output_summary_path: str = "spatiotemporal_summary.csv",
) -> pd.DataFrame:
    """
    Обрабатывает все пары файлов и сохраняет сводку по пространственно-временным
    параметрам шага в CSV:
      trial,
      mean_step_length_vicon_m, mean_step_length_caliscope_m,
      step_length_error_m, step_length_error_rel_percent,
      mean_step_time_vicon_s, mean_step_time_caliscope_s,
      cadence_vicon_steps_per_min, cadence_caliscope_steps_per_min,
      cadence_error_steps_per_min
    """
    rows: List[Dict[str, object]] = []

    for idx, (vicon_path, cal_path) in enumerate(pairs, start=1):
        print(f"Спат.-врем. параметры, пара {idx}:")
        result = compute_errors_for_trial(
            vicon_path,
            cal_path,
            vicon_fps=vicon_fps,
            cal_fps=cal_fps,
            event_mode="xcorr",
        )
        st_metrics = compute_spatiotemporal_for_trial(result)
        st_metrics_with_trial: Dict[str, object] = {"trial": f"trial_{idx}"}
        st_metrics_with_trial.update(st_metrics)
        rows.append(st_metrics_with_trial)

    df = pd.DataFrame(rows)
    df.to_csv(output_summary_path, index=False)
    print(f"Сводка по пространственно-временным параметрам сохранена в {output_summary_path}")
    return df

# ---------- Точка входа ----------

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pairs: List[Tuple[str, str]] = [
        (os.path.join(data_dir, "1.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_1.csv")),
        (os.path.join(data_dir, "2.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_2.csv")),
        (os.path.join(data_dir, "3.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_3.csv")),
        (os.path.join(data_dir, "4.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_4.csv")),
        (os.path.join(data_dir, "5.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_5.csv")),
        (os.path.join(data_dir, "6.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_6.csv")),
        (os.path.join(data_dir, "7.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_7.csv")),
        (os.path.join(data_dir, "8.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_8.csv")),
        (os.path.join(data_dir, "9.xlsx"),  os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_9.csv")),
        (os.path.join(data_dir, "10.xlsx"), os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_10.csv")),
    ]

    summary_path = os.path.join(results_dir, "results_summary.csv")
    df_summary = process_all_pairs(
        pairs,
        vicon_fps=100.0,
        cal_fps=20.0,
        output_summary_path=summary_path,
    )
    print(df_summary)

    spatio_path = os.path.join(results_dir, "spatiotemporal_summary.csv")
    df_spatio = process_spatiotemporal_all_pairs(
        pairs,
        vicon_fps=100.0,
        cal_fps=20.0,
        output_summary_path=spatio_path,
    )
    print(df_spatio)

    # Дополнительно: графики Z для PELVIS и RANK по всем парам
    plot_z_for_joint_all_trials(
        pairs=pairs,
        joint="PELVIS",
        vicon_fps=100.0,
        cal_fps=20.0,
        event_mode="xcorr",
        output_dir=os.path.join(results_dir, "z_pelvis"),
    )
    plot_z_for_joint_all_trials(
        pairs=pairs,
        joint="RANK",
        vicon_fps=100.0,
        cal_fps=20.0,
        event_mode="xcorr",
        output_dir=os.path.join(results_dir, "z_rank"),
    )


if __name__ == "__main__":
    main()

    # при желании можно оставить отладку для первой пары
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    vicon_path = os.path.join(data_dir, "1.xlsx")
    cal_path = os.path.join(data_dir, "xyz_SIMPLE_HOLISTIC_labelled_1.csv")

    debug_plot_jump(vicon_path, cal_path, vicon_fps=100.0, cal_fps=20.0)
    debug_plot_aligned_joint(
        vicon_path,
        cal_path,
        joint="RANK",
        vicon_fps=100.0,
        cal_fps=20.0,
        event_mode="xcorr",
    )
