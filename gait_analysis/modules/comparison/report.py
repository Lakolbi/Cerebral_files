"""Assemble the canonical comparison_report.json (angle + position layers)."""
import datetime as dt


def _df_to_keyed(df, key="joint"):
    out = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        out[d.pop(key)] = d
    return out


def build_report(angle_df, position_df, overlay: dict, meta: dict) -> dict:
    """Combine the angle table, position table, and overlay curves into one dict.

    meta: pair_id, model, caliscope_fps, vicon_fps, time_shift_s, scale, low_confidence.
    """
    angle = _df_to_keyed(angle_df)
    knee = angle.get("left_knee", angle.get("right_knee", {}))
    worst = max((v["verdict"] for v in angle.values()),
                key=lambda x: {"good": 0, "acceptable": 1, "poor": 2, "n/a": -1}.get(x, -1),
                default="n/a")
    summary = (f"{len(angle)} joints compared; knee ICC="
               f"{knee.get('icc', float('nan')):.3f}; worst verdict: {worst}")
    return {
        "pair_id": meta.get("pair_id"),
        "model": meta.get("model"),
        "caliscope_fps": meta.get("caliscope_fps"),
        "vicon_fps": meta.get("vicon_fps"),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "angle": angle,
        "angle_overlay": overlay,
        "position": {
            "joints": _df_to_keyed(position_df) if len(position_df) else {},
            "time_shift_s": meta.get("time_shift_s"),
            "scale": meta.get("scale"),
            "low_confidence": meta.get("low_confidence", False),
        },
        "verdict_summary": summary,
    }
