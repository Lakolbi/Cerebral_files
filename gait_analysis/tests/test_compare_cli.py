import json

import cli


def test_compare_writes_report(tmp_path, monkeypatch):
    from tests.test_compare_pipeline import _walking_df

    cal = _walking_df()
    vic = cal.copy()
    vic.attrs["fps"] = 100.0

    monkeypatch.setattr(cli, "load_caliscope_session", lambda *a, **k: cal)
    monkeypatch.setattr(cli, "_load_vicon_mapped", lambda *a, **k: vic)

    out = tmp_path / "comparison_report.json"
    res = cli.compare("ignored_session", "ignored.xlsx", "SIMPLE_HOLISTIC", str(out))
    assert out.exists()
    saved = json.loads(out.read_text())
    assert "angle" in saved and "position" in saved
    assert res["pair_id"]
