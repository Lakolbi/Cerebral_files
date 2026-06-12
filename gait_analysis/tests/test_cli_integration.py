import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GAIT = REPO / "gait_analysis"
SESSION = GAIT / "data" / "caliscope_project" / "recordings" / "p1_3"

pytestmark = pytest.mark.skipif(not SESSION.exists(),
                                reason="real caliscope data not present")


def test_analyze_writes_gait_results(tmp_path):
    out = tmp_path / "p1_3_results.json"
    cmd = [sys.executable, str(GAIT / "cli.py"), "analyze",
           "--session", str(SESSION), "--model", "SIMPLE_HOLISTIC",
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["session_id"] == "p1_3"
    assert data["model"] == "SIMPLE_HOLISTIC"
    assert data["fps"] > 0
    assert "gait_events" in data and "spatiotemporal" in data
    assert "left_HS" in data["gait_events"]
    assert len(data["gait_events"]["left_HS"]) >= 3  # criterion #3: events found
    assert len(data["joint_angles_mean"]["left_knee"]) == 101


RECORDINGS = GAIT / "data" / "caliscope_project" / "recordings"
HAVE_P1 = all((RECORDINGS / f"p1_{i}").exists() for i in range(1, 6))


@pytest.mark.skipif(not HAVE_P1, reason="p1_1..p1_5 not present")
def test_reproducibility_computes_cv(tmp_path):
    out = tmp_path / "repro.json"
    cmd = [sys.executable, str(GAIT / "cli.py"), "reproducibility",
           "--recordings", str(RECORDINGS), "--model", "SIMPLE_HOLISTIC",
           "--sessions", "p1_1,p1_2,p1_3,p1_4,p1_5", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert "per_session" in data and "cv_percent" in data
    assert len(data["per_session"]) == 5
    assert "cadence_steps_per_min" in data["cv_percent"]
    assert all("n_left_cycles" in s for s in data["per_session"].values())
    assert data["cv_percent"]["cadence_steps_per_min"] is not None
    assert data["cv_percent"]["cadence_steps_per_min"] < 15.0  # criterion #6 (cadence)


@pytest.mark.skipif(not (SESSION / "port_1.mp4").exists(),
                    reason="p1_3 raw videos not present")
def test_produce_videos_cli_writes_marked_mp4s(tmp_path):
    cmd = [sys.executable, str(GAIT / "cli.py"), "produce-videos",
           "--session", str(SESSION), "--model", "SIMPLE_HOLISTIC",
           "--out", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    written = list(tmp_path.glob("port_*_marked.mp4"))
    assert len(written) >= 1
