import numpy as np

import cli


def test_pool_curves_concatenates_cycles_across_pairs():
    # two pairs, each contributing left_knee cycle matrices -> one ensemble mean
    a = {"cal_cycles": {"left_knee": np.full((2, 101), 10.0)},
         "vic_cycles": {"left_knee": np.full((3, 101), 10.0)}}
    b = {"cal_cycles": {"left_knee": np.full((1, 101), 20.0)},
         "vic_cycles": {"left_knee": np.full((1, 101), 20.0)}}
    cal_curves, vic_curves = cli._pool_curves([a, b])
    assert "left_knee" in cal_curves
    # cal ensemble mean = mean of 2x10 + 1x20 over 3 cycles = 13.333...
    assert cal_curves["left_knee"][0] == np.float64(np.mean([10, 10, 20]))
    assert vic_curves["left_knee"][0] == np.float64(np.mean([10, 10, 10, 20]))
