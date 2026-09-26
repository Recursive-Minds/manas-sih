"""
tests/test_forced_shim_parity.py
---------------------------------
Verifies that:
1. Default laptop environment uses real scipy (signal, Rotation).
2. With SIH_FORCE_SCIPY_SHIM=1, all fallback imports strictly resolve to sih.core.scipy_shim.
3. Quick parity with SIH_FORCE_SCIPY_SHIM=1 and TFLite / ONNX predictor achieves < 0.05m endpoint diff.
"""

import os
import sys
import subprocess
import pytest


def test_default_environment_uses_real_scipy():
    """Verify that when SIH_FORCE_SCIPY_SHIM is not set, real scipy is used."""
    # Ensure env var is unset for this test
    os.environ.pop("SIH_FORCE_SCIPY_SHIM", None)

    import scipy.signal
    import scipy.spatial.transform
    from sih.core.scipy_shim import _HAVE_SCIPY
    from sih.features import streaming
    from sih.calibration import mount
    from sih.fusion import es_ekf

    assert _HAVE_SCIPY is True, "Laptop environment must have scipy available"
    assert streaming.signal.__name__ == "scipy.signal", "Streaming feature extractor must use scipy.signal by default"
    assert mount.R is scipy.spatial.transform.Rotation, "Mount calibrator must use scipy Rotation by default"
    assert es_ekf.R is scipy.spatial.transform.Rotation, "ES-EKF must use scipy Rotation by default"


def test_forced_shim_imports():
    """Verify that SIH_FORCE_SCIPY_SHIM=1 directs all imports to scipy_shim."""
    code = """
import os
os.environ["SIH_FORCE_SCIPY_SHIM"] = "1"

from sih.core.scipy_shim import _HAVE_SCIPY, RotationShim
from sih.features import streaming
from sih.calibration import mount
from sih.fusion import es_ekf, naive
import server.engine_adapter as adapter

assert _HAVE_SCIPY is False, "scipy_shim._HAVE_SCIPY must be False when forced"
assert streaming.signal.__name__ == "sih.core.scipy_shim", f"Expected scipy_shim, got {streaming.signal.__name__}"
assert mount.R is RotationShim, f"Expected RotationShim, got {mount.R}"
assert es_ekf.R is RotationShim, f"Expected RotationShim, got {es_ekf.R}"
assert naive.R is RotationShim, f"Expected RotationShim, got {naive.R}"
print("FORCED_SHIM_IMPORTS_OK")
"""
    env = os.environ.copy()
    env["SIH_FORCE_SCIPY_SHIM"] = "1"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Forced shim import check failed: {result.stderr}"
    assert "FORCED_SHIM_IMPORTS_OK" in result.stdout


def test_forced_shim_quick_parity_tflite():
    """Verify that quick parity with forced shim and TFLite predictor passes < 0.05m."""
    env = os.environ.copy()
    env["SIH_FORCE_SCIPY_SHIM"] = "1"
    cmd = [
        sys.executable,
        os.path.join("scripts", "quick_parity.py"),
        "--raw",
        "--cpu",
        "--predictor",
        "tflite",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Quick parity failed: {result.stderr}"
    assert "All Scenarios Passed (<0.05m): True" in result.stdout
