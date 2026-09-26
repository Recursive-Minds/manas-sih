"""
Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion.
"""

__version__ = "0.1.0"


def _register_defaults():
    import sih.calibration.mount
    import sih.fusion.naive
    import sih.fusion.es_ekf
    import sih.velocity.ai_estimator


def __getattr__(name: str):
    if name in ("calibration", "fusion", "velocity", "core", "data", "features", "models", "map", "handoff"):
        import importlib
        return importlib.import_module(f"sih.{name}")
    if name == "register_defaults":
        return _register_defaults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

