"""
Data ingestion, vibration conditioning, and spectral feature extraction module.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sih.data.loader import GenericDataLoader, TripSequence
    from sih.data.vibration import VibrationConditioner
    from sih.data.spectral import DualBandSpectralExtractor

__all__ = [
    "GenericDataLoader",
    "TripSequence",
    "VibrationConditioner",
    "DualBandSpectralExtractor",
]


def __getattr__(name: str) -> Any:
    if name in ("GenericDataLoader", "TripSequence"):
        from sih.data.loader import GenericDataLoader, TripSequence
        return GenericDataLoader if name == "GenericDataLoader" else TripSequence
    if name == "VibrationConditioner":
        from sih.data.vibration import VibrationConditioner
        return VibrationConditioner
    if name == "DualBandSpectralExtractor":
        from sih.data.spectral import DualBandSpectralExtractor
        return DualBandSpectralExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

