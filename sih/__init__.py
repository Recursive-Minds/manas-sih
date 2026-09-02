"""
Smartphone Intelligent Dead Reckoning (IDR) with GNSS Fusion.
"""

# Ensure default algorithms are registered
import sih.calibration.mount
import sih.fusion.naive
import sih.fusion.es_ekf
import sih.velocity.ai_estimator

__version__ = "0.1.0"
