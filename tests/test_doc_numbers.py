"""
tests/test_doc_numbers.py
-------------------------
Pytest wrapper for documentation number integrity and registry checking.
Ensures zero untraceable numbers, zero conflicts, and zero unallowlisted numbers.
"""

from scripts.check_number_registry import (
    check_registry_integrity,
    check_label_consistency,
    check_unregistered_doc_numbers,
)


def test_registry_recomputation():
    """Verify that all registered numbers in docs/NUMBER_SOURCES.json recompute from raw files."""
    assert check_registry_integrity() is True, "Registry recomputation failed"


def test_cross_document_label_consistency():
    """Verify that headline metric labels have consistent values across documentation."""
    assert check_label_consistency() is True, "Cross-document metric label consistency failed"


def test_unregistered_doc_numbers():
    """Verify that no unregistered or unallowlisted numbers exist in in-scope documentation."""
    assert check_unregistered_doc_numbers() is True, "Found unregistered numbers in documentation"
