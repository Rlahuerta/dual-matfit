import pytest

import dualmatfit.optimization.drivers as drivers


def test_get_ipyopt_minimizer_class_has_clear_optional_dependency_error(monkeypatch):
    monkeypatch.setattr(drivers, "_IpyoptMinimizer", None)
    monkeypatch.setattr(
        drivers,
        "_IPYOPT_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'ipyopt'"),
    )

    with pytest.raises(ImportError, match=r"\.\[ipopt\]"):
        drivers._get_ipyopt_minimizer_class()
