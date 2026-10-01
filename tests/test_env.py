"""T0: 环境与脚手架自检。"""
import socg  # noqa: F401
from socg.constants import KB, kT


def test_kt_value():
    # kT(300 K) ≈ 2.4943 kJ/mol（相对误差 1e-4）
    assert abs(kT(300.0) - 2.4943) / 2.4943 < 1e-4


def test_kt_is_linear():
    assert abs(kT(300.0) - KB * 300.0) < 1e-12


def test_package_version():
    assert isinstance(socg.__version__, str)
