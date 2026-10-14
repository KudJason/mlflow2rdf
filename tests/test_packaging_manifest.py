"""源码层打包清单测试 —— 在源码里就能跑（CI 的 wheel job 才是权威，这里只是快速护栏）。"""
from pathlib import Path
import tomllib


def test_pyproject_declares_config_package_data():
    """pyproject.toml 的 package-data 必须包含 config 数据文件。

    这是 0.2.3 掉文件的根因：声明里只有 shapes，没有 config。
    """
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pkg = data["tool"]["setuptools"]["package-data"]["mlflow2rdf"]
    joined = " ".join(pkg)
    assert "config/*.yaml" in joined, f"package-data 缺少 'config/*.yaml'，实际: {pkg}"
    assert "config/*.ini" in joined, f"package-data 缺少 'config/*.ini'（morph_kgc.ini 进不了 wheel）"


def test_config_files_present_in_source_tree():
    """源码树里这些文件必须存在（没有它们 wheel 就无从打包）。"""
    from mlflow2rdf.config_loader import _CONFIG_FILES
    root = Path("mlflow2rdf/config")
    missing = [fn for fn in _CONFIG_FILES.values() if not (root / fn).exists()]
    assert not missing, f"源码树缺少 config 文件: {missing}"
