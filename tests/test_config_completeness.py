"""
完整性测试 —— 任何一个 config 文件缺失都必须让测试失败。

为什么需要这个文件
------------------
0.2.3 发布到 PyPI 的 wheel 里，`mlflow2rdf/config/` **一个文件都没打进去**
（`pyproject.toml` 的 `package-data` 只声明了 `shapes/*.ttl`）。

后果不是崩溃，而是**静默降级**：
`config_loader._load()` 在文件不存在时只打一条 `logger.warning`，
然后**返回空字典 `{}`**。于是转换照跑不误，只是产出的 RDF 语义为空 ——
**一份看起来正常、实际是错的图**。

这个测试就是用来堵这个洞的：
    在**安装好的包**上（而不是源码树里）断言每一个 config 都能加载且非空。

⚠️ 关键：必须在**构建出的 wheel 安装之后**运行。
   在源码树里跑这个测试永远会通过（yaml 就在旁边），
   所以 CI 必须先 `pip install dist/*.whl`，再跑 pytest。
"""

import pytest

from mlflow2rdf.config_loader import load_config, reset_config


#: 允许为空的最小集合 —— 一个都没有就说明配置整批丢了
#: （故意不写死"全部非空"：有些配置在任何合理版本里都可能合法地是空 dict）
_REQUIRED_NONEMPTY = (
    "property_routing",
    "paradigm_labels",
    "metric_types",
    "task_types",
)


@pytest.fixture(autouse=True)
def _fresh():
    reset_config()
    yield
    reset_config()


def test_config_dir_exists():
    """config 目录本身必须存在于**安装位置**。"""
    from pathlib import Path
    import mlflow2rdf

    cfg_dir = Path(mlflow2rdf.__file__).parent / "config"
    assert cfg_dir.is_dir(), (
        f"config 目录不存在: {cfg_dir}\n"
        "这几乎总是打包问题：pyproject.toml 的 "
        "[tool.setuptools.package-data] 里漏了 'config/*.yaml' 与 'config/*.ini'"
    )


def test_no_config_file_is_missing():
    """逐个检查：注册在案的 config 文件都必须在磁盘上真实存在。

    直接查文件，不走 load_config —— 因为 load_config 会把缺失静默吞掉。
    """
    from pathlib import Path
    import mlflow2rdf
    from mlflow2rdf.config_loader import _CONFIG_FILES

    cfg_dir = Path(mlflow2rdf.__file__).parent / "config"
    missing = [fn for fn in _CONFIG_FILES.values() if not (cfg_dir / fn).exists()]
    assert not missing, (
        f"以下 config 文件在安装包里不存在: {missing}\n"
        f"查找目录: {cfg_dir}"
    )


def test_required_configs_load_nonempty():
    """关键配置必须加载出**非空**结果。

    空字典 = 静默降级。这个断言就是用来把"静默"变成"响"。
    """
    cfg = load_config()
    empty = [name for name in _REQUIRED_NONEMPTY if not cfg.get(name)]
    assert not empty, (
        f"以下配置加载为空: {empty}\n"
        "空配置不会抛错，只会让产出的 RDF 语义缺失 —— 必须当成失败。"
    )


def test_all_registered_configs_load_without_error():
    """所有注册的 config 都要能加载（哪怕内容为空也不该抛异常）。"""
    cfg = load_config()
    bad = []
    for name in cfg.available_configs():
        try:
            cfg.get(name)
        except Exception as e:  # noqa: BLE001
            bad.append((name, f"{type(e).__name__}: {e}"))
    assert not bad, f"以下配置加载抛异常: {bad}"


def test_task_types_has_expected_keys():
    """抽查 task_types.yaml 的结构 —— 打包丢文件时这里也会红。"""
    tt = load_config().get("task_types") or {}
    for key in ("task_types", "openml_measures"):
        assert key in tt, f"task_types.yaml 缺少 '{key}'，文件可能没被打进 wheel"
