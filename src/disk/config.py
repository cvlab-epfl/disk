from importlib.machinery import SourceFileLoader
from types import ModuleType


def load_config(config_path: str) -> ModuleType:
    if not config_path.endswith(".py"):
        raise ValueError(f"{config_path=} does not end in .py")

    return SourceFileLoader("config", config_path).load_module()
