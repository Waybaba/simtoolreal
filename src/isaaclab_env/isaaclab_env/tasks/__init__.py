"""Task registry for local Isaac Lab environments."""

import importlib
import pkgutil

_BLACKLIST_PKGS = ["utils", ".mdp"]


def _import_packages(package_name: str, blacklist_pkgs: list[str]) -> None:
    package = importlib.import_module(package_name)
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if any(blacklist_pkg in info.name for blacklist_pkg in blacklist_pkgs):
            continue
        if info.ispkg:
            importlib.import_module(info.name)


_import_packages(__name__, _BLACKLIST_PKGS)
