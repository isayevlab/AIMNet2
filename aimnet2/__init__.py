from . import _version
from aimnet2.utils import load_model


__all__ = [load_model]

__version__ = _version.get_versions()['version']
