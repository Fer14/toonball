"""MotionAGFormer (Mehraban et al., WACV 2024), vendored from
https://github.com/TaatiTeam/MotionAGFormer under the Apache License 2.0 (see LICENSE).
Changes: package-relative imports, timm's DropPath replaced by droppath.py."""

from .model import MotionAGFormer

__all__ = ["MotionAGFormer"]
