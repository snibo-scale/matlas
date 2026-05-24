"""Torso / pelvis assembly component."""

from __future__ import annotations

from components.base import Component, Mount


class TorsoAssembly(Component):
    """Pelvis and central revolute mass (part_2)."""

    BASE_MOUNT = Mount(parent="base")

    @classmethod
    def load(cls) -> TorsoAssembly:
        return cls.from_package("torso_assembly")
