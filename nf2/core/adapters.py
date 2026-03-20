from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable, Type


class GeometryAdapter(ABC):
    geometry = None

    @classmethod
    @abstractmethod
    def matches(cls, data_config, series=False):
        raise NotImplementedError

    def prepare_data_config(self, data_config, series=False):
        return deepcopy(data_config)

    @abstractmethod
    def create_data_module(self, data_config):
        raise NotImplementedError

    @abstractmethod
    def create_series_data_module(self, data_config, current_step):
        raise NotImplementedError


_ADAPTERS = []


def register_geometry_adapter(adapter_cls: Type[GeometryAdapter]):
    _ADAPTERS.append(adapter_cls)
    return adapter_cls


def iter_geometry_adapters() -> Iterable[Type[GeometryAdapter]]:
    return tuple(_ADAPTERS)


def resolve_geometry_adapter(data_config, series=False):
    matches = [adapter_cls for adapter_cls in _ADAPTERS if adapter_cls.matches(data_config, series=series)]
    if len(matches) == 1:
        return matches[0]()
    if len(matches) > 1:
        explicit_type = data_config.get("type")
        for adapter_cls in matches:
            if adapter_cls.geometry == explicit_type:
                return adapter_cls()
        return matches[0]()
    raise ValueError(
        "Unable to resolve geometry adapter for data configuration. "
        f"Available adapters: {[adapter.geometry for adapter in _ADAPTERS]}"
    )
