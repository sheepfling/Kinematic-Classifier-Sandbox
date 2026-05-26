from __future__ import annotations

from typing import Any


class FeatureValueMappingMixin:
    feature_values: dict[str, float]

    def feature_value(self, name: str) -> float:
        return self.feature_values[name]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.feature_values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_values)
