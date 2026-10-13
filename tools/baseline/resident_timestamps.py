"""Completed numerical statistics carrying checked metadata for optional GPU timestamp reuse."""

import numpy as np


class ResidentTimestamp(np.float64):
    def __new__(cls, value, owner, generation, position):
        result = super().__new__(cls, value)
        result.owner, result.generation, result.position = owner, generation, int(position)
        return result


class ResidentStatistics(np.ndarray):
    def __new__(cls, values, owner, generation):
        result = np.asarray(values).view(cls)
        result.owner, result.generation = owner, generation
        result.positions = np.arange(result.size, dtype=np.uint32).reshape(result.shape)
        return result

    def __array_finalize__(self, source):
        if source is not None:
            self.owner, self.generation, self.positions = getattr(source, "owner", None), getattr(source, "generation", None), getattr(source, "positions", None)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        positions = self.positions[key]
        if isinstance(result, np.ndarray):
            result.positions = positions
            return result
        return ResidentTimestamp(result, self.owner, self.generation, positions)
