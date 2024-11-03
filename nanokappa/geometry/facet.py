# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

from pydantic import model_validator
import numpy as np
from nanokappa.geometry import FaceGroup


class Facet(FaceGroup, frozen=True):
    @model_validator(mode="after")
    def are_coplanar(self):
        normals = np.vstack([tri.normal for tri in self.faces])
        dots = np.abs((normals[0] * normals).sum(axis=1))
        try:
            assert np.all(np.isclose(dots, 1))
        except AssertionError:
            raise ValueError("Triangles are not coplanar.")
        return self

    @model_validator(mode="after")
    def are_adjacent(self):
        _, counts = np.unique(self.faces_edges, return_counts=True)
        for i, fe in enumerate(self.faces_edges):
            try:
                assert np.any(counts[fe] >= 2)
            except AssertionError:
                raise ValueError(f"Triangle {i} is not adjacent.")
        return self
