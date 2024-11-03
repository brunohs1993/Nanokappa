# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from pydantic import computed_field
from functools import cached_property
from nanokappa.base import BaseModel, TriVectors, Vector3D


class Triangle(BaseModel, frozen=True):
    vertices: TriVectors[np.float64]

    @computed_field
    @cached_property
    def origin(self) -> Vector3D:
        return self.vertices[0]

    @computed_field
    @cached_property
    def basis(self) -> TriVectors:
        b1 = self.vertices[1] - self.origin
        b2 = self.vertices[2] - self.origin
        return np.vstack((self.normal, b1, b2))

    @computed_field
    @cached_property
    def normal(self) -> Vector3D:
        v1 = self.vertices[1] - self.origin
        v2 = self.vertices[2] - self.origin
        cross = np.cross(v1, v2)
        return cross / np.linalg.norm(cross)

    @computed_field
    @cached_property
    def plane_k(self) -> float:
        return -np.sum(self.normal * self.origin)

    @computed_field
    @cached_property
    def bounds(self) -> float:
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    @computed_field
    @cached_property
    def area(self) -> float:
        v1 = self.vertices[1] - self.origin
        v2 = self.vertices[2] - self.origin
        cross = np.cross(v1, v2)
        return np.linalg.norm(cross) / 2

    @computed_field
    @cached_property
    def centroid(self) -> Vector3D:
        return np.mean(self.vertices, axis=0)

