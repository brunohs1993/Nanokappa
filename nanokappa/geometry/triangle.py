# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from pydantic import BaseModel
from nanokappa.geometry import Vector3D, TriVectors


class Triangle(BaseModel):
    vertices : TriVectors

    def __init__(self):
        super.__init__()
        self.origin = self.vertices[0]
        self.normal = self.get_normal()
        self.basis = self.get_basis()
        self.plane_k = self.get_plane_k()

    def get_basis(self) -> TriVectors:
        b1 = self.vertices[1] - self.origin
        b2 = self.vertices[2] - self.origin
        return np.vstack(self.normal, b1, b2)

    def get_normal(self) -> Vector3D:
        v1 = self.vertices[1] - self.origin
        v2 = self.vertices[2] - self.origin
        return np.cross(v1, v2)

    def get_plane_k(self) -> float:
        return -np.sum(self.normal * self.origin)
