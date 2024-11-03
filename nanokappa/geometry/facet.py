# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

from pydantic import computed_field, model_validator
from functools import cached_property
import numpy as np
from nanokappa.base import BaseModel, Vector3D, VectorList3D, VectorList2D
from nanokappa.geometry import Triangle


class Facet(BaseModel, frozen=True):
    faces: list[Triangle]

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

    @computed_field
    @cached_property
    def n_faces(self) -> int:
        return len(self.faces)

    @computed_field
    @cached_property
    def normal(self) -> Vector3D:
        return self.faces[0].normal

    @computed_field
    @cached_property
    def origin(self) -> Vector3D:
        return self.vertices[0]

    @computed_field
    @cached_property
    def vertices(self) -> VectorList3D:
        return np.unique(np.vstack([tri.vertices for tri in self.faces]), axis=0)

    @computed_field
    @cached_property
    def bounds(self) -> VectorList3D:
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    @computed_field
    @cached_property
    def faces_vertices(self) -> VectorList3D:
        fv = np.zeros((self.n_faces, 3), dtype=int)
        for i, tri in enumerate(self.faces):
            for j, v in enumerate(tri.vertices):
                fv[i, j] = np.all(v == self.vertices, axis=1).nonzero()[0]
        return fv

    @computed_field
    @cached_property
    def edges(self) -> VectorList2D:
        edges = np.zeros((0, 2), dtype=int)
        for v in self.faces_vertices:
            edges = np.vstack((edges, [v[[0, 1]], v[[0, 2]], v[[1, 2]]]))
        return np.sort(np.unique(edges, axis=0), axis=1)

    @computed_field
    @cached_property
    def faces_edges(self) -> VectorList3D:
        fe = np.zeros((self.n_faces, 3), dtype=int)
        for i, fv in enumerate(self.faces_vertices):
            inface = [np.all([ei in fv for ei in e]) for e in self.edges]
            fe[i, :] = np.arange(len(inface))[inface]
        return np.sort(fe, axis=1)

    @computed_field
    @cached_property
    def n_edges(self) -> int:
        return len(self.edges)

    @computed_field
    @cached_property
    def border_edges(self) -> np.ndarray:
        unique, counts = np.unique(self.faces_edges, return_counts=True)
        return unique[counts == 1]

    @computed_field
    @cached_property
    def area(self) -> int:
        return sum([tri.area for tri in self.faces])

    @computed_field
    @cached_property
    def centroid(self) -> int:
        wfc = [tri.area * tri.centroid for tri in self.faces]
        return np.sum(wfc, axis=0) / self.area
    
    
