# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

from pydantic import computed_field
from functools import cached_property
import numpy as np
from nanokappa.base import , VectorList3D, VectorList2D
from nanokappa.geometry import FaceGroup, Triangle, Facet
import networkx as nx


class Mesh(FaceGroup, frozen=False):
    @computed_field
    @cached_property
    def n_facets(self) -> int:
        return len(self.facets)

    
    


