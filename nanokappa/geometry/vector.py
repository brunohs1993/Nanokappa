# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

Vector3D = Annotated[npt.NDArray[DType], Literal[3]]
VectorList3D = Annotated[npt.NDArray[DType], [int, Literal[3]]]
TriVectors = Annotated[npt.NDArray[DType], Literal[3, 3]]
