"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

import numpy as np
import pytest
import requests
import torch
from syrupy.extensions.amber import AmberSnapshotExtension

if TYPE_CHECKING:
    from syrupy.types import SerializableData, SerializedData, SnapshotIndex

DEFAULT_RTOL = 1.0e-03
DEFAULT_ATOL = 1.0e-03


class Approx:
    """
    Wrapper object for approximately compared numpy arrays.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list],
        *,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to np.array")

        self.rtol: float = rtol if rtol is not None else DEFAULT_RTOL
        self.atol: float = atol if atol is not None else DEFAULT_ATOL
        self.tol_repr = True

    def __repr__(self) -> str:
        data = np.array_repr(self.data)
        data = "\n".join(f"\t{line}" for line in data.splitlines())
        tol_repr = ""
        if self.tol_repr:
            tol_repr = f", \n\trtol={self.rtol}, \n\tatol={self.atol}"
        return f"Approx(\n{data}{tol_repr}\n)"


class _ApproxNumpyFormatter:
    def __init__(self, data) -> None:
        self.data = data

    def __repr__(self) -> str:
        return Approx(
            self.data.expected,
            rtol=self.data.rel,
            atol=self.data.abs,
        ).__repr__()


def _try_parse_approx(data: "SerializableData") -> Optional[Approx]:
    """
    Parse the string representation of an Approx object.
    We can just use eval here, since we know the string is safe.
    """
    if not isinstance(data, str):
        return None

    data = data.strip()
    if not data.startswith("Approx("):
        return None

    approx = eval(
        data.replace("dtype=", "dtype=np."),
        {"Approx": Approx, "np": np},
        {"array": np.array},
    )
    if not isinstance(approx, Approx):
        return None

    return approx


class ApproxExtension(AmberSnapshotExtension):
    """
    By default, syrupy uses the __repr__ of the expected (snapshot) and actual values
    to serialize them into strings. Then, it compares the strings to see if they match.

    However, this behavior is not ideal for comparing floats/ndarrays. For example,
    if we have a snapshot with a float value of 0.1, and the actual value is 0.10000000000000001,
    then the strings will not match, even though the values are effectively equal.

    To work around this, we override the serialize method to seralize the expected value
    into a special representation. Then, we override the matches function (which originally does a
    simple string comparison) to parse the expected and actual values into numpy arrays.
    Finally, we compare the arrays using np.allclose.
    """

    def matches(
        self,
        *,
        serialized_data: "SerializableData",
        snapshot_data: "SerializableData",
    ) -> bool:
        # if both serialized_data and snapshot_data are serialized Approx objects,
        # then we can load them as numpy arrays and compare them using np.allclose
        serialized_approx = _try_parse_approx(serialized_data)
        snapshot_approx = _try_parse_approx(snapshot_data)
        if serialized_approx is not None and snapshot_approx is not None:
            return np.allclose(
                snapshot_approx.data,
                serialized_approx.data,
                rtol=serialized_approx.rtol,
                atol=serialized_approx.atol,
            )

        return super().matches(
            serialized_data=serialized_data, snapshot_data=snapshot_data
        )

    def serialize(self, data, **kwargs):
        # we override the existing serialization behavior
        # of the `pytest.approx()` object to serialize it into a special string.
        if isinstance(data, type(pytest.approx(np.array(0.0)))):
            return super().serialize(_ApproxNumpyFormatter(data), **kwargs)
        elif isinstance(data, type(pytest.approx(0.0))):
            raise NotImplementedError("Scalar approx not implemented yet")
        return super().serialize(data, **kwargs)

    def write_snapshot(
        self, *, data: "SerializedData", index: "SnapshotIndex"
    ) -> None:
        # Right before writing to file, we update the serialized snapshot data
        # and remove the atol/rtol from the string representation.
        # This is an implementation detail, and is not necessary for the extension to work.
        # It just makes the snapshot files a bit cleaner.
        approx = _try_parse_approx(data)
        if approx is not None:
            approx.tol_repr = False
            data = self.serialize(approx)
        return super().write_snapshot(data=data, index=index)


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(ApproxExtension)


@pytest.fixture()
def torch_deterministic() -> Iterator[bool]:
    # Setup
    torch.use_deterministic_algorithms(True)
    print("TORCH IS NOW DETERMINSTIC")
    yield True  # Usability: prints `torch_deterministic=True` if a test fails
    # Tear down
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="session")
def tutorial_dataset_path(tmp_path_factory: Any) -> Path:
    """
    Download the tutorial dataset and extract it to a temporary directory.
    This directory will persist until restart to avoid eating bandwidth.
    """
    TUTORIAL_DATASET_URL = "http://dl.fbaipublicfiles.com/opencatalystproject/data/tutorial_data.tar.gz"

    tmpdir = tmp_path_factory.getbasetemp()

    response = requests.get(TUTORIAL_DATASET_URL, stream=True)
    assert response.status_code == 200

    tarfile.open(fileobj=response.raw, mode="r|gz").extractall(path=tmpdir)

    return tmpdir
