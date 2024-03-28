import pytest
import torch
from torch_geometric.data import Data

from ocpmodels.datasets._utils import rename_data_object_keys


@pytest.fixture()
def pyg_data():
    return Data(rand_tensor=torch.rand((3, 3)))


def test_rename_data_object_keys(pyg_data):
    assert "rand_tensor" in pyg_data.keys
    key_mapping = {"rand_tensor": "random_tensor"}
    pyg_data = rename_data_object_keys(pyg_data, key_mapping)
    assert "rand_tensor" not in pyg_data.keys
    assert "random_tensor" in pyg_data.keys
