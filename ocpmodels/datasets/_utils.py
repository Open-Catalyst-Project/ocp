"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from torch_geometric.data import Data


def rename_data_object_keys(
    data_object: Data, key_mapping: dict[str, str]
) -> Data:
    """Rename data object keys

    Args:
        data_object: data object
        key_mapping: dictionary specifying keys to rename and new names {prev_key: new_key}
    """
    for _property in key_mapping:
        # catch for test data not containing labels
        if _property in data_object:
            new_property = key_mapping[_property]
            if new_property not in data_object:
                data_object[new_property] = data_object[_property]
                del data_object[_property]

    return data_object
