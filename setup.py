"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="ocp-models",
    version="0.0.3.5",
    description="Machine learning models for use in catalysis as part of the Open Catalyst Project",
    url="https://github.com/Open-Catalyst-Project/ocp",
    packages=find_packages(),
    include_package_data=True,
)
