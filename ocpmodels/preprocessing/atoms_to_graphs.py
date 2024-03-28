"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import Optional, Sequence

import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from torch_geometric.data import Data

from ocpmodels.common.utils import collate

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    AseAtomsAdaptor = None


try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_stress (bool): Return the stress with other properties. Default is False, so the stress will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_edges (bool): Return interatomic edges with other properties. Default is True, so edges will be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.
        r_pbc (bool): Return the periodic boundary conditions with other properties.
        Default is False, so the periodic boundary conditions will not be returned.
        r_data_keys (sequence of str, optional): Return values corresponding to given keys in atoms.info data with other
        properties. Default is None, so no data will be returned as properties.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_stress (bool): Return the stress with other properties. Default is False, so the stress will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_edges (bool): Return interatomic edges with other properties. Default is True, so edges will be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.
        r_pbc (bool): Return the periodic boundary conditions with other properties.
        Default is False, so the periodic boundary conditions will not be returned.
        r_data_keys (sequence of str, optional): Return values corresponding to given keys in atoms.info data with other
        properties. Default is None, so no data will be returned as properties.
    """

    def __init__(
        self,
        max_neigh: int = 200,
        radius: int = 6,
        r_energy: bool = False,
        r_forces: bool = False,
        r_distances: bool = False,
        r_edges: bool = True,
        r_fixed: bool = True,
        r_pbc: bool = False,
        r_stress: bool = False,
        r_data_keys: Optional[Sequence[str]] = None,
    ) -> None:
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_stress = r_stress
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges
        self.r_pbc = r_pbc
        self.r_data_keys = r_data_keys

    def _get_neighbors_pymatgen(self, atoms: ase.Atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        if AseAtomsAdaptor is None:
            raise RuntimeError(
                "Unable to import pymatgen.io.ase.AseAtomsAdaptor. Make sure pymatgen is properly installed."
            )

        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(self, atoms: ase.Atoms, sid=None):
        """Convert a single atomic structure to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

            sid (uniquely identifying object): An identifier that can be used to track the structure in downstream
            tasks. Common sids used in OCP datasets include unique strings or integers.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]
        # initialized to torch.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = torch.Tensor(atoms.get_tags())

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )

        # Optionally add a systemid (sid) to the object
        if sid is not None:
            data.sid = sid

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.energy = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.forces = forces
        if self.r_stress:
            stress = torch.Tensor(
                atoms.get_stress(apply_constraint=False, voigt=False)
            )
            data.stress = stress
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms, dtype=torch.int)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx
        if self.r_pbc:
            data.pbc = torch.tensor(atoms.pbc)
        if self.r_data_keys is not None:
            for data_key in self.r_data_keys:
                data[data_key] = (
                    atoms.info[data_key]
                    if isinstance(atoms.info[data_key], (int, float))
                    else torch.Tensor(atoms.info[data_key])
                )

        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path: Optional[str] = None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list
