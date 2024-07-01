"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import ast
import collections
import copy
import importlib
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from bisect import bisect
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter, segment_coo, segment_csr

import fairchem.core
from fairchem.core.modules.loss import AtomwiseL2Loss, L2MAELoss

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.nn.modules.module import _IncompatibleKeys


# copied from https://stackoverflow.com/questions/33490870/parsing-yaml-in-python-detect-duplicated-keys
# prevents loading YAMLS where keys have been overwritten
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(
                    f"Duplicate Key: {each_key!r} is found in YAML File.\n"
                    f"Error File location: {key_node.end_mark}"
                )
            mapping.add(each_key)
        return super().construct_mapping(node, deep)


def pyg2_data_transform(data: Data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(**{k: v for k, v in data.__dict__.items() if v is not None})

    return data


def save_checkpoint(
    state,
    checkpoint_dir: str = "checkpoints/",
    checkpoint_file: str = "checkpoint.pt",
) -> str:
    filename = os.path.join(checkpoint_dir, checkpoint_file)
    torch.save(state, filename)
    return filename


class Complete:
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def warmup_lr_lambda(current_step: int, optim_config):
    """Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    # keep this block for older configs that have warmup_epochs instead of warmup_steps
    # and lr_milestones are defined in epochs
    if (
        any(x < 100 for x in optim_config["lr_milestones"])
        or "warmup_epochs" in optim_config
    ):
        raise Exception(
            "ConfigError: please define lr_milestones in steps not epochs and define warmup_steps instead of warmup_epochs"
        )

    if current_step <= optim_config["warmup_steps"]:
        alpha = current_step / float(optim_config["warmup_steps"])
        return optim_config["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(optim_config["lr_milestones"], current_step)
        return pow(optim_config["lr_gamma"], idx)


def print_cuda_usage() -> None:
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print(
        "Max Memory Allocated:",
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def plot_histogram(data, xlabel: str = "", ylabel: str = "", title: str = ""):
    assert isinstance(data, list)

    # Preset
    fig = Figure(figsize=(5, 4), dpi=150)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    # Plot
    ax.hist(data, bins=20, rwidth=0.9, zorder=3)

    # Axes
    ax.grid(color="0.95", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout(pad=2)

    # Return numpy array
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))


# Override the collation method in `pytorch_geometric.data.InMemoryDataset`
def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
        elif isinstance(item[key], (int, float)):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


def add_edge_distance_to_graph(
    batch,
    device="cpu",
    dmin: float = 0.0,
    dmax: float = 6.0,
    num_gaussians: int = 50,
):
    # Make sure x has positions.
    if not all(batch.pos[0][:] == batch.x[0][-3:]):
        batch.x = torch.cat([batch.x, batch.pos.float()], dim=1)
    # First set computations to be tracked for positions.
    batch.x = batch.x.requires_grad_(True)
    # Then compute Euclidean distance between edge endpoints.
    pdist = torch.nn.PairwiseDistance(p=2.0)
    distances = pdist(
        batch.x[batch.edge_index[0]][:, -3:],
        batch.x[batch.edge_index[1]][:, -3:],
    )
    # Expand it using a gaussian basis filter.
    gdf_filter = torch.linspace(dmin, dmax, num_gaussians)
    var = gdf_filter[1] - gdf_filter[0]
    gdf_filter, var = gdf_filter.to(device), var.to(device)
    gdf_distances = torch.exp(-((distances.view(-1, 1) - gdf_filter) ** 2) / var**2)
    # Reassign edge attributes.
    batch.edge_weight = distances
    batch.edge_attr = gdf_distances.float()
    return batch


def _import_local_file(path: Path, *, project_root: Path) -> None:
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.parent.resolve()

    module_name = ".".join(
        path.absolute().relative_to(project_root.absolute()).with_suffix("").parts
    )
    logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


def setup_experimental_imports(project_root: Path) -> None:
    """
    Import selected directories of modules from the "experimental" subdirectory.

    If a file named ".include" is present in the "experimental" subdirectory,
    this will be read as a list of experimental subdirectories whose module
    (including in any subsubdirectories) should be imported.

    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    """
    experimental_dir = (project_root / "experimental").resolve()
    if not experimental_dir.exists() or not experimental_dir.is_dir():
        return

    experimental_files = []
    include_file = experimental_dir / ".include"

    if include_file.exists():
        with open(include_file) as f:
            include_dirs = [line.rstrip("\n") for line in f.readlines() if line.strip()]

        for inc_dir in include_dirs:
            experimental_files.extend(
                f.resolve().absolute()
                for f in (experimental_dir / inc_dir).rglob("*.py")
            )

    for f in experimental_files:
        _import_local_file(f, project_root=project_root)


def _get_project_root() -> Path:
    """
    Gets the root folder of the project (the "ocp" folder)
    :return: The absolute path to the project root.
    """
    from fairchem.core.common.registry import registry

    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("fairchem_core_root", no_warning=True)

    if root_folder is not None:
        assert isinstance(root_folder, str), "fairchem_core_root must be a string"
        root_folder = Path(root_folder).resolve().absolute()
        assert root_folder.exists(), f"{root_folder} does not exist"
        assert root_folder.is_dir(), f"{root_folder} is not a directory"
    else:
        root_folder = Path(__file__).resolve().absolute().parent.parent

    # root_folder is the "ocpmodes" folder, so we need to go up one more level
    return root_folder.parent


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports(config: dict | None = None) -> None:
    from fairchem.core.common.registry import registry

    skip_experimental_imports = (config or {}).get("skip_experimental_imports", False)

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        project_root = _get_project_root()
        logging.info(f"Project root: {project_root}")
        importlib.import_module("fairchem.core.common.logger")

        import_keys = ["trainers", "datasets", "models", "tasks"]
        for key in import_keys:
            for f in (project_root / "core" / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)

        if not skip_experimental_imports:
            setup_experimental_imports(project_root)
    finally:
        registry.register("imports_setup", True)


def dict_set_recursively(dictionary, key_sequence, val) -> None:
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def parse_value(value):
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        keys_concat, val = arg.removeprefix("--").split("=")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def load_config(path: str, previous_includes: list | None = None):
    if previous_includes is None:
        previous_includes = []
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = [*previous_includes, path]

    with open(path) as fp:
        direct_config = yaml.load(fp, Loader=UniqueKeyLoader)

    # Load config from included files.
    includes = direct_config.pop("includes") if "includes" in direct_config else []
    if not isinstance(includes, list):
        raise AttributeError(f"Includes must be a list, '{type(includes)}' provided")

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def build_config(args, args_override):
    config, duplicates_warning, duplicates_error = load_config(args.config_yml)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten config parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )

    # Some other flags.
    config["mode"] = args.mode
    config["identifier"] = args.identifier
    config["timestamp_id"] = args.timestamp_id
    config["seed"] = args.seed
    config["is_debug"] = args.debug
    config["run_dir"] = args.run_dir
    config["print_every"] = args.print_every
    config["amp"] = args.amp
    config["checkpoint"] = args.checkpoint
    config["cpu"] = args.cpu
    # Submit
    config["submit"] = args.submit
    config["summit"] = args.summit
    # Distributed
    config["local_rank"] = args.local_rank
    config["distributed_port"] = args.distributed_port
    config["world_size"] = args.num_nodes * args.num_gpus
    config["distributed_backend"] = args.distributed_backend
    config["noddp"] = args.no_ddp
    config["gp_gpus"] = args.gp_gpus

    # Check for overridden parameters.
    if args_override != []:
        overrides = create_dict_from_args(args_override)
        config, _ = merge_dicts(config, overrides)

    return config


def create_grid(base_config, sweep_file: str):
    def _flatten_sweeps(sweeps, root_key: str = "", sep: str = "."):
        flat_sweeps = []
        for key, value in sweeps.items():
            new_key = root_key + sep + key if root_key else key
            if isinstance(value, collections.MutableMapping):
                flat_sweeps.extend(_flatten_sweeps(value, new_key).items())
            else:
                flat_sweeps.append((new_key, value))
        return collections.OrderedDict(flat_sweeps)

    def _update_config(config, keys, override_vals, sep: str = "."):
        for key, value in zip(keys, override_vals):
            key_path = key.split(sep)
            child_config = config
            for name in key_path[:-1]:
                child_config = child_config[name]
            child_config[key_path[-1]] = value
        return config

    with open(sweep_file) as fp:
        sweeps = yaml.load(fp, Loader=UniqueKeyLoader)

    flat_sweeps = _flatten_sweeps(sweeps)
    keys = list(flat_sweeps.keys())
    values = list(itertools.product(*flat_sweeps.values()))

    configs = []
    for i, override_vals in enumerate(values):
        config = copy.deepcopy(base_config)
        config = _update_config(config, keys, override_vals)
        config["identifier"] = config["identifier"] + f"_run{i}"
        configs.append(config)
    return configs


def save_experiment_log(args, jobs, configs):
    log_file = args.logdir / "exp" / time.strftime("%Y-%m-%d-%I-%M-%S%p.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)
    with open(log_file, "w") as f:
        for job, config in zip(jobs, configs):
            print(
                json.dumps(
                    {
                        "config": config,
                        "slurm_id": job.job_id,
                        "timestamp": time.strftime("%I:%M:%S%p %Z %b %d, %Y"),
                    }
                ),
                file=f,
            )
    return log_file


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=None,
):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def get_pruned_edge_idx(
    edge_index, num_atoms: int, max_neigh: float = 1e9
) -> torch.Tensor:
    assert num_atoms is not None  # TODO: Shouldn't be necessary

    # removes neighbors > max_neigh
    # assumes neighbors are sorted in increasing distance
    _nonmax_idx_list = []
    for i in range(num_atoms):
        idx_i = torch.arange(len(edge_index[1]))[(edge_index[1] == i)][:max_neigh]
        _nonmax_idx_list.append(idx_i)
    return torch.cat(_nonmax_idx_list)


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level: int, max_level: int) -> None:
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record) -> bool:
        return self.min_level <= record.levelno < self.max_level


def setup_logging() -> None:
    root = logging.getLogger()

    # Perform setup only if logging has not been configured
    if not root.hasHandlers():
        root.setLevel(logging.INFO)

        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(SeverityLevelBetween(logging.INFO, logging.WARNING))
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    return segment_csr(num_neighbors, image_indptr)


def check_traj_files(batch, traj_dir) -> bool:
    if traj_dir is None:
        return False
    traj_dir = Path(traj_dir)
    sid_list = batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
    traj_files = [traj_dir / f"{sid}.traj" for sid in sid_list]
    return all(fl.exists() for fl in traj_files)


@contextmanager
def new_trainer_context(*, config: dict[str, Any], distributed: bool = False):
    from fairchem.core.common import distutils, gp_utils
    from fairchem.core.common.registry import registry

    if TYPE_CHECKING:
        from fairchem.core.tasks.task import BaseTask
        from fairchem.core.trainers import BaseTrainer

    @dataclass
    class _TrainingContext:
        config: dict[str, Any]
        task: BaseTask
        trainer: BaseTrainer

    setup_logging()
    original_config = config
    config = copy.deepcopy(original_config)

    if distributed:
        distutils.setup(config)
        if config["gp_gpus"] is not None:
            gp_utils.setup_gp(config)
    try:
        setup_imports(config)
        trainer_name = config.get("trainer", "ocp")
        # backwards compatibility for older configs
        if trainer_name in ["forces", "equiformerv2_forces"]:
            task_name = "s2ef"
        elif trainer_name in ["energy", "equiformerv2_energy"]:
            task_name = "is2re"
        else:
            task_name = "ocp"

        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer not found"
        trainer = trainer_cls(
            task=config.get("task", {}),
            model=config["model"],
            outputs=config.get("outputs", {}),
            dataset=config["dataset"],
            optimizer=config["optim"],
            loss_functions=config.get("loss_functions", {}),
            evaluation_metrics=config.get("evaluation_metrics", {}),
            identifier=config["identifier"],
            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", "./"),
            is_debug=config.get("is_debug", False),
            print_every=config.get("print_every", 10),
            seed=config.get("seed", 0),
            logger=config.get("logger", "wandb"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            noddp=config.get("noddp", False),
            name=task_name,
            gp_gpus=config.get("gp_gpus"),
        )

        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task not found"
        task = task_cls(config)
        start_time = time.time()
        ctx = _TrainingContext(config=original_config, task=task, trainer=trainer)
        yield ctx
        distutils.synchronize()
        if distutils.is_master():
            logging.info(f"Total time taken: {time.time() - start_time}")
    finally:
        if distributed:
            distutils.cleanup()


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    from fairchem.core.modules.scaling.scale_factor import ScaleFactor

    try:
        scale = model.get_submodule(name)
        if not isinstance(scale, ScaleFactor):
            return None
        return scale
    except AttributeError:
        return None


def _report_incompat_keys(
    model: nn.Module,
    keys: _IncompatibleKeys,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    # filter out the missing scale factor keys for the new scaling factor module
    missing_keys: list[str] = []
    for full_key_name in keys.missing_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(model, parent_module_name)
        if scale_factor is not None:
            continue
        missing_keys.append(full_key_name)

    # filter out unexpected scale factor keys that remain from the old scaling modules
    unexpected_keys: list[str] = []
    for full_key_name in keys.unexpected_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(model, parent_module_name)
        if scale_factor is not None:
            continue
        unexpected_keys.append(full_key_name)

    error_msgs = []
    if len(unexpected_keys) > 0:
        error_msgs.insert(
            0,
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join(f'"{k}"' for k in unexpected_keys)
            ),
        )
    if len(missing_keys) > 0:
        error_msgs.insert(
            0,
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join(f'"{k}"' for k in missing_keys)
            ),
        )

    if len(error_msgs) > 0:
        error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
        )
        if strict:
            raise RuntimeError(error_msg)
        logging.warning(error_msg)

    return missing_keys, unexpected_keys


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    incompat_keys = module.load_state_dict(state_dict, strict=False)  # type: ignore
    return _report_incompat_keys(module, incompat_keys, strict=strict)


def scatter_det(*args, **kwargs):
    from fairchem.core.common.registry import registry

    if registry.get("set_deterministic_scatter", no_warning=True):
        torch.use_deterministic_algorithms(mode=True)

    out = scatter(*args, **kwargs)

    if registry.get("set_deterministic_scatter", no_warning=True):
        torch.use_deterministic_algorithms(mode=False)

    return out


def get_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "-C", fairchem.core.__path__[0], "describe", "--always"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode("ascii")
        )
    # catch instances where code is not being run from a git repo
    except Exception:
        commit_hash = None

    return commit_hash


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def update_config(base_config):
    """
    Configs created prior to FAIRChem/OCP 2.0 are organized a little different than they
    are now. Update old configs to fit the new expected structure.
    """
    ### TODO: better format check for older configs
    # some configs have a loss_functions key with an empty dictionary, those need to be updated as well
    if len(base_config.get("loss_functions", {})) > 0:
        return base_config

    logging.warning(
        "Detected old config, converting to new format. Consider updating to avoid potential incompatibilities."
    )

    # do we need a copy?
    config = copy.deepcopy(base_config)

    # initial fairchem/ocp 2.0 configs renamed loss_functions -> loss_fns and evaluation_metrics -> eval_metrics
    # so some checkpoints may have configs in new format with the exception of renamed loss_funs and eval_metrics
    if "loss_fns" in config:
        config["loss_functions"] = config.pop("loss_fns")
        if "eval_metrics" in config:
            config["evaluation_metrics"] = config.pop("eval_metrics")
        return config

    # If config["dataset"]["format"] is missing, get it from the task (legacy location).
    # If it is not there either, default to LMDB.
    config["dataset"]["format"] = config["dataset"].get(
        "format", config["task"].get("dataset", "lmdb")
    )

    ### Read task based off config structure, similar to OCPCalculator.
    if config["task"]["dataset"] in [
        "trajectory_lmdb",
        "lmdb",
        "trajectory_lmdb_v2",
        "oc22_lmdb",
        "ase_read",
        "ase_read_multi",
        "ase_db",
    ]:
        task = "s2ef"
    elif config["task"]["dataset"] == "single_point_lmdb":
        task = "is2re"
    else:
        raise NotImplementedError

    if task == "is2re":
        ### Define loss functions
        _loss_fns = [
            {
                "energy": {
                    "fn": config["optim"].get("loss_energy", "mae"),
                    "coefficient": config["optim"].get("energy_coefficient", 1),
                },
            }
        ]
        ### Define evaluation metrics
        _eval_metrics = {
            "metrics": {"energy": ["mae", "mse", "energy_within_threshold"]},
        }
        if "primary_metric" in config["task"]:
            _eval_metrics["primary_metric"] = config["task"]["primary_metric"]
        ### Define outputs
        _outputs = {"energy": {"level": "system"}}
        ### Define key mapping
        config["dataset"]["key_mapping"] = {"y_relaxed": "energy"}
    elif task == "s2ef":
        ### Define loss functions
        _loss_fns = [
            {
                "energy": {
                    "fn": config["optim"].get("loss_energy", "mae"),
                    "coefficient": config["optim"].get("energy_coefficient", 1),
                },
            },
            {
                "forces": {
                    "fn": config["optim"].get("loss_forces", "l2mae"),
                    "coefficient": config["optim"].get("force_coefficient", 30),
                },
            },
        ]
        ### Define evaluation metrics
        _eval_metrics = {
            "metrics": {
                "misc": ["energy_forces_within_threshold"],
                "energy": ["mae"],
                "forces": [
                    "forcesx_mae",
                    "forcesy_mae",
                    "forcesz_mae",
                    "mae",
                    "cosine_similarity",
                    "magnitude_error",
                ],
            },
        }
        if "primary_metric" in config["task"]:
            _eval_metrics["primary_metric"] = config["task"]["primary_metric"]
        elif "eval_metrics" in config and "primary_metric" in config["eval_metrics"]:
            _eval_metrics["primary_metric"] = config["eval_metrics"]["primary_metric"]
        ### Define outputs
        _outputs = {
            "energy": {"level": "system"},
            "forces": {
                "level": "atom",
                "train_on_free_atoms": (
                    config["task"].get("train_on_free_atoms", False)
                ),
                "eval_on_free_atoms": (config["task"].get("eval_on_free_atoms", True)),
            },
        }
        ### Define key mapping
        config["dataset"]["key_mapping"] = {"y": "energy", "force": "forces"}

    if config["dataset"].get("normalize_labels", False):
        normalizer = {
            "energy": {
                "mean": config["dataset"].get("target_mean", 0),
                "stdev": config["dataset"].get("target_std", 1),
            },
            "forces": {
                "mean": config["dataset"].get("grad_target_mean", 0),
                "stdev": config["dataset"].get("grad_target_std", 1),
            },
        }

        transforms = config["dataset"].get("transforms", {})
        transforms["normalizer"] = normalizer
        config["dataset"]["transforms"] = transforms

    ### Update config
    config.update({"loss_functions": _loss_fns})
    config.update({"evaluation_metrics": _eval_metrics})
    config.update({"outputs": _outputs})

    return config


def get_loss_module(loss_name):
    if loss_name in ["l1", "mae"]:
        loss_fn = nn.L1Loss()
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "l2mae":
        loss_fn = L2MAELoss()
    elif loss_name == "atomwisel2":
        loss_fn = AtomwiseL2Loss()
    else:
        raise NotImplementedError(f"Unknown loss function name: {loss_name}")

    return loss_fn
