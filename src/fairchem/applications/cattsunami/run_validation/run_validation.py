"""
A python script to run a validation of the ML NEB model on a set of NEB calculations.
This script has not been written to run in parallel, but should be modified to do so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.io import read
from ase.optimize import BFGS
import torch
import argparse
from fairchem.core import FAIRChemCalculator 
from fairchem.applications.cattsunami.core.ocpneb import OCPNEB
import os
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import ase


def get_results_sp(df2: pd.DataFrame):
    """
    Get the % success and % convergence for the model considered with
    single points performed on the transition states.

    Args:
        df2 (pd.DataFrame): The dataframe containing the results of the
            NEB calculations.
    Returns:
        (tuple[str]): a tuple of strings containing the % success and
            % convergence
    """
    n_converged = df2[df2.all_converged | (df2.barrierless_ml)].shape[0]
    n_calculated = df2[~(df2.failed_sp)].shape[0]

    n = len(
        df2[((df2.sp_residual <= 0.05) & (df2.both_barriered)) | (df2.both_barrierless)]
    )
    d = len(df2[~(df2.failed_sp)])
    prop_05_uc_f = n * 100 / d

    n = len(
        df2[((df2.sp_residual <= 0.1) & (df2.both_barriered)) | (df2.both_barrierless)]
    )
    d = len(df2[~(df2.failed_sp)])
    prop_1_uc_f = n * 100 / d

    n = len(
        df2[
            ((df2.sp_residual <= 0.05) & (df2.all_converged) & (df2.both_barriered))
            | (df2.barrierless_converged)
        ]
    )
    d = len(df2[df2.all_converged | ((df2.barrierless_ml) & df2.converged_ml)])
    prop_05 = n * 100 / d

    n = len(
        df2[
            ((df2.sp_residual <= 0.1) & (df2.all_converged) & (df2.both_barriered))
            | (df2.barrierless_converged)
        ]
    )
    d = len(df2[df2.all_converged | ((df2.barrierless_ml) & df2.converged_ml)])
    prop_1 = n * 100 / d

    return (
        f"{prop_1:1.1f}%",
        f"{prop_05:1.1f}%",
        f"{prop_1_uc_f:1.1f}%",
        f"{prop_05_uc_f:1.1f}%",
        f"{100*n_converged/n_calculated:1.0f}%",
    )


def get_results_ml(df2):
    """
    Get the % success and % convergence for the model considered with
    just ML energy and force calls.

    Args:
        df2 (pd.DataFrame): The dataframe containing the results of the
            NEB calculations.
    Returns:
        (tuple[str]): a tuple of strings containing the % success and
            % convergence
    """
    n_converged = len(df2[df2.all_converged_ml])
    n_calculated = len(df2)

    n = len(
        df2[((df2.ml_residual <= 0.05) & (df2.both_barriered)) | (df2.both_barrierless)]
    )
    d = len(df2)
    prop_05_uc_f = n * 100 / d

    n = len(
        df2[((df2.ml_residual <= 0.1) & (df2.both_barriered)) | (df2.both_barrierless)]
    )
    d = len(df2)
    prop_1_uc_f = n * 100 / d

    n = len(
        df2[
            ((df2.ml_residual <= 0.05) & (df2.all_converged_ml) & (df2.both_barriered))
            | (df2.barrierless_converged)
        ]
    )
    d = len(df2[df2.all_converged_ml])
    prop_05 = n * 100 / d

    n = len(
        df2[
            ((df2.ml_residual <= 0.1) & (df2.all_converged_ml) & (df2.both_barriered))
            | (df2.barrierless_converged)
        ]
    )
    d = len(df2[df2.all_converged_ml])
    prop_1 = n * 100 / d

    return (
        f"{prop_1:1.1f}%",
        f"{prop_05:1.1f}%",
        f"{prop_1_uc_f:1.1f}%",
        f"{prop_05_uc_f:1.1f}%",
        f"{100*n_converged/n_calculated:1.0f}%",
    )


def all_converged(row, ml=True):
    """
    Dataframe function which makes the job of filtering to get % success cleaner.
    It assesses the convergence.

    Args:
        row: the dataframe row which the function is applied to
        ml: boolean value. If `True` just the ML NEB and DFT NEB convergence are
            considered. If `False`, the single point convergence is also considered.
    Returns:
        bool: whether the system is converged
    """
    if row.converged_ml and row.converged and ml:
        return True
    elif row.converged_ml and row.converged and (not np.isnan(row.E_TS_SP)):
        return True
    return False


def both_barrierless(row):
    """
    Dataframe function which makes the job of filtering to get % success cleaner.
    It assesses if both DFT and ML find a barrierless transition state.

    Args:
        row: the dataframe row which the function is applied to

    Returns:
        bool: True if both ML and DFT find a barrierless transition state, False otherwise
    """
    if row.barrierless_ml and row.barrierless:
        return True
    return False


def both_barriered(row):
    """
    Dataframe function which makes the job of filtering to get % success cleaner.
    It assesses if both DFT and ML find a barriered transition state.

    Args:
        row: the dataframe row which the function is applied to

    Returns:
        bool: True if both ML and DFT find a barriered transition state, False otherwise
    """
    if (not row.barrierless_ml) and (not row.barrierless):
        return True
    return False


def barrierless_converged(row):
    """
    Dataframe function which makes the job of filtering to get % success cleaner.
    It assesses if both DFT and ML find a barrierless, converged transition state.

    Args:
        row: the dataframe row which the function is applied to

    Returns:
        bool: True if both ML and DFT find a barrierless converged transition state,
             False otherwise
    """
    if row.converged and row.converged_ml and row.barrierless and row.barrierless_ml:
        return True
    return False


def is_failed_sp(row):
    """
    Dataframe function which makes the job of filtering to get % success cleaner.
    It assesses if the single point failed.

    Args:
        row: the dataframe row which the function is applied to

    Returns:
        bool: True if ths single point failed, otherwise False
    """
    if not row.barrierless_ml and np.isnan(row.E_TS_SP):
        return True
    return False


def parse_neb_info(neb_frames: list, calc, conv: bool, entry: dict):
    """
    At the conclusion of the ML NEB, this function processes the important
    results and adds them to the entry dictionary.

    Args:
        neb_frames (list[ase.Atoms]): the ML relaxed NEB frames
        calc: the ocp ase Atoms calculator
        conv (bool): whether or not the NEB achieved forces below the threshold within
            the number of allowed steps
        entry (dict): the entry corresponding to the NEB performed
    """
    e_along_traj = []
    for frame in neb_frames:
        frame.calc = calc
        e_along_traj.append(frame.get_potential_energy())
    barrier_height = max(e_along_traj) - e_along_traj[0]
    E_rxn = e_along_traj[-1] - e_along_traj[0]

    if barrier_height <= 0.1 or barrier_height <= E_rxn + 0.1:
        barrierless = True
        ts_idx = None
    else:
        barrierless = False
        ts_idx = e_along_traj.index(max(e_along_traj))
    entry["E_a_ml"] = barrier_height
    entry["E_rxn_ml"] = E_rxn
    entry["converged_ml"] = conv
    entry["barrierless_ml"] = barrierless
    entry["transition_state_idx"] = ts_idx
    return entry


def get_single_point(
    atoms: ase.Atoms, vasp_dir: str, vasp_flags: dict, vasp_command: str
):
    """
    Gets a single point on the atoms passed.

    Args:
        atoms (ase.Atoms): the atoms object on which the single point will be performed
        vasp_dir (str): the path where the vasp files should be written
        vasp_flags: a dictionary of the vasp INCAR flags
        vasp_command (str): the
    """
    with VaspInteractive(
        directory=vasp_dir,
        command=vasp_command,
        **vasp_flags,
    ) as calc:
        atoms.calc = calc
        try:
            e_vasp = atoms.get_potential_energy()
            f_vasp = atoms.get_forces()
        except:
            print(
                "Single point calculation terminated in error. Unfortunately, this is a normal occurance despite the"
                "DFT calculation running fine."
            )
            atoms = read(f"{vasp_dir}/vasprun.xml")
            e_vasp = atoms.get_potential_energy()
            f_vasp = atoms.get_forces()
    return e_vasp, f_vasp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--k")
    parser.add_argument("--fmax")
    parser.add_argument("--output_file_path")
    parser.add_argument("--batch_size")
    parser.add_argument("--delta_fmax_climb")
    parser.add_argument("--mapping_file_path")
    parser.add_argument("--trajectory_path")
    parser.add_argument("--vasp_command", default=None)
    parser.add_argument("--get_ts_sp", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)

    # Unpack arguments
    args = parser.parse_args()
    df = pd.read_pickle(args.mapping_file_path)
    entries = df.to_dict("records")
    checkpoint_path = args.checkpoint_path
    delta_fmax_climb = float(args.delta_fmax_climb)
    k = float(args.k)
    fmax = float(args.fmax)
    calc = FAIRChemCalculator (checkpoint_path=checkpoint_path, cpu=args.cpu)
    model_id = checkpoint_path.split("/")[-1].split(".")[0]
    vasp_command = args.vasp_command
    os.makedirs(f"{args.output_file_path}/{model_id}", exist_ok=True)

    if args.get_ts_sp:
        os.makedirs(f"{args.output_file_path}/{model_id}/vasp_files", exist_ok=True)

    # Iterate over the systems and perform an ML NEB calculation
    for entry in entries:
        file = entry["trajectory_name"]
        try:
            neb_id = entry["neb_id"]
            if not os.path.exists(args.trajectory_path + "/" + file):
                print(f"File {file} not found at {args.trajectory_path}")
            neb_frames = read(args.trajectory_path + "/" + file, index=":")[0:10]

            conv = False
            torch.cuda.empty_cache()

            # Optimize:
            neb = OCPNEB(
                neb_frames,
                checkpoint_path=checkpoint_path,
                k=k,
                batch_size=int(args.batch_size),
                cpu=args.cpu,
            )
            optimizer = BFGS(
                neb,
                trajectory=f"{args.output_file_path}/{model_id}/{neb_id}-k_now.traj",
            )
            conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
            # Climbing image scheme
            if conv:
                neb.climb = True
                conv = optimizer.run(fmax=fmax, steps=300)

            # If single points are to be performed, perform them
            if args.get_ts_sp:
                from vasp_interactive import VaspInteractive
                from fairchem.data.oc.utils.vasp import calculate_surface_k_points

                os.makedirs(
                    f"{args.output_file_path}/{model_id}/vasp_files/{neb_id}",
                    exist_ok=True,
                )
                vasp_dir = f"{args.output_file_path}/{model_id}/vasp_files/{neb_id}"
                VASP_FLAGS = {
                    "ibrion": -1,
                    "nsw": 300,
                    "isif": 0,
                    "isym": 0,
                    "lreal": "Auto",
                    "ediffg": 0.0,
                    "symprec": 1e-10,
                    "encut": 350.0,
                    "ncore": 1,
                    "lcharg": False,
                    "lwave": False,
                    "gga": "RP",
                    "pp": "PBE",
                    "xc": "PBE",
                    "kpts": calculate_surface_k_points(neb_frames[0]),
                }
            entry = parse_neb_info(neb_frames, calc, conv, entry)

            if entry["transition_state_idx"] is not None and args.get_ts_sp:
                os.makedirs(f"{vasp_dir}/ts", exist_ok=True)
                e_ts, f_ts = get_single_point(
                    neb_frames[entry["transition_state_idx"]],
                    f"{vasp_dir}/ts",
                    VASP_FLAGS,
                    args.vasp_command,
                )
                entry["E_TS_SP"] = e_ts
                entry["F_TS_SP"] = f_ts
                entry["transition_state_atoms"] = neb_frames[
                    entry["transition_state_idx"]
                ]
            else:
                entry["E_TS_SP"] = None
        except:
            print(f"Error with {neb_id}")
            entry["E_a_ml"] = np.nan
            entry["E_rxn_ml"] = np.nan
            entry["converged_ml"] = False
            entry["barrierless_ml"] = None
            entry["transition_state_idx"] = np.nan

    # Process the results to get the % success and % convergence
    df = pd.DataFrame(entries)
    df.to_pickle(f"{args.output_file_path}/{model_id}/results.pkl")

    df["all_converged_ml"] = df.apply(all_converged, axis=1)
    df["both_barrierless"] = df.apply(both_barrierless, axis=1)
    df["both_barriered"] = df.apply(both_barriered, axis=1)
    df["ml_residual"] = abs(df["E_a_ml"] - df["Ea"])
    df["barrierless_converged"] = df.apply(barrierless_converged, axis=1)

    (
        conv_success_in_0_1,
        conv_success_in_0_05,
        all_success_in_0_1,
        all_success_in_0_05,
        convergence,
    ) = get_results_ml(df)
    print(
        f"Results for ML calls only:\n% Success within 0.1 eV for converged: {conv_success_in_0_1}\n% Success within 0.05 eV for converged: {conv_success_in_0_05}\n% Success within 0.1 eV for all: {all_success_in_0_1}\n% Success within 0.05 eV for all: {all_success_in_0_05}\n% Convergence: {convergence}"
    )

    if args.get_ts_sp:
        df["sp_residual"] = abs(df["E_TS_SP"] - df["E_raw_TS"])
        df["E_a_sp"] = df["E_TS_SP"] - df["E_raw_initial"]
        df["failed_sp"] = df.apply(is_failed_sp, axis=1)
        df["all_converged"] = df.apply(all_converged, ml=False, axis=1)
        (
            conv_success_in_0_1,
            conv_success_in_0_05,
            all_success_in_0_1,
            all_success_in_0_05,
            convergence,
        ) = get_results_sp(df)

    print(
        f"Results including DFT SPs:\n% Success within 0.1 eV for converged: {conv_success_in_0_1}\n% Success within 0.05 eV for converged: {conv_success_in_0_05}\n% Success within 0.1 eV for all: {all_success_in_0_1}\n% Success within 0.05 eV for all: {all_success_in_0_05}\n% Convergence: {convergence}"
    )
    df.to_pickle(f"{args.output_file_path}/{model_id}/results.pkl")
