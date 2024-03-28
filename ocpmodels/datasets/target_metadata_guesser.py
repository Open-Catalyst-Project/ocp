import logging

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress


def uniform_atoms_lengths(atoms_lens) -> bool:
    # If all of the structures have the same number of atoms, it's really hard to know
    # whether the entries are intensive or extensive, and whether
    # some of the entries are per-atom or not
    return len(set(atoms_lens)) == 1


def target_constant_shape(atoms_lens, target_samples) -> bool:
    # Given a bunch of atoms lengths, and the corresponding samples for the target,
    # determine whether the shape is always the same regardless of atom size
    return len(set([sample.shape for sample in target_samples])) == 1


def target_per_atom(atoms_lens, target_samples) -> bool:
    # Given a bunch of atoms lengths, and the corresponding samples for the target,
    # determine whether the target is per-atom (first dimension == # atoms, others constant)

    # If a sample target is just a number/float/etc, it can't be per-atom
    if len(np.array(target_samples[0]).shape) == 0:
        return False

    first_dim_proportional = all(
        [
            np.array(sample).shape[0] == alen
            for alen, sample in zip(atoms_lens, target_samples)
        ]
    )

    if len(np.array(target_samples[0]).shape) == 1:
        other_dim_constant = True
    else:
        other_dim_constant = (
            len(set([np.array(sample).shape[1:] for sample in target_samples]))
            == 1
        )

    if first_dim_proportional and other_dim_constant:
        return True
    else:
        return False


def target_extensive(atoms_lens, target_samples, threshold: float = 0.2):
    # Guess whether a property is intensive or extensive.
    # We guess by checking whether standard deviation of the per-atom
    # properties capture >20% of the variation in the property
    # Of course, with a small amount of data!

    # If the targets are all the same shapes, we shouldn't be asking if the property
    # is intensive or extensive!
    assert target_constant_shape(
        atoms_lens, target_samples
    ), "The shapes of this target are not constant!"

    # Get the per-atom normalized properties
    try:
        compiled_target_array = np.array(
            [
                sample / atom_len
                for sample, atom_len in zip(atoms_lens, target_samples)
            ]
        )
    except TypeError:
        return False

    # Calculate the normalized standard deviation of each element in the property output
    target_samples_mean = np.mean(compiled_target_array, axis=0)
    target_samples_normalized = compiled_target_array / target_samples_mean

    # If there's not much variation in the per-atom normalized properties,
    # guess extensive!
    extensive_guess = target_samples_normalized.std(axis=0) < (
        threshold * target_samples_normalized.mean(axis=0)
    )
    if extensive_guess.shape == ():
        return extensive_guess
    elif (
        target_samples_normalized.std(axis=0)
        < (threshold * target_samples_normalized.mean(axis=0))
    ).all():
        return True
    else:
        return False


def guess_target_metadata(atoms_len, target_samples):
    example_array = np.array(target_samples[0])
    if example_array.dtype == object or example_array.dtype == str:
        return {
            "shape": None,
            "type": "unknown",
            "extensive": None,
            "units": "unknown",
            "comment": "Guessed property metadata. The property didn't seem to be a numpy array with any numeric type, so we dob't know what to do.",
        }
    elif target_constant_shape(atoms_len, target_samples):
        target_shape = np.array(target_samples[0]).shape

        if uniform_atoms_lengths(atoms_len):
            if atoms_len[0] > 3 and target_per_atom(atoms_len, target_samples):
                target_shape = list(target_samples[0].shape)
                target_shape[0] = "N"
                return {
                    "shape": tuple(target_shape),
                    "type": "per-atom",
                    "extensive": True,
                    "units": "unknown",
                    "comment": "Guessed property metadata. Because all the sampled atoms are the same length, we can't really know if it is per-atom or per-frame, but the first dimension happens to match the number of atoms.",
                }
            else:
                return {
                    "shape": tuple(target_shape),
                    "type": "per-image",
                    "extensive": True,
                    "units": "unknown",
                    "comment": "Guessed property metadata. Because all the sampled atoms are the same length, we can't know if this is intensive of extensive, or per-image or per-frame",
                }

        elif target_extensive(atoms_len, target_samples):
            return {
                "shape": tuple(target_shape),
                "type": "per-image",
                "extensive": True,
                "comment": "Guessed property metadata. It appears to be extensive based on a quick correlation with atom sizes",
            }
        else:
            return {
                "shape": tuple(target_shape),
                "type": "per-image",
                "extensive": False,
                "units": "unknown",
                "comment": "Guess property metadata. It appears to be intensive based on a quick correlation with atom sizes.",
            }
    elif target_per_atom(atoms_len, target_samples):
        target_shape = list(target_samples[0].shape)[1:]
        return {
            "shape": tuple(target_shape),
            "type": "per-atom",
            "extensive": True,
            "units": "unknown",
            "comment": "Guessed property metadata. It appears to be a per-atom property.",
        }
    else:
        return {
            "shape": None,
            "type": "unknown",
            "extensive": None,
            "units": "unknown",
            "comment": "Guessed property metadata. The property was variable across different samples and didn't seem to be a per-atom property",
        }


def guess_property_metadata(atoms_list):
    atoms = atoms_list[0]
    atoms_len = [len(atoms) for atoms in atoms_list]

    targets = {}

    if hasattr(atoms, "info"):
        for key in atoms.info:
            # Grab the property samples from the list of atoms
            target_samples = [
                np.array(atoms.info[key]) for atoms in atoms_list
            ]

            # Guess the metadata
            targets[f"info.{key}"] = guess_target_metadata(
                atoms_len, target_samples
            )

            # Log a warning so the user knows what's happening
            logging.warning(
                f'Guessed metadata for atoms.info["{key}"]: {str(targets[f"info.{key}"])}'
            )
    if hasattr(atoms, "calc") and atoms.calc is not None:
        for key in atoms.calc.results:
            # Grab the property samples from the list of atoms
            target_samples = [
                np.array(atoms.calc.results[key]) for atoms in atoms_list
            ]

            # stress needs to be handled separately in case it was saved in voigt (6, ) notation
            # atoms2graphs will always request voigt=False so turn it into full 3x3
            if key == "stress":
                target_samples = [
                    voigt_6_to_full_3x3_stress(sample)
                    if sample.shape != (3, 3)
                    else sample
                    for sample in target_samples
                ]

            # Guess the metadata
            targets[f"{key}"] = guess_target_metadata(
                atoms_len, target_samples
            )

            # Log a warning so the user knows what's happening
            logging.warning(
                f'Guessed metadata for ASE calculator property ["{key}"]: {str(targets[key])}'
            )

    return targets
