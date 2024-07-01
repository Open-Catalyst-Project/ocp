---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# AdsorbML tutorial

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
import ase.io
from ase.optimize import BFGS

from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
import os
from glob import glob
import pandas as pd
from fairchem.data.oc.utils import DetectTrajAnomaly
from fairchem.data.oc.utils.vasp import write_vasp_input_files

# Optional - see below
import numpy as np
from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
from x3dase.visualize import view_x3d_n
```

## Enumerate the adsorbate-slab configurations to run relaxations on

+++

AdsorbML incorporates random placement, which is especially useful for more complicated adsorbates which may have many degrees of freedom. I have opted sample a few random placements and a few heuristic. Here I am using *CO on copper (1,1,1) as an example.

```{code-cell} ipython3
bulk_src_id = "mp-30"
adsorbate_smiles = "*CO"

bulk = Bulk(bulk_src_id_from_db = bulk_src_id)
adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles)
slabs = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,1,1))

# There may be multiple slabs with this miller index.
# For demonstrative purposes we will take the first entry.
slab = slabs[0]
```

```{code-cell} ipython3
# Perform heuristic placements
heuristic_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="heuristic")

# Perform random placements
# (for AdsorbML we use `num_sites = 100` but we will use 4 for brevity here)
random_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="random_site_heuristic_placement", num_sites = 4)
```

## Run ML relaxations:

There are 2 options for how to do this.
 1. Using `FAIRChemCalculator` as the calculator within the ASE framework
 2. By writing objects to lmdb and relaxing them using `main.py` in the ocp repo
 
(1) is really only adequate for small stuff and it is what I will show here, but if you plan to run many relaxations, you should definitely use (2). More details about writing lmdbs has been provided [here](../core/lmdb_dataset_creation.md) - follow the IS2RS/IS2RE instructions. And more information about running relaxations once the lmdb has been written is [here](../core/model_training.md).

You need to provide the calculator with a path to a model checkpoint file. That can be downloaded [here](../core/model_checkpoints)

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
import os

checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')

os.makedirs(f"data/{bulk}_{adsorbate}", exist_ok=True)

# Define the calculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path) # if you have a gpu, add `cpu=False` to speed up calculations

adslabs = [*heuristic_adslabs.atoms_list, *random_adslabs.atoms_list]
# Set up the calculator
for idx, adslab in enumerate(adslabs):
    adslab.calc = calc
    opt = BFGS(adslab, trajectory=f"data/{bulk}_{adsorbate}/{idx}.traj")
    opt.run(fmax=0.05, steps=100) # For the AdsorbML results we used fmax = 0.02 and steps = 300, but we will use less strict values for brevity.
```

## Parse the trajectories and post-process

As a post-processing step we check to see if:
1. the adsorbate desorbed
2. the adsorbate disassociated
3. the adsorbate intercalated
4. the surface has changed

We check these because they effect our referencing scheme and may result in erroneous energies. For (4), the relaxed surface should really be supplied as well. It will be necessary when correcting the SP / RX energies later. Since we don't have it here, we will ommit supplying it, and the detector will instead compare the initial and final slab from the adsorbate-slab relaxation trajectory. If a relaxed slab is provided, the detector will compare it and the slab after the adsorbate-slab relaxation. The latter is more correct! Note: for the results in the AdsorbML paper, we did not check if the adsorbate was intercalated (`is_adsorbate_intercalated()`) because it is a new addition.

```{code-cell} ipython3
# Iterate over trajs to extract results
results = []
for file in glob(f"data/{bulk}_{adsorbate}/*.traj"):
    rx_id = file.split("/")[-1].split(".")[0]
    traj = ase.io.read(file, ":")
    
    # Check to see if the trajectory is anomolous
    initial_atoms = traj[0]
    final_atoms = traj[-1]
    atom_tags = initial_atoms.get_tags()
    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anom = (
        detector.is_adsorbate_dissociated()
        or detector.is_adsorbate_desorbed()
        or detector.has_surface_changed()
        or detector.is_adsorbate_intercalated()
    )
    rx_energy = traj[-1].get_potential_energy()
    results.append({"relaxation_idx": rx_id, "relaxed_atoms": traj[-1],
                    "relaxed_energy_ml": rx_energy, "anomolous": anom})
```

```{code-cell} ipython3
df = pd.DataFrame(results)
df
```

```{code-cell} ipython3
#scrap anomalies
df = df[~df.anomolous].copy().reset_index()
```

## (Optional) Deduplicate structures
We may have enumerated very similar structures or structures may have relaxed to the same configuration. For this reason, it is advantageous to cull systems if they are very similar. This results in marginal improvements in the recall metrics we calculated for AdsorbML, so it wasnt implemented there. It is, however, a good way to prevent wasteful VASP calculations. You can also imagine that if we would have enumerated 1000 configs per slab adsorbate combo rather than 100 for AdsorbML, it is more likely that having redundant systems would reduce performance, so its a good thing to keep in mind. This may be done by eye for a small number of systems, but with many systems it is easier to use an automated approach. Here is an example of one such approach, which uses a SOAP descriptor to find similar systems.

```{code-cell} ipython3
# Extract the configs and their energies
def deduplicate(configs_for_deduplication: list,
                adsorbate_binding_index: int,
                cosine_similarity = 1e-3,
               ):
    """
    A function that may be used to deduplicate similar structures.
    Among duplicate entries, the one with the lowest energy will be kept.
    
    Args:
        configs_for_deduplication: a list of ML relaxed adsorbate-
            surface configurations.
        cosine_similarity: The cosine simularity value above which,
            configurations are considered duplicate.
            
    Returns:
        (list): the indices of configs which should be kept as non-duplicate
    """
    
    energies_for_deduplication = np.array([atoms.get_potential_energy() for atoms in configs_for_deduplication])
    # Instantiate the soap descriptor
    soap = SOAP(
        species=np.unique(configs_for_deduplication[0].get_chemical_symbols()),
        r_cut = 2.0,
        n_max=6,
        l_max=3,
        periodic=True,
    )
    #Figure out which index cooresponds to 
    ads_len = list(configs_for_deduplication[0].get_tags()).count(2)
    position_idx = -1*(ads_len-adsorbate_binding_index)
    # Iterate over the systems to get the SOAP vectors
    soap_desc = []
    for config in configs_for_deduplication:
        soap_ex = soap.create(config, centers=[position_idx])
        soap_desc.extend(soap_ex)

    soap_descs = np.vstack(soap_desc)

    #Use euclidean distance to assess similarity
    distance = squareform(pdist(soap_descs, metric="cosine"))

    bool_matrix = np.where(distance <= cosine_similarity, 1, 0)
    # For configs that are found to be similar, just keep the lowest energy one
    idxs_to_keep = []
    pass_idxs = []
    for idx, row in enumerate(bool_matrix):
        if idx in pass_idxs:
            continue
            
        elif sum(row) == 1:
            idxs_to_keep.append(idx)
        else:
            same_idxs = [row_idx for row_idx, val in enumerate(row) if val == 1]
            pass_idxs.extend(same_idxs)
            # Pick the one with the lowest energy by ML
            min_e = min(energies_for_deduplication[same_idxs])
            idxs_to_keep.append(list(energies_for_deduplication).index(min_e))
    return idxs_to_keep
```

```{code-cell} ipython3
configs_for_deduplication =  df.relaxed_atoms.tolist()
idxs_to_keep = deduplicate(configs_for_deduplication, adsorbate.binding_indices[0])
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Flip through your configurations to check them out (and make sure deduplication looks good)
print(idxs_to_keep)
view_x3d_n(configs_for_deduplication[2].repeat((2,2,1)))
```

```{code-cell} ipython3
df = df.iloc[idxs_to_keep]
```

```{code-cell} ipython3
low_e_values = np.round(df.sort_values(by = "relaxed_energy_ml").relaxed_energy_ml.tolist()[0:5],3)
print(f"The lowest 5 energies are: {low_e_values}")
df
```

## Write VASP input files

This assumes you have access to VASP pseudopotentials and the right environment variables configured for ASE. The default VASP flags (which are equivalent to those used to make OC20) are located in `ocdata.utils.vasp`. Alternatively, you may pass your own vasp flags to the `write_vasp_input_files` function as `vasp_flags`. 

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Grab the 5 systems with the lowest energy
configs_for_dft = df.sort_values(by = "relaxed_energy_ml").relaxed_atoms.tolist()[0:5]
config_idxs = df.sort_values(by = "relaxed_energy_ml").relaxation_idx.tolist()[0:5]

# Write the inputs
for idx, config in enumerate(configs_for_dft):
    os.mkdir(f"data/{config_idxs[idx]}")
    write_vasp_input_files(config, outdir = f"data/{config_idxs[idx]}/")
```
