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

# CatTSunami tutorial

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
from fairchem.applications.cattsunami.core import Reaction
from fairchem.data.oc.core import Slab, Adsorbate, Bulk, AdsorbateSlabConfig
from fairchem.core import FAIRChemCalculator
from ase.optimize import BFGS
from x3dase.visualize import view_x3d_n
from ase.io import read
from x3dase.x3d import X3D
from fairchem.applications.cattsunami.databases import DISSOCIATION_REACTION_DB_PATH
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH, BULK_PKL_PATH
from fairchem.core.models.model_registry import model_name_to_local_file
import matplotlib.pyplot as plt
from fairchem.applications.cattsunami.core.autoframe import AutoFrameDissociation
from fairchem.applications.cattsunami.core import OCPNEB
from ase.io import read
from IPython.display import Image

# Optional
# from x3dase.x3d import X3D

# Set random seed
import numpy as np
np.random.seed(22)
```

## Do enumerations in an AdsorbML style for CH dissociation on Ru (001)

To start, we generate placements for the reactant and product species on the surface. We utilize the random placement approach which was developed for AdsorbML, and use an OCP model to relax our placements on the surface. These placements and their ML-determined energies are used as input to the CatTSunami automatic NEB frame generation approach.


```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Instantiate the reaction class for the reaction of interest
reaction = Reaction(reaction_str_from_db="*CH -> *C + *H",
                    reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
                    adsorbate_db_path = ADSORBATE_PKL_PATH)

# Instantiate our adsorbate class for the reactant and product
reactant = Adsorbate(adsorbate_id_from_db=reaction.reactant1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product1 = Adsorbate(adsorbate_id_from_db=reaction.product1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product2 = Adsorbate(adsorbate_id_from_db=reaction.product2_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)

# Grab the bulk and cut the slab we are interested in
bulk = Bulk(bulk_src_id_from_db="mp-33", bulk_db_path=BULK_PKL_PATH)
slab = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(0,0,1))

# Perform site enumeration
# For AdsorbML num_sites = 100, but we use 5 here for brevity. This should be increased for practical use.
reactant_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = reactant,
                                       mode="random_site_heuristic_placement",
                                       num_sites = 5).atoms_list
product1_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product1,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 5).atoms_list
product2_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product2,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 5).atoms_list
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Instantiate the calculator
# NOTE: If you have a GPU, use cpu = False
# NOTE: Change the checkpoint path to locally downloaded files as needed
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
cpu = True
calc = FAIRChemCalculator(checkpoint_path = checkpoint_path, cpu = cpu)
```

### Run ML local relaxations:

There are 2 options for how to do this.
 1. Using `FAIRChemCalculator` as the calculator within the ASE framework
 2. By writing objects to lmdb and relaxing them using `main.py` in the ocp repo

(1) is really only adequate for small stuff and it is what I will show here, but if you plan to run many relaxations, you should definitely use (2). More details about writing lmdbs has been provided [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/tutorials/lmdb_dataset_creation.ipynb) - follow the IS2RS/IS2RE instructions. And more information about running relaxations once the lmdb has been written is [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md#initial-structure-to-relaxed-structure-is2rs).

You need to provide the calculator with a path to a model checkpoint file. That can be downloaded [here](../core/model_checkpoints)

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Relax the reactant systems
reactant_energies = []
for config in reactant_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    reactant_energies.append(config.get_potential_energy())

# Relax the product systems
product1_energies = []
for config in product1_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product1_energies.append(config.get_potential_energy())

product2_energies = []
for config in product2_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product2_energies.append(config.get_potential_energy())
```

## Enumerate NEBs
Here we use the class we created to handle automatic generation of NEB frames to create frames using the structures we just relaxed as input.
![dissociation_scheme](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/applications/cattsunami/tutorial/dissociation_scheme.png)

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
af = AutoFrameDissociation(
            reaction = reaction,
            reactant_system = reactant_configs[reactant_energies.index(min(reactant_energies))],
            product1_systems = product1_configs,
            product1_energies = product1_energies,
            product2_systems = product2_configs,
            product2_energies = product2_energies,
            r_product1_max=2, #r1 in the above fig
            r_product2_max=3, #r3 in the above fig
            r_product2_min=1, #r2 in the above fig
)

nframes = 10
frame_sets, mapping_idxs = af.get_neb_frames(calc,
                               n_frames = nframes,
                               n_pdt1_sites=4, # = 5 in the above fig (step 1)
                               n_pdt2_sites = 4, # = 5 in the above fig (step 2)
                              )
```

## Run NEBs
Here we use the custom child class we created to run NEB relaxations using ML. The class we created allows the frame relaxations to be batched, improving efficiency.

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
## This will run all NEBs enumerated - to just run one, run the code cell below.
# On GPU, each NEB takes an average of ~1 minute so this could take around a half hour on GPU
# But much longer on CPU
# Remember that not all NEBs will converge -- the k, nframes would be adjusted to achieve convergence

# fmax = 0.05 # [eV / ang**2]
# delta_fmax_climb = 0.4
# converged_idxs = []

# for idx, frame_set in enumerate(frame_sets):
#     neb = OCPNEB(
#         frame_set,
#         checkpoint_path=checkpoint_path,
#         k=1,
#         batch_size=8,
#         cpu = cpu,
#     )
#     optimizer = BFGS(
#         neb,
#         trajectory=f"ch_dissoc_on_Ru_{idx}.traj",
#     )
#     conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
#     if conv:
#         neb.climb = True
#         conv = optimizer.run(fmax=fmax, steps=300)
#         if conv:
#             converged_idxs.append(idx)

# print(converged_idxs)
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# If you run the above cell -- dont run this one
fmax = 0.05 # [eV / ang**2]
delta_fmax_climb = 0.4
neb = OCPNEB(
    frame_sets[0],
    checkpoint_path=checkpoint_path,
    k=1,
    batch_size=8,
    cpu = cpu,
)
optimizer = BFGS(
    neb,
    trajectory=f"ch_dissoc_on_Ru_0.traj",
)
conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
if conv:
    neb.climb = True
    conv = optimizer.run(fmax=fmax, steps=300)
```

## (Optional) Visualize the results

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
idx_of_interest = 0
optimized_neb = read(f"n2_dissoc_on_Ru_{idx_of_interest}.traj", ":")[-1*nframes:]
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
es  = []
for frame in optimized_neb:
    frame.set_calculator(calc)
    es.append(frame.get_potential_energy())

# Plot the reaction coordinate

es = [e - es[0] for e in es]
plt.plot(es)
plt.xlabel("frame number")
plt.ylabel("relative energy [eV]")
plt.title(f"CH dissociation on Ru(0001), Ea = {max(es):1.2f} eV")
plt.savefig("CH_dissoc_on_Ru_0001.png")
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Make an interative html file of the optimized neb trajectory
x3d = X3D(optimized_neb)
x3d.write("optimized_neb_ch_disoc_on_Ru0001.html")
```
